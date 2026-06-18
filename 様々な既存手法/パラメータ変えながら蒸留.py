import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import itertools
from deepsparse.benchmark.benchmark_model import benchmark_model
import pandas as pd
import time

# ===================================================================
# === 1. モデル定義 ===
# ===================================================================

class SimpleCNN(nn.Module):
    """
    知識蒸留のための生徒モデル。
    パラメータ数を約12万に調整済み。
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        channels = 35 # この値でパラメータ数を約12万に調整
        
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear((channels * 4), num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def count_parameters(model):
    """モデルの訓練可能なパラメータ数を計算するヘルパー関数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_conv_pruning_stats(model):
    total_conv_elements = 0
    nonzero_conv_elements = 0

    for name, module in model.named_modules():
        # 畳み込み層のみを抽出
        if isinstance(module, nn.Conv2d):
            # 重み (weight) の集計
            w = module.weight
            total_conv_elements += w.numel()
            nonzero_conv_elements += torch.count_nonzero(w).item()

            # バイアス (bias) がある場合はそれも集計
            if module.bias is not None:
                b = module.bias
                total_conv_elements += b.numel()
                nonzero_conv_elements += torch.count_nonzero(b).item()

    if total_conv_elements == 0:
        print("モデル内に畳み込み層が見つかりませんでした。")
        return

    reduction_rate = 100 * (1 - nonzero_conv_elements / total_conv_elements)
    remaining_rate = 100 * (nonzero_conv_elements / total_conv_elements)

    print("--- 畳み込み層限定の統計 ---")
    print(f"全要素数 (Total):    {total_conv_elements:,}")
    print(f"非ゼロ要素数 (Active): {nonzero_conv_elements:,}")
    print(f"削減率 (Sparsity):     {reduction_rate:.4f}%")
    print(f"生存率 (Remaining):    {remaining_rate:.4f}%")

# ===================================================================
# === 2. 損失関数の定義 ===
# ===================================================================

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """知識蒸留のための損失関数"""
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    ) * (T * T)

    hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
    
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    return total_loss

# onnxファイルを作成する関数
def model_to_onnx(model, output_file, input_shape = (1, 3, 224, 224)):
    model.eval()
    input_tensor = torch.randn(input_shape).to('cpu')
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(model, input_tensor, output_file, verbose = False, input_names = input_names, output_names = output_names)
    return output_file

# 評価関数 (正解率を返すように変更)
def accuracy_test(model, test_loader, device):
    model.eval()
    model = model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'正解率: {accuracy:.2f}%')
    return accuracy

# 推論速度ベンチマーク (速度を返すように変更)
def run_benchmark(model_path, batch_size):
    result = benchmark_model(model_path, batch_size=batch_size)
    stats = result["benchmark_result"]
    median_time = stats["median"]
    print(f"推論レイテンシ中央値: {median_time:.2f} ms")
    return median_time

# ===================================================================
# === 3. メインの実行ブロック ===
# ===================================================================

if __name__ == '__main__':
    # --- 3a. 基本設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3b. データセットとデータローダーの準備 ---
    print("Preparing dataset...")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4) # バッチサイズやワーカー数は環境に合わせて調整
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=400, shuffle=False)

    # --- 3c. 教師モデルと生徒モデルの準備 ---
    print("Preparing models...")
    
    # 教師モデル (訓練済みのResNet-18)
    teacher_model_path = "../dense.pt"
    teacher_model = models.resnet18(weights=None)
    teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 10)
    try:
        teacher_model.load_state_dict(torch.load(teacher_model_path))
    except FileNotFoundError:
        print(f"Error: Teacher model weights file not found at '{teacher_model_path}'")
        print("Please make sure the pre-trained dense model file exists.")
        exit()
        
    teacher_model.to(device)
    teacher_model.eval()

    alpha_list = [0.3]
    T_list = [2, 4, 6, 8, 10]
    search_space = list(itertools.product(alpha_list, T_list))
    # 結果を格納するためのリスト
    results_log = []

    for i, (num_alpha, num_T) in enumerate(search_space):
        # 生徒モデル (これから訓練するSimpleCNN)
        student_model = SimpleCNN(num_classes=10).to(device)

        # パラメータ数の表示
        print("-" * 50)
        print(f"Teacher model (ResNet-18) parameters: {count_parameters(teacher_model):,}")
        print(f"Student model (SimpleCNN) parameters : {count_parameters(student_model):,}")
        print(f"Parameter ratio: {count_parameters(student_model) / count_parameters(teacher_model) * 100:.2f}%")
        print(f"alpha = {num_alpha}")
        print(f"T = {num_T}")
        print("-" * 50)
        calculate_conv_pruning_stats(student_model)

        # --- 3d. 訓練の設定 ---
        num_epochs = 40
        epoch_loss_list = []
        optimizer = optim.Adam(student_model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        
        # 蒸留ハイパーパラメータ
        T = num_T      # 温度
        alpha = num_alpha  # Hard Lossの割合


        
        print("Starting distillation training...")
        # --- 3e. 訓練ループ ---
        for epoch in range(num_epochs):
            student_model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # 生徒モデルの出力を計算
                student_outputs = student_model(inputs)
                
                # 教師モデルの出力を計算 (勾配は不要)
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                    
                # 蒸留損失を計算
                loss = distillation_loss(
                    student_logits=student_outputs,
                    teacher_logits=teacher_outputs,
                    labels=labels,
                    T=T,
                    alpha=alpha
                )
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            # エポックごとの損失と学習率を表示
            epoch_loss = running_loss / len(trainloader)
            epoch_loss_list.append(epoch_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}")
            
            scheduler.step()

        # 損失グラフ作成
        plt.plot(epoch_loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        print("Distillation training finished.")
        
        # --- 3f. 訓練済み生徒モデルの保存 ---
        student_model = student_model.to('cpu')
        onnx_path = "distilled_student40.onnx"
        model_to_onnx(student_model, onnx_path)

        # --- ステップG: 評価 ---
        start = time.time()
        final_accuracy = accuracy_test(student_model, testloader, device)
        gpu_time = time.time()-start
        inference_speed = run_benchmark(onnx_path, batch_size=400)

        # --- ステップH: 結果の記録 ---
        current_result = {
            "T": T,
            "alpha": alpha,
            "accuracy(%)": final_accuracy,
            "median_speed(ms)": inference_speed,
            "gpu_time(s)": gpu_time
        }
        results_log.append(current_result)

# ==============================================================================
# 4. 全ての試行が完了した後、結果を表示
# ==============================================================================

print("\n" + "#"*80)
print("グリッドサーチが完了しました。")
print("#"*80)

# pandas DataFrameで見やすく表示
df_results = pd.DataFrame(results_log)
print(df_results)

# 最も正解率が高い組み合わせ
best_accuracy_run = df_results.loc[df_results['accuracy(%)'].idxmax()]
print("\n--- 正解率が最も高い結果 ---")
print(best_accuracy_run)

output_excel_path = "distillation_alpha_search_results.xlsx"

# ExcelWriterを使って、複数のシートに書き込む
with pd.ExcelWriter(output_excel_path) as writer:
    df_results.to_excel(writer, sheet_name='全結果', index=False)
    best_accuracy_run.to_excel(writer, sheet_name='正解率ベスト', index=False)

print(f"\n詳細な結果を '{output_excel_path}' の複数シートに保存しました。")