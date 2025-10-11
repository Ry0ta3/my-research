import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import itertools
import pandas as pd
from deepsparse.benchmark.benchmark_model import benchmark_model

# 結果を格納するためのリスト
results_log = []

# ==============================================================================
# 2. 関数定義 (ご提示のコードから変更なし、または戻り値を追加)
# ==============================================================================

# onnxファイルを作成する関数
def model_to_onnx(model, output_file, input_shape=(1, 3, 224, 224)):
    model.to('cpu').eval()
    input_tensor = torch.randn(input_shape)
    torch.onnx.export(model, input_tensor, output_file, verbose=False, input_names=['input'], output_names=['output'])
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


# ==============================================================================
# 3. グリッドサーチのメインループ
# ==============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
pt_model_path = "dense.pt"
# データローダーの準備
transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=400, shuffle=True)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=400, shuffle=False)


for i in [10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 99]:
    print("\n" + "="*80)
    print(f"{i}%枝刈り")
    print("="*80)

    # --- ステップA: モデルを毎回初期化 ---
    model = models.resnet18() 
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(pt_model_path))
    model.to(device)

    # すべての畳み込み層を枝刈り対象に
    parameters_to_prune = [
        (module, "weight") for module in model.modules() if isinstance(module, torch.nn.Conv2d)
    ]

    # 大域的・非構造・強度枝刈り
    cut_rate = i/100
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method = prune.L1Unstructured,
        amount = cut_rate,
    )

    # ゼロ比率を表示
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"{name} Zero-Ratio: {100.0 * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()):.2f}%")

    # 再訓練
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)   #lr = 0.001で損失0.02とか
    num_epochs = 15
    epoch_loss_list = []
    # CosineAnnealingLRを定義
    # T_max: 学習率が半周期で最小値になるまでのエポック数。通常、総エポック数を指定。
    # eta_min: 最小学習率。
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    for epoch in range(num_epochs):
        model.train()
        model = model.to(device)
        total_loss = 0.0
        num_train = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)   # データ代入

            # 重み更新のための手順
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 損失を記録
            total_loss += loss.item()*labels.size(0)
            num_train+=labels.size(0)

        # 出力と記録
        epoch_loss_list.append(total_loss/num_train)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch[{epoch+1}/{num_epochs}], Loss: {epoch_loss_list[epoch]:.4f}, LR: {current_lr:.8f}')


        # 次のエポックへの準備
        if epoch >= 1:
            if np.abs(epoch_loss_list[epoch]-epoch_loss_list[epoch-1]) < 0.001:
                break
        scheduler.step()
    
    # 永続化（マスクと重みを掛ける）
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, "weight")

    # --- ステップF: 評価 ---
    final_accuracy = accuracy_test(model, testloader, device)
    
    onnx_path = f"temp_model_{i}.onnx"
    model_to_onnx(model, onnx_path)
    inference_speed = run_benchmark(onnx_path, batch_size=400)

    # --- ステップG: 結果の記録 ---
    current_result = {
        "sparsity(%)": i,
        "accuracy(%)": final_accuracy,
        "median_speed(ms)": inference_speed
    }
    results_log.append(current_result)

# ==============================================================================
# 4. 全ての試行が完了した後、結果を表示
# ==============================================================================
############################################################################
#pip installの代わり
import subprocess
import sys

# pipコマンドをリスト形式で指定
# 'numpy'の部分をインストールしたいパッケージ名に書き換える
command = [sys.executable, "-m", "pip", "install", "openpyxl"]

# コマンドを実行
subprocess.run(command)
########################################################################

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

# 最も枝刈り率が高い組み合わせ
best_sparsity_run = df_results.loc[df_results['sparsity(%)'].idxmax()]
print("\n--- 枝刈り率が最も高い結果 ---")
print(best_sparsity_run)

output_excel_path = "kyodo_results.xlsx"

# ExcelWriterを使って、複数のシートに書き込む
with pd.ExcelWriter(output_excel_path) as writer:
    df_results.to_excel(writer, sheet_name='全結果', index=False)
    best_accuracy_run.to_excel(writer, sheet_name='正解率ベスト', index=False)
    best_sparsity_run.to_excel(writer, sheet_name='枝刈り率ベスト', index=False)

print(f"\n詳細な結果を '{output_excel_path}' の複数シートに保存しました。")