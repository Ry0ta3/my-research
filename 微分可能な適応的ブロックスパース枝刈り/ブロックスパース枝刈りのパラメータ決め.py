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

# ==============================================================================
# 1. 探索するハイパーパラメータ空間を定義
# ==============================================================================
# 色々な値を試せるようにリストで定義
lambda_min_list = [0.03, 0.04, 0.05]
lambda_max_list = [0.07, 0.08, 0.09]
alpha_max_list = [1.0, 1.2, 1.4]
# alpha_min は0で固定

# すべての組み合わせを生成
search_space = list(itertools.product(lambda_min_list, lambda_max_list, alpha_max_list))

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

def Omega(delta):
    h = 1.0
    diff_dim0 = delta[1:, :, :, :] - delta[:-1, :, :, :]
    diff_dim1 = delta[:, 1:, :, :] - delta[:, :-1, :, :]
    diff_dim2 = delta[:, :, 1:, :] - delta[:, :, :-1, :]
    diff_dim3 = delta[:, :, :, 1:] - delta[:, :, :, :-1]
    constrain = torch.sum(torch.abs(diff_dim0)) + torch.sum(torch.abs(diff_dim1)) + torch.sum(torch.abs(diff_dim2)) + torch.sum(torch.abs(diff_dim3))
    return constrain / h

def Ofunction(w_tensor, mu_tensor, delta_tensor, Lambda, alpha):
    term1 = torch.sum((w_tensor - mu_tensor * delta_tensor)**2) / 2
    term2 = Lambda / 2 * (torch.sum(mu_tensor**2) + torch.sum(delta_tensor**2))
    constrain = Omega(delta_tensor)
    return term1 + term2 + alpha * constrain

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


for i, (l_min, l_max, a_max) in enumerate(search_space):
    print("\n" + "="*80)
    print(f"グリッドサーチ試行: {i+1}/{len(search_space)}")
    print(f"パラメータ: lambda_min={l_min}, lambda_max={l_max}, alpha_max={a_max}")
    print("="*80)

    # --- ステップA: モデルを毎回初期化 ---
    model = models.resnet18() 
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(pt_model_path))
    model.to(device)

    # --- ステップB: TV計算とalpha, lambdaの決定 ---
    layer_tvs = {}
    conv_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w_tensor = module.weight.detach().clone().to(device)
            tv = Omega(w_tensor).item()
            relative_tv = tv / (torch.sum(torch.abs(w_tensor)).item())   
            layer_tvs[name] = relative_tv
            conv_layer_names.append(name)

    # 現在のループのハイパーパラメータを使用
    alpha_min = 0
    alpha_max = a_max
    lambda_min = l_min
    lambda_max = l_max
    
    max_tv = max(layer_tvs.values())
    min_tv = min(layer_tvs.values())
    tv_threshold = min_tv + (max_tv - min_tv) / 2
    
    layer_alphas = {}
    layer_lambdas = {}
    for name in conv_layer_names:
        tv = layer_tvs[name]
        if tv >= tv_threshold:
            alpha = alpha_min
            Lambda = lambda_min
        else:
            scale = abs((tv - tv_threshold) / (min_tv - tv_threshold))
            alpha = alpha_min + (alpha_max - alpha_min) * scale
            Lambda = lambda_min + (lambda_max - lambda_min) * (1 / (1 + np.exp(-50 * (scale - 0.5)))) # シグモイド補間
        layer_alphas[name] = alpha
        layer_lambdas[name] = Lambda

        # print(layer_alphas[name], layer_lambdas[name])

    # --- ステップC: 最適化ループ ---
    final_weights_to_assign = {}
    loss_list = {}
    lr_threshold = 5e-8
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"--- レイヤー '{name}' の処理を開始 ---")
            w_tensor = module.weight.detach().clone().to(device)
            w_abs_sqrt = torch.sqrt(torch.abs(w_tensor))
            w_sign = torch.sign(w_tensor)
            mu_tensor = torch.nn.Parameter(w_abs_sqrt.clone())
            delta_tensor = torch.nn.Parameter(w_abs_sqrt.clone())
            optimizer = torch.optim.Adam([mu_tensor, delta_tensor], lr=0.1)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=100)
            loss_list[name] = []
            
            for step in range(20000): # (注意)ループ回数を少し減らしています。必要に応じて調整してください。
                optimizer.zero_grad()
                loss = Ofunction(torch.abs(w_tensor), mu_tensor, delta_tensor, layer_lambdas[name], layer_alphas[name])
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                if step % 100 == 0:
                    # print(f"Step {step}: Loss = {loss.item():.4f}")
                    loss_list[name].append(loss.item())
                    if step >= 100:
                        if np.abs(loss_list[name][int(step/100)]-loss_list[name][int(step/100) - 1]) < 1e-4:
                            print(f"損失の差{np.abs(loss_list[name][int(step/100)]-loss_list[name][int(step/100)-1])}が1e-4を下回ったため、学習を終了します。")
                            break

                current_lr = optimizer.param_groups[0]['lr']   # 今の学習率を取り出す
                
                if current_lr < lr_threshold:
                    print(f"学習率 ({current_lr:.9f}) が閾値 ({lr_threshold}) を下回ったため、学習を終了します。")
                    break

            with torch.no_grad():
                final_weight_tensor = w_sign * mu_tensor * delta_tensor
                final_weights_to_assign[name] = final_weight_tensor.cpu()

    with torch.no_grad():
        for name, module in model.named_modules():
            if name in final_weights_to_assign:
                module.weight.copy_(final_weights_to_assign[name].to(device))

    # --- ステップD: 枝刈りとスパース率の計算 ---
    threshold = 1e-4
    masks = {}
    all_conv_params = 0
    all_conv_zeros = 0
    
    # 枝刈り前に元の密モデルを再度ロード
    pruned_model = models.resnet18()
    pruned_model.fc = nn.Linear(pruned_model.fc.in_features, 10)
    pruned_model.load_state_dict(torch.load(pt_model_path))
    pruned_model.to(device)
    
    # マスクの作成
    for name, module in model.named_modules(): # 最適化後のモデルからマスクを作る
        if isinstance(module, nn.Conv2d):
             weights = module.weight.detach()
             masks[name] = (torch.abs(weights) > threshold).float()

    # マスクの適用
    for name, module in pruned_model.named_modules(): # 密モデルにマスクを適用
        if name in masks:
            prune.CustomFromMask.apply(module, "weight", mask=masks[name].to(device))
            weight_after_pruning = module.weight
            num_zeros = (weight_after_pruning == 0).sum().item()
            total_params = weight_after_pruning.numel()
            print(f"{name} Zero-Ratio: {(100.0*num_zeros/total_params):.4f}")
            print(f"{total_params - num_zeros}")  #これが0ならほんとに100%
            all_conv_params += total_params
            all_conv_zeros += num_zeros
            
    sparsity = 100. * all_conv_zeros / all_conv_params if all_conv_params > 0 else 0
    print(f"全畳み込み層の0率：{sparsity:.2f}%")

    # --- ステップE: 再訓練 ---
    pruned_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pruned_model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=0)
    epoch_loss_list = []
    for epoch in range(15):
        pruned_model.train()
        total_loss = 0.0
        num_train = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = pruned_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*labels.size(0)
            num_train+=labels.size(0)

        epoch_loss_list.append(total_loss/num_train)
        print(f'Epoch[{epoch+1}/15], Loss: {epoch_loss_list[epoch]:.4f}')
        # 次のエポックへの準備
        if epoch >= 1:
            if np.abs(epoch_loss_list[epoch]-epoch_loss_list[epoch-1]) < 0.001:
                break
        scheduler.step()
        # print(f"再訓練 Epoch {epoch+1}/15, Loss: {loss.item():.4f}")
    
    # 永続化
    for name, module in pruned_model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and prune.is_pruned(module):
            prune.remove(module, "weight")

    # --- ステップF: 評価 ---
    final_accuracy = accuracy_test(pruned_model, testloader, device)
    
    onnx_path = f"temp_model_{i}.onnx"
    model_to_onnx(pruned_model, onnx_path)
    inference_speed = run_benchmark(onnx_path, batch_size=400)

    # --- ステップG: 結果の記録 ---
    current_result = {
        "lambda_min": l_min,
        "lambda_max": l_max,
        "alpha_max": a_max,
        "sparsity(%)": sparsity,
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

output_excel_path = "hyperparameter_search_results.xlsx"

# ExcelWriterを使って、複数のシートに書き込む
with pd.ExcelWriter(output_excel_path) as writer:
    df_results.to_excel(writer, sheet_name='全結果', index=False)
    best_accuracy_run.to_excel(writer, sheet_name='正解率ベスト', index=False)
    best_sparsity_run.to_excel(writer, sheet_name='枝刈り率ベスト', index=False)

print(f"\n詳細な結果を '{output_excel_path}' の複数シートに保存しました。")
