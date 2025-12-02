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
import time

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
pt_model_path = "../dense.pt"
# データローダーの準備
transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
trainset = torchvision.datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=400, shuffle=True)
testset = torchvision.datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=400, shuffle=False)


for i in [99, 99.1, 99.2, 99.3, 99.4, 99.5, 99.6]:
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

    # この後に非構造枝刈りを行うのでマスクとモデルのサイズが異なるのを防ぐために永続化（マスクと重みを掛ける）
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, "weight")

    # ゼロ比率を表示
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"{name} Zero-Ratio: {100.0 * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()):.2f}%")

    import torch_pruning as tp

    # 準備ができたモデルをGPU/CPUに送る
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    ####### ADMM後のモデルを用いて枝刈りターゲットを特定

    # 枝刈りしたいモジュール名と、その中で削除するチャネルのインデックスを保存する辞書
    pruning_targets_by_name = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 重みテンソルを取得 (out_channels, in_channels, h, w)
            weight = module.weight.detach()
            
            # フィルター（out_channels次元）ごとに、絶対値の合計を計算
            # dim=(1,2,3) は in_channels, height, width の次元を潰すという意味
            filter_sum = torch.sum(torch.abs(weight), dim=(1, 2, 3))
            filter_ave = filter_sum / (weight.size()[1]*weight.size()[2]*weight.size()[3])
            
            # 合計が0のフィルター（＝全ての重みが0のフィルター）のインデックスを見つける
            indices_to_prune = torch.where((filter_sum <= 0.5) & (filter_ave<=0.005))[0].tolist()   # filter_sum == 0 
            
            # 刈るべきフィルターが1つでもあれば、辞書に記録
            if indices_to_prune:
                pruning_targets_by_name[name] = indices_to_prune
                print(f"レイヤー '{name}': {len(indices_to_prune)}個のフィルターを枝刈り対象として特定。")

    #######

    # 非構造枝刈りのためのmaskを作るためにADMM後のモデルに対し非構造枝刈りを行う
    # 依存関係グラフを構築するためのダミー入力
    # 値はランダムでOK。形状が正しいことが重要。
    example_inputs = torch.randn(1, 3, 224, 224).to(device)

    # ★★★ ここからが、より堅牢なワークフロー ★★★

    # 1. 依存関係グラフを構築 (これは変更なし)
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=example_inputs)

    # 2. 枝刈りグループのリストを作成する
    pruning_groups = []
    for name, indices in pruning_targets_by_name.items():
        # ターゲットのモジュールを取得
        module_to_prune = model.get_submodule(name)
        
        # 3. DG.get_pruning_groupを使って、依存関係を明示的に解決させる
        #    このグループには、conv1だけでなく、bn1なども自動的に含まれる
        group = DG.get_pruning_group(
            module=module_to_prune, 
            pruning_fn=tp.function.prune_conv_out_channels,  
            idxs=indices
        )
        pruning_groups.append(group)
        print(f"レイヤー '{name}' の枝刈りグループを作成: {group}")

    # 4. グループを一つずつ実行する
    for group in pruning_groups:
        # 念のため、グループが有効かチェック
        if DG.check_pruning_group(group):
            group.prune()

    # maskをつくる
    threshold = 0

    masks = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # 現在の重みを取得
            weights = module.weight.detach()
            
            # 閾値に基づいてマスクを作成 (絶対値が0でない要素が1、0であれば0)
            masks[name] = (torch.abs(weights) != threshold).float().to(weights.device)

    # maskをかける
    for name, module in model.named_modules(): 
        if name in masks:
            prune.CustomFromMask.apply(module, "weight", mask=masks[name])

    # 構造的かつ非構造枝刈りしたときのゼロ比率を表示
    all_conv_params = 0
    all_conv_zeros = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"{name} Zero-Ratio: {100.0 * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()):.4f}%")
            weight_after_pruning = module.weight
            num_zeros = (weight_after_pruning == 0).sum().item()
            total_params = weight_after_pruning.numel()
            all_conv_params += total_params
            all_conv_zeros += num_zeros

    sparsity = 100. * all_conv_zeros / all_conv_params if all_conv_params > 0 else 0
    print(f"全畳み込み層の0率：{sparsity:.4f}%")

    # 再訓練
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)   #lr = 0.001で損失0.02とか
    num_epochs = 30  #15
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
    start = time.time()
    final_accuracy = accuracy_test(model, testloader, device)
    gpu_speed = time.time()-start
    
    onnx_path = f"temp_model_{i}.onnx"
    model_to_onnx(model, onnx_path)
    inference_speed = run_benchmark(onnx_path, batch_size=400)

    # --- ステップG: 結果の記録 ---
    current_result = {
        "pruning_sparsity(%)": i,
        "final_sparsity(%)": sparsity,
        "accuracy(%)": final_accuracy,
        "median_speed(ms)": inference_speed,
        "gpu_speed(s)": gpu_speed
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