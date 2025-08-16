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

# 乱数を一定にする
seed = 777  # 好きな数字でOK
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# 1. 探索するハイパーパラメータ空間を定義
# ==============================================================================
# 色々な値を試せるようにリストで定義
K1_rating_list = [1, 1.0, 1]

# すべての組み合わせを生成
search_space = list(itertools.product(K1_rating_list))

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

def project_l1_ball(tensor_x: np.ndarray, radius: float) -> np.ndarray:
    """
    テンソル `tensor_x` を、中心が原点のL1ノルム半径 `radius` の超球（L1ボール）の表面または内部に射影する。
    これは、制約条件 ||z||_1 <= radius を満たす、最も `tensor_x` に近い点 `z` を見つける操作。
    アルゴリズムは、"Projection onto the L1-ball" として知られる効率的な手法に基づいている。
    """
    # ステップ 1: 入力テンソルのL1ノルムを計算し、そもそも射影が必要かを確認
    # もしL1ノルムが既に半径以下なら、テンソルは制約を満たしており、何もしなくてよい。
    if np.sum(np.abs(tensor_x)) <= radius:
        return tensor_x

    # ステップ 3: 符号情報を無視し、絶対値のテンソルを作成
    # 射影アルゴリズムは、正の象限（すべての要素が非負）で動作し、最後に元の符号を復元する。
    u_tensor = np.abs(tensor_x)

    # ステップ 4: 絶対値テンソル `u_tensor` を降順にソートする。
    # このソートが、効率的なアルゴリズムの鍵となる。
    u_sorted = np.sort(u_tensor, axis=None)[::-1]
    n_features = len(u_sorted)

    # ステップ 5: ソートされたベクトルの累積和を計算する。
    # sv[i-1] は、u の中で最も大きい i 個の要素の和を表す。
    sv = np.cumsum(u_sorted)

    # ステップ 6: 射影の閾値 `theta` を見つけるための条件を満たす `rho` を探す。
    # ここがアルゴリズムの核心部分。
    # 数学的な導出により、射影後のベクトルは `y = sign(v) * max(abs(v) - theta, 0)` という形になることが分かっている。
    # この `y` のL1ノルムが `radius` に等しくなるような `theta` を見つけたい。
    # そのための条件が `u_sorted[i-1] > (sv[i-1] - radius) / i` であり、
    # この条件を満たす最大のインデックスが `rho` となる。
    
    # i = 1, 2, ..., n_features を生成
    arange_i = np.arange(1, n_features + 1)
    
    # 条件を満たすインデックスの候補を探す
    rho_candidates = np.where(u_sorted > (sv - radius) / arange_i)[0]
    
    # 候補の中から最大のインデックスをrhoとして採用する（0-basedから1-basedに変換するため+1）
    # 候補がない場合は、すべての要素が同じだけシフトされるケース
    rho = rho_candidates[-1] + 1 if len(rho_candidates) > 0 else n_features

    # ステップ 7: 決定した `rho` を使って、最終的なシフト量 `theta` を計算する。
    # thetaは、各要素から（絶対値の形で）引き算される値。
    theta = (sv[rho - 1] - radius) / rho

    # ステップ 8: ソフトしきい値処理（soft-thresholding）を実行して、射影を完了させる。
    # 1. 各要素の絶対値から `theta` を引く。
    # 2. 結果が負になった場合は0にする（`np.maximum(..., 0)`）。
    # 3. 元の符号 `sign(v)` を掛け合わせて、正しい象限に戻す。
    projected_tensor_x = np.sign(tensor_x) * np.maximum(np.abs(tensor_x) - theta, 0)

    # ステップ 9: 返す。
    return projected_tensor_x

def hard_threshold(tensor_x: np.ndarray, K0: int) -> np.ndarray:
    """
    テンソル `tensor_x` に対してハードしきい値処理を適用する。
    これは、制約条件 ||z||_0 <= K0 を満たす、最も `tensor_x` に近い点 `z` を見つける操作。
    簡単に言うと、テンソルの要素のうち、絶対値が大きい方から `K0` 個だけを残し、
    それ以外をすべて厳密に0にする。
    """
    # ステップ 1: そもそも処理が必要かを確認
    # もし残したい非ゼロ要素の数 `K0` が、テンソルの全要素数以上なら、
    # どの要素も0にする必要はないので、そのまま返す。
    if K0 >= tensor_x.size:
        return tensor_x
    
    # ステップ 3: 0にすべき要素を決定するための「しきい値」を見つける。
    # しきい値は、ベクトル `v` の絶対値を小さい順にソートしたとき、
    # 後ろから `K0` 番目（つまり、絶対値が `K0` 番目に大きい要素）の値となる。
    # 例: v=[10, -2, 5, -8], K0=2 の場合、abs(v)=[10, 2, 5, 8]。
    #    ソートすると [2, 5, 8, 10]。後ろから2番目は `8`。これがしきい値となる。
    # 同じ値の要素があった場合は非ゼロの要素がK0
    threshold = np.sort(np.abs(tensor_x), axis=None)[-(K0+1)]

    # ステップ 4: しきい値に基づいて、値を0にする。
    # 絶対値がこのしきい値より「小さい」すべての要素を0で上書きする。
    # 例: abs(v) < 8 となるのは `2` と `5`。対応する元の要素 `-2` と `5` が0になる。
    tensor_x[np.abs(tensor_x) <= threshold] = 0

    return tensor_x

def D_op_4d(tensor: np.ndarray) -> np.ndarray:
    """
    4階数テンソル (out_ch, in_ch, h, w) の4方向の差分を計算し、
    5階数テンソル (out_ch, in_ch, h, w, 4) として返す。
    """
    # dim=0 (出力チャネル方向)
    diff_dim0 = np.roll(tensor, -1, axis=0) - tensor
    # dim=1 (入力チャネル方向)
    diff_dim1 = np.roll(tensor, -1, axis=1) - tensor
    # dim=2 (縦方向)
    diff_dim2 = np.roll(tensor, -1, axis=2) - tensor
    # dim=3 (横方向)
    diff_dim3 = np.roll(tensor, -1, axis=3) - tensor
    
    # 最後の軸に4つの差分をスタックする
    return np.stack([diff_dim0, diff_dim1, diff_dim2, diff_dim3], axis=-1)

# 転置差分がどんな演算なのかいまいちわかってない
def DT_op_4d(stacked_tensor: np.ndarray) -> np.ndarray:
    """
    D_op_4dの転置演算。5階数テンソルを受け取り、4階数テンソルを返す。
    """
    # 各差分を取り出す
    d0 = stacked_tensor[..., 0]
    d1 = stacked_tensor[..., 1]
    d2 = stacked_tensor[..., 2]
    d3 = stacked_tensor[..., 3]

    # 各々の転置差分を計算し、合計する (D^T = -D_backward)
    term0 = np.roll(d0, 1, axis=0) - d0
    term1 = np.roll(d1, 1, axis=1) - d1
    term2 = np.roll(d2, 1, axis=2) - d2
    term3 = np.roll(d3, 1, axis=3) - d3
    
    return term0 + term1 + term2 + term3

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

from scipy.sparse.linalg import cg, LinearOperator # LinearOperatorをインポート

def solve_admm(
    w: np.ndarray, 
    D_op, 
    DT_op, 
    K1: float, 
    K0_ratio: float, 
    rho: float, 
    gamma: float,
    n_iter: int,
):
    # --- 初期化 ---
    # 全ての変数を float64 で厳密に初期化
    x  = w.copy().astype(np.float64)
    y = D_op(x).astype(np.float64)
    z = x.copy().astype(np.float64)
    u1 = np.zeros_like(y, dtype=np.float64)
    u2 = np.zeros_like(z, dtype=np.float64)
    K0 = int(K0_ratio * w.size)
    
    # テンソルの総要素数を計算 (LinearOperatorのshape定義のため)
    n_elements = w.size

    print(f"Tensor shape: {w.shape}, K1={K1}, K0={K0} ({K0_ratio*100:.1f}%)")

    # ★★★ ここからが大幅な変更点 ★★★

    # 1. matvec関数を定義 (入力と出力が1次元ベクトルであることを明示)
    def matvec_for_lo(p_flat: np.ndarray) -> np.ndarray:
        # 内部で型変換は行わず、入力の型を信頼する
        p = p_flat.reshape(w.shape)
        DTD_p = DT_op(D_op(p))
        result = rho * DTD_p + gamma * p + p
        return result.flatten()

    # 2. LinearOperatorを明示的に作成
    # これにより、shapeとdtypeをcg関数に正確に伝える
    A = LinearOperator(
        shape=(n_elements, n_elements), # 作用素は (N, N) の正方行列として振る舞う
        matvec=matvec_for_lo,
        dtype=np.float64               # 扱うデータ型は float64 であることを明記
    )

    # ★★★ 変更ここまで ★★★


    # --- ADMM反復ループ ---
    for i in range(n_iter):
        # 1. x の更新
        # v は float64 であることを確認済み
        v = w + rho * DT_op(y - u1) + gamma * (z - u2)
        
        # LinearOperatorオブジェクト A を直接cgに渡す
        x_flat, _ = cg(A, v.flatten(), x0=x.flatten())
        x = x_flat.reshape(w.shape)

        # 2. y の更新
        v_proj1 = D_op(x) + u1
        y = project_l1_ball(v_proj1, K1)

        # 3. z の更新
        v_proj2 = x + u2
        z = hard_threshold(v_proj2, K0)

        # 4. 双対変数 u1, u2 の更新
        u1 = u1 + D_op(x) - y
        u2 = u2 + x - z
        
        if i % 10 == 0:
            recon_error = np.linalg.norm(x - w) / np.linalg.norm(w)
            print(f"Iteration {i+1}/{n_iter}, Reconstruction Error: {recon_error:.4f}")

    return x
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

import json
# JSONファイルから辞書を読み込む
with open('90cut_data.json', 'r') as f:
    zero_ratings = json.load(f)

for i, (k1_rating, ) in enumerate(search_space):
    print("\n" + "="*80)
    print(f"グリッドサーチ試行: {i+1}/{len(search_space)}")
    print(f"パラメータ: k1_rating={k1_rating}")
    print("="*80)

    # --- ステップA: モデルを毎回初期化 ---
    model = models.resnet18() 
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(pt_model_path))
    model.to(device)

    layer_tvs = {}
    layer_relative_tvs = {}
    conv_layer_names = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w_tensor = module.weight.detach().clone().to(device)
            tv = Omega(w_tensor).item()
            ave_tv = tv/(w_tensor).numel()
            relative_tv = tv / torch.sum(torch.abs(w_tensor)).item()
            layer_tvs[name] = tv
            layer_relative_tvs[name] = relative_tv
            conv_layer_names.append(name)
            print(f"{name}：TV={tv:.4f}     ave_TV={ave_tv:.4f}     relative_TV={relative_tv:.4f}")

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"\n--- レイヤー '{name}' の処理を開始 ---")

            # wを4階テンソルのまま取得
            # .detach().clone()で勾配情報を切り離し、安全なコピーを作成し、numpyに変換、scipyのcg法に合うようにfloat64に
            w_numpy = module.weight.detach().clone().to('cpu').numpy().astype(np.float64)

            # --- パラメータ設定と実行 ---
            K1_val = layer_tvs[name]*k1_rating       # Total Variationの上限
            K0_ratio_val = 1 - zero_ratings[name]   # 10%を非ゼロにする (目標のスパース率90%)
            rho_val = 1.0        # block性ペナルティパラメータ
            gamma_val = 15.0   # sparse性ペナルティパラメータ
            iterations = 30   # 繰り返し回数

            # ADMMソルバーを実行
            x_final = solve_admm(
                w=w_numpy,
                D_op=D_op_4d,        
                DT_op=DT_op_4d,      
                K1=K1_val,
                K0_ratio=K0_ratio_val,
                rho=rho_val,
                gamma=gamma_val,
                n_iter=iterations
            )
            with torch.no_grad():
                module.weight.copy_(torch.tensor(x_final))

            # 閾値で0に近い値を完全な0にする
            x_final[np.abs(x_final) <= 1e-4] = 0

            # 結果の簡単な確認
            print("\n結果:")
            #print(x_final)
            print(f"元の重みテンソルの非ゼロの要素数: {np.sum(w_numpy != 0)}")
            print(f"ADMM後の重みテンソルの非ゼロの要素数: {np.sum(x_final != 0)} (目標数{int(K0_ratio_val * w_numpy.size)})")
            print(f"||Dx||_1: {np.sum(np.abs(D_op_4d(x_final))):.4f} (制約は{K1_val}以下)")

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

    final_layer_tvs = {}
    final_layer_relative_tvs = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w_tensor = module.weight.detach().clone().to(device)
            tv = Omega(w_tensor).item()
            ave_tv = tv/(w_tensor).numel()
            relative_tv = tv / torch.sum(torch.abs(w_tensor)).item()
            final_layer_tvs[name] = tv
            final_layer_relative_tvs[name] = relative_tv
            print(f"{name}：TV={tv:.4f}     ave_TV={ave_tv:.4f}     relative_TV={relative_tv:.4f}")

    # --- ステップG: 結果の記録 ---
    current_result = {
        "layer_tvs": layer_tvs,
        "final_layer_tvs": final_layer_tvs,
        "layer_relative_tvs": layer_relative_tvs,
        "final_layer_relative_tvs": final_layer_relative_tvs,
        "k1_rating": k1_rating,
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

output_excel_path = "ADMM_hyperparameter_search_results.xlsx"

# ExcelWriterを使って、複数のシートに書き込む
with pd.ExcelWriter(output_excel_path) as writer:
    df_results.to_excel(writer, sheet_name='全結果', index=False)
    best_accuracy_run.to_excel(writer, sheet_name='正解率ベスト', index=False)
    best_sparsity_run.to_excel(writer, sheet_name='枝刈り率ベスト', index=False)

print(f"\n詳細な結果を '{output_excel_path}' の複数シートに保存しました。")
