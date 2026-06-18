import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from deepsparse.benchmark.benchmark_model import benchmark_model
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time
import numpy as np

# ===================================================================
# === 1. モデルのクラス定義 ===
# ===================================================================
# ★重要★
# 保存された重みをロードするためには、その重みが元々属していたモデルと
# 全く同じアーキテクチャを持つクラスが定義されている必要があります。
# distillation.pyで使ったSimpleCNNクラスを、そのままここにコピーします。

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        channels = 35 # distillation.pyで設定した値と必ず同じにする
        
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

# cpuでの推論速度テスト
def pretty_benchmark(model_path, batch_size):
    result = benchmark_model(model_path, batch_size=batch_size)
    stats = result["benchmark_result"]

    print(f"モデル: {model_path}")
    print(f"バッチサイズ: {batch_size}")
    print(f"推論速度: {stats['items_per_sec']:.2f} items/sec")
    print(f"推論レイテンシ中央値: {stats['median']:.2f} ms")
    print(f"レイテンシ平均: {stats['mean']:.2f} ms")
    print(f"レイテンシ標準偏差: {stats['std']:.2f} ms")
    print(f"実行回数: {stats['iterations']} 回")

# GPUによる正解率テスト（cpuでもgpuでも正解か不正解かは変わらない）
def accuracy_test(model, test_loader, device):
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        all_labels = []
        all_predictions = []

        for images, labels in test_loader:
            images = images.to(device)   # モデルに入力
            outputs = model(images)   # モデル出力を受け取る
            predicted = torch.max(outputs.data, 1)[1]   # 予測したラベル
            predicted = predicted.to('cpu')
            predicted = predicted.numpy()

            all_labels.extend(labels)   # 正解ラベル格納
            all_predictions.extend(predicted)   # 予測したラベル格納

        # 正解率の計算
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        cm = confusion_matrix(all_labels, all_predictions)

        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')  # annot=True
        plt.xlabel('Predicted label') # X軸ラベル
        plt.ylabel('True label') # Y軸ラベル
        plt.title('Confusion Matrix') # タイトル
        plt.show()

        CAnumber=0
        for i in range(cm.shape[0]):
            CAnumber+=cm[i, i]
        print(f'正解率:{CAnumber/cm.sum()*100:.2f}%')
# ===================================================================
# === 2. メインの実行ブロック ===
# ===================================================================

if __name__ == '__main__':
    # --- 2a. 基本設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "distilled_student.pt"

    # --- 2b. モデルの「器」を作成 ---
    print(f"Loading model architecture: SimpleCNN")
    # まず、重みが入っていない空のモデルオブジェクトを作成します
    model = SimpleCNN(num_classes=10)

    # --- 2c. 保存された重みをロード ---
    print(f"Loading weights from: {model_path}")
    try:
        # load_state_dictを使って、ファイルから読み込んだ重みをモデルに流し込みます
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit()

    # --- 2d. モデルを評価モードにし、デバイスに送る ---
    model.to(device)
    model.eval() # 評価モードに設定（DropoutやBatchNormの挙動が変わる）
    
    print("Model loaded successfully!")
    
    # (オプション) ロードしたモデルのパラメータ数を確認
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {count_parameters(model):,}")

    # 乱数を一定にする
    seed = 777  # 好きな数字でOK
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # テストデータセット作成
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])   # 正規化したほうがいいらしい

    testset = torchvision.datasets.CIFAR10(root = "./data", train = False, download = True, transform = transform)
    testloader = DataLoader(testset, batch_size = 400, shuffle = False)

    onnx_model_path = "distilled_student.onnx"
    pretty_benchmark(onnx_model_path, batch_size=400)
    start_time = time.time()
    accuracy_test(model, testloader, device)
    print(f'GPUによる推論時間{time.time() - start_time}')