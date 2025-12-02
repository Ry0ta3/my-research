import time
import torch
import torchvision
from torch.utils.data import DataLoader
import torch_pruning as tp
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# 乱数を一定にする
seed = 777  # 好きな数字でOK
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# onnxファイルとptファイルの作成
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

from deepsparse.benchmark.benchmark_model import benchmark_model
# 推論速度ベンチマーク (速度を返すように変更)
def run_benchmark(model_path, batch_size):
    result = benchmark_model(model_path, batch_size=batch_size)
    stats = result["benchmark_result"]
    median_time = stats["median"]
    print(f"推論レイテンシ中央値: {median_time:.2f} ms")
    return median_time

# モデル読み込み
pt_model_path = "../dense.pt"
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 にあわせて出力層変更
model.load_state_dict(torch.load(pt_model_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# データローダーの準備
transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=400, shuffle=True)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=400, shuffle=False)

# 枝刈り前のパラメータ数を取得
all_conv_params = 0
layer_params = {}

for name, module in model.named_modules():
  if isinstance(module, torch.nn.Conv2d):
    print(f"{name} Zero-Ratio: {100.0 * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()):.4f}%")
    weight_pruning = module.weight
    layer_params[name] = module.weight.numel()
    total_params = module.weight.numel()
    all_conv_params += total_params

save_all_conv_params = all_conv_params

# サンプルの入力データ（依存関係の解決に必要）
example_inputs = torch.randn(1, 3, 224, 224)
# 最終目標: 87%削減
final_sparsity = 0.3
# 何回に分けて削るか
iterative_steps = 15

# set network-level sparsity: all layers have a sparsity level of 30%
imp = tp.importance.FPGMImportance(p = 2)
pruner = tp.pruner.MetaPruner(
    model,
    example_inputs, 
    importance=imp, # <--- ここにセット！
    global_pruning = True,
    pruning_ratio=final_sparsity,
    iterative_steps=iterative_steps,
    ignored_layers=[],  
    root_module_types=[torch.nn.Conv2d]
)

# ループで実行
epoch_loss_list = []
optimizer = optim.Adam(model.parameters(), lr = 0.001)   #lr = 0.001で損失0.02とか
criterion = nn.CrossEntropyLoss()
for i in range(iterative_steps):
    pruner.step()

    print(f"Step {i+1}/{iterative_steps}: Pruning done. Now fine-tuning...")
    
    # 1回再訓練
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
    print(f'Epoch[{i+1}/{iterative_steps}], Loss: {epoch_loss_list[i]:.4f}, LR: {current_lr:.8f}')

    # 3. 精度確認
    accuracy_test(model, testloader, device)

    all_conv_params = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            before_weight = layer_params[name]
            after_weight = module.weight.numel()
            all_conv_params += after_weight
            print(f"{name} Zero-Ratio: {100.0 * (1 - float(after_weight) / float(before_weight)):.4f}%")

    sparsity = 100. * (1 - float(all_conv_params) / float(save_all_conv_params)) if save_all_conv_params > 0 else 0
    print(f"全畳み込み層の0率：{sparsity:.4f}%")

#枝刈り後のゼロ比率
all_conv_params = 0

for name, module in model.named_modules():
  if isinstance(module, torch.nn.Conv2d):
    before_weight = layer_params[name]
    after_weight = module.weight.numel()
    all_conv_params += after_weight
    print(f"{name} Zero-Ratio: {100.0 * (1 - float(after_weight) / float(before_weight)):.4f}%")

sparsity = 100. * (1 - float(all_conv_params) / float(save_all_conv_params)) if save_all_conv_params > 0 else 0
print(f"全畳み込み層の0率：{sparsity:.4f}%")
print(model)

# 再訓練
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)   #lr = 0.001で損失0.02とか
num_epochs = 30 #15
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
      print('early stop')
      break
  scheduler.step()

model = model.to('cpu')
model_to_onnx(model, "FGSM_onlyFilterPruning.onnx")
start_time = time.time()
accuracy_test(model, testloader, device)
print(f"testloaderをバッチサイズ400で分けた時のGPUによる全推論が終わるまでの時間：{time.time()-start_time:.4f} sec")
run_benchmark("FGSM_onlyFilterPruning.onnx", batch_size=400)