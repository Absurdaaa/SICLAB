# SICLAB

这个仓库里目前有三块主要内容：

- [code](/Users/linshangjin/Desktop/SICLAB/code)：我这边搭的 CIFAR-10 diffusion 训练与评测脚本
- [edm](/Users/linshangjin/Desktop/SICLAB/edm)：NVLabs 官方 EDM 代码
- [weight](/Users/linshangjin/Desktop/SICLAB/weight)：下载好的官方 `.pkl` 权重

## 官方 EDM 权重评测流程

下面这套流程用于测试官方权重，例如：

- `weight/edm-cifar10-32x32-uncond-vp.pkl`
- `weight/edm-cifar10-32x32-uncond-ve.pkl`
- `weight/baseline-cifar10-32x32-uncond-vp.pkl`
- `weight/baseline-cifar10-32x32-uncond-ve.pkl`

注意：

- 官方 `.pkl` 权重不能直接用 [code/generate_eval_images.py](/Users/linshangjin/Desktop/SICLAB/code/generate_eval_images.py)
- 生成图片要用 [edm/generate.py](/Users/linshangjin/Desktop/SICLAB/edm/generate.py)
- 标准 FID 更推荐用 [edm/fid.py](/Users/linshangjin/Desktop/SICLAB/edm/fid.py)
- `IS` 可以用 [code/eval_metrics.py](/Users/linshangjin/Desktop/SICLAB/code/eval_metrics.py)

## 1. 先导出真实测试集图片

CIFAR-10 官方测试集本来就是 `10000` 张，所以这里导出 `10000` 张是正常的。

```bash
cd /Users/linshangjin/Desktop/SICLAB/code
python3 export_real_images.py --split test --output-dir outputs/real_test_images
```

如果你想导出训练集 `50000` 张：

```bash
cd /Users/linshangjin/Desktop/SICLAB/code
python3 export_real_images.py --split train --output-dir outputs/real_train_images
```

## 2. 用官方 EDM 权重生成图片

先以 `edm-cifar10-32x32-uncond-vp.pkl` 为例。

### 快速测试版

先生成 `1000` 张，方便检查流程是否通：

```bash
cd /Users/linshangjin/Desktop/SICLAB/edm

torchrun --standalone --nproc_per_node=4 generate.py \
  --outdir=/Users/linshangjin/Desktop/SICLAB/code/outputs/edm_eval_images_small \
  --seeds=0-999 \
  --subdirs \
  --steps=18 \
  --network=/Users/linshangjin/Desktop/SICLAB/weight/edm-cifar10-32x32-uncond-vp.pkl
```

这里的 `--seeds=0-999` 表示一共生成 `1000` 张图。  
`--steps=18` 是 EDM 官方在 CIFAR-10 32x32 上推荐的采样设置。

### 正式评测版

如果你想更接近官方口径，生成 `50000` 张：

```bash
cd /Users/linshangjin/Desktop/SICLAB/edm

torchrun --standalone --nproc_per_node=4 generate.py \
  --outdir=/Users/linshangjin/Desktop/SICLAB/code/outputs/edm_eval_images \
  --seeds=0-49999 \
  --subdirs \
  --steps=18 \
  --network=/Users/linshangjin/Desktop/SICLAB/weight/edm-cifar10-32x32-uncond-vp.pkl
```

## 3. 计算官方 FID

更推荐直接用 EDM 官方 `fid.py`：

```bash
cd /Users/linshangjin/Desktop/SICLAB/edm

torchrun --standalone --nproc_per_node=4 fid.py calc \
  --images=/Users/linshangjin/Desktop/SICLAB/code/outputs/edm_eval_images \
  --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
```

说明：

- 这条命令更接近 EDM 官方论文和 README 的评测口径
- 正式 FID 建议用 `50000` 张生成图
- `1000` 张只适合 smoke test，不适合和论文数字直接对比

## 4. 计算 Inception Score 和目录版 FID

如果你想继续用本仓库 [code](/Users/linshangjin/Desktop/SICLAB/code) 里的脚本测 `IS`，可以这样：

```bash
cd /Users/linshangjin/Desktop/SICLAB/code

python3 eval_metrics.py \
  --real-dir outputs/real_test_images \
  --generated-dir outputs/edm_eval_images_small \
  --batch-size 64 \
  --device cuda:0
```

正式版：

```bash
cd /Users/linshangjin/Desktop/SICLAB/code

python3 eval_metrics.py \
  --real-dir outputs/real_test_images \
  --generated-dir outputs/edm_eval_images \
  --batch-size 64 \
  --device cuda:0
```

说明：

- `FID` 越低越好
- `Inception Score` 越高越好
- 这里算出来的 `FID` 是“真实图片目录 vs 生成图片目录”的版本
- 如果你要和官方 EDM 结果更严格对齐，优先看上面的 `edm/fid.py`

## 5. 更换其他官方权重

只需要替换命令中的 `--network` 和输出目录，例如：

```bash
/Users/linshangjin/Desktop/SICLAB/weight/edm-cifar10-32x32-uncond-ve.pkl
/Users/linshangjin/Desktop/SICLAB/weight/baseline-cifar10-32x32-uncond-vp.pkl
/Users/linshangjin/Desktop/SICLAB/weight/baseline-cifar10-32x32-uncond-ve.pkl
```

建议每个权重使用不同输出目录，避免混淆。
