# DMD 跑通指南

这份文档对应仓库里的 [dmd](/Users/linshangjin/Desktop/SICLAB/dmd) 项目，目标是先把 **CIFAR-10 上的 DMD 训练流程跑通**。

适用场景：

- 你现在只做 `CIFAR-10`
- 你想先跑通原始 `DMD`
- 暂时不碰更复杂的 [DMD2](/Users/linshangjin/Desktop/SICLAB/DMD2)

## 1. 先理解这个项目在做什么

[dmd](/Users/linshangjin/Desktop/SICLAB/dmd) 是一个基于 EDM teacher 的 **原始 DMD 非官方复现**，README 里已经说明它主要聚焦 CIFAR-10。

这个项目训练时用到三样核心东西：

- `teacher model`：EDM 预训练模型
- `distillation dataset`：提前准备好的 CIFAR-10 蒸馏数据，格式是 `.hdf5`
- `DMD generator`：你要训练出来的一步生成器

训练入口在：

- [dmd/dmd/train.py](/Users/linshangjin/Desktop/SICLAB/dmd/dmd/train.py)
- CLI 入口在 [dmd/dmd/__main__.py](/Users/linshangjin/Desktop/SICLAB/dmd/dmd/__main__.py)

## 2. 创建环境

项目自带环境文件 [dmd/environment.yml](/Users/linshangjin/Desktop/SICLAB/dmd/environment.yml)。

```bash
cd /Users/linshangjin/Desktop/SICLAB/dmd
conda env create -f environment.yml -n dmd
conda activate dmd
```

如果环境已经存在：

```bash
conda activate dmd
```

建议先检查 CLI 是否正常：

```bash
cd /Users/linshangjin/Desktop/SICLAB/dmd
python -m dmd --help
```

## 3. 下载蒸馏数据和参考模型

项目自带下载脚本 [dmd/scripts/download_data.sh](/Users/linshangjin/Desktop/SICLAB/dmd/scripts/download_data.sh)。

它会下载：

- `data/cifar.hdf5`
- `data/cifar_toy.hdf5`
- `models/dmd_cifar_10_cond.pt`

执行：

```bash
cd /Users/linshangjin/Desktop/SICLAB/dmd
bash scripts/download_data.sh
```

下载完成后，你至少应该看到：

```bash
ls data
ls models
```

如果你只想先 smoke test，可以先用：

- `data/cifar_toy.hdf5`

如果你想正式训练，用：

- `data/cifar.hdf5`

## 4. Teacher 模型怎么选

`dmd` 训练需要一个 teacher EDM 模型。

项目 README 默认给的是在线 URL：

```text
https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
```

你可以直接这么用，也可以改成你本地的权重路径。

如果你之前已经在仓库根目录下载过 EDM 权重，推荐直接用本地路径，避免重复下载。例如：

```text
/Users/linshangjin/Desktop/SICLAB/weight/edm-cifar10-32x32-cond-vp.pkl
```

注意：

- `dmd` 这里更适合用 **cond** 版 teacher
- 不是 `uncond`

## 5. 最小跑通命令

先用 toy 数据检查流程：

```bash
cd /Users/linshangjin/Desktop/SICLAB/dmd

python -m dmd train \
  --model-path /Users/linshangjin/Desktop/SICLAB/weight/edm-cifar10-32x32-cond-vp.pkl \
  --data-path /Users/linshangjin/Desktop/SICLAB/dmd/data/cifar_toy.hdf5 \
  --output-dir /Users/linshangjin/Desktop/SICLAB/dmd/outputs/toy_run \
  --epochs 1 \
  --batch-size 32
```

如果你本地没有 `cond` teacher，也可以先用在线 URL：

```bash
cd /Users/linshangjin/Desktop/SICLAB/dmd

python -m dmd train \
  --model-path https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
  --data-path /Users/linshangjin/Desktop/SICLAB/dmd/data/cifar_toy.hdf5 \
  --output-dir /Users/linshangjin/Desktop/SICLAB/dmd/outputs/toy_run \
  --epochs 1 \
  --batch-size 32
```

这个命令的目标不是训好，而是确认：

- 环境没问题
- 数据能读
- teacher 能加载
- 训练循环能跑
- checkpoint 和中间图能写出来

## 6. 正式训练命令

如果 toy 版跑通，再切完整数据：

```bash
cd /Users/linshangjin/Desktop/SICLAB/dmd

python -m dmd train \
  --model-path /Users/linshangjin/Desktop/SICLAB/weight/edm-cifar10-32x32-cond-vp.pkl \
  --data-path /Users/linshangjin/Desktop/SICLAB/dmd/data/cifar.hdf5 \
  --output-dir /Users/linshangjin/Desktop/SICLAB/dmd/outputs/cifar_run \
  --epochs 20 \
  --batch-size 56 \
  --eval-batch-size 128 \
  --num-workers 10
```

如果显存不够，优先改：

- `--batch-size 32`
- `--eval-batch-size 64`

## 7. 训练时会输出什么

从 [dmd/dmd/train.py](/Users/linshangjin/Desktop/SICLAB/dmd/dmd/train.py) 看，训练时会做这些事：

- 训练 generator 和 `mu_fake`
- 用 CIFAR-10 test set 计算 FID
- 周期性保存 checkpoint
- 保存训练中的可视化图片

输出目录就是你传的 `--output-dir`，例如：

```text
/Users/linshangjin/Desktop/SICLAB/dmd/outputs/cifar_run
```

重点关注：

- checkpoint 文件
- 日志
- 周期性保存的图
- `test_fid`

## 8. FID 是怎么算的

[dmd/dmd/fid.py](/Users/linshangjin/Desktop/SICLAB/dmd/dmd/fid.py) 的实现是：

- 真实图：直接用 CIFAR-10 test set
- 假图：用当前 generator 生成
- 然后用 InceptionV3 提取特征再算 FID

这意味着：

- 不需要你手动再导出真实图目录
- `dmd` 训练时会自己在测试集上算 FID

## 9. 训练完成后怎么生成图片

README 里给了最简单的方式：

```python
from dmd.generate import DMDGenerator

gen = DMDGenerator(network_path="/path/to/model_checkpoint.pt")
samples = gen.generate_batch(seeds=list(range(25)), class_ids=0)
```

如果你后面要，我可以再单独帮你补一个 `dmd_generate_example.py`。

## 10. 常见问题

### 1. 我只有 uncond teacher，能不能先跑？

理论上你可以试，但这个仓库 README 明确说它重点是 **CIFAR-10 conditioned model**。  
所以建议优先用：

```text
edm-cifar10-32x32-cond-vp.pkl
```

### 2. 我只想先检查训练链路，不关心结果

直接用：

```text
data/cifar_toy.hdf5
```

### 3. 我只做 CIFAR-10，还需要 DMD2 吗？

不用。  
先把 [dmd](/Users/linshangjin/Desktop/SICLAB/dmd) 跑通更合理。

### 4. 输出里 FID 很差正常吗？

前期正常。  
重点先看能不能稳定训练、FID 是否大致下降。

## 11. 推荐顺序

建议你按这个顺序来：

1. `conda activate dmd`
2. `python -m dmd --help`
3. `bash scripts/download_data.sh`
4. 用 `cifar_toy.hdf5` 跑 1 epoch
5. 确认输出目录里有 checkpoint 和图片
6. 再切 `cifar.hdf5` 做正式训练

如果你愿意，我下一步可以继续帮你写：

- `DMD 故障排查.md`
- 或者一个一键启动脚本 `run_dmd_cifar.sh`
