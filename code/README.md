# CIFAR-10 DDPM

这个目录提供了一个基于 PyTorch 的最小 DDPM 训练骨架，直接读取 `../data/cifar-10-batches-py` 下的 CIFAR-10 官方 Python 批文件。

当前默认配置已经向 EDM 的 CIFAR-10 设置靠拢了一步：更宽的 UNet、更深的残差块、低分辨率 attention、更小的学习率，以及默认开启 AMP，适合多卡 NVIDIA GPU。

## 目录

- `train.py`：训练入口。
- `sample.py`：从训练好的 checkpoint 采样并导出图片网格。
- `show_dataset.py`：导出数据集样本网格图，快速检查数据是否正常。
- `export_real_images.py`：导出 CIFAR-10 真实图片目录，供 FID/IS 使用。
- `generate_eval_images.py`：批量生成图片目录，供 FID/IS 使用。
- `eval_metrics.py`：计算 `FID` 和 `Inception Score`。
- `train_distill.py`：从已训练好的 diffusion teacher 蒸馏一步生成器。
- `sample_distill.py`：从蒸馏后的一步生成器采样。
- `distill_config.json`：一步蒸馏配置。
- `train_config.json`：训练配置。
- `ddpm_cifar/`：模型、扩散过程、数据集和工具函数。

说明：

- `ddpm_cifar/` 现在只保留 diffusion 训练和采样相关代码
- 蒸馏相关代码统一放在 `dmd_cifar/`：
  - `dmd_cifar/config.py`
  - `dmd_cifar/student.py`
  - `dmd_cifar/loss.py`
  - `dmd_cifar/train.py`
  - `dmd_cifar/sample.py`
- 顶层的 `train_distill.py` 和 `sample_distill.py` 只是薄封装，默认从 `dmd_cifar/` 启动。

## 环境

```bash
cd code
# 创建conda环境（Python3，环境名diffusion）
conda create -n diffusion python=3.10 -y
# 激活环境
conda activate diffusion
# 安装依赖
pip install -r requirements.txt
```

你的机器没有 GPU 也可以训练，只是会比较慢。默认 `device` 是 `auto`，没有 CUDA 时会自动回退到 CPU。

如果你有多张 GPU，默认优先走单机 `DDP`。推荐用 `torchrun` 启动，每张卡一个进程，效率比 `DataParallel` 更高。
如果你有 4 张 3090，建议直接从当前默认配置开始跑，再根据显存把 `batch_size` 调到 `256` 或 `512`。

## 训练

```bash
cd code
python3 train.py --config train_config.json
```

单机 4 卡推荐这样启动：

```bash
cd code
torchrun --standalone --nproc_per_node=4 train.py --config train_config.json
```

训练输出会写到 `code/outputs/`：

- `config.json`：实际使用的配置。
- `samples/`：周期性采样图。
- `checkpoints/`：模型权重。

## 采样

```bash
cd code
python3 sample.py --checkpoint outputs/checkpoints/checkpoint_epoch_0005.pt --output outputs/sample.png
```

## 查看数据集

```bash
cd code
python3 show_dataset.py --num-images 16 --output outputs/dataset_preview.png
```

这个脚本会：

- 读取 `../data/cifar-10-batches-py`
- 打印数据集大小、单张图片 shape、数值范围
- 保存一个样本拼图到 `outputs/dataset_preview.png`

## 评估 FID / IS

先导出真实图片目录，默认导出 `test_batch`：

```bash
cd code
python3 export_real_images.py --split test --output-dir outputs/real_test_images
```

再从训练好的 checkpoint 批量生成图片：

```bash
cd code
python3 generate_eval_images.py \
  --checkpoint outputs/checkpoints/checkpoint_epoch_0010.pt \
  --output-dir outputs/generated_eval_images \
  --num-images 10000 \
  --batch-size 100
```

如果你要用单机多卡并行生成，例如 4 卡：

```bash
cd code
torchrun --standalone --nproc_per_node=4 generate_eval_images.py \
  --checkpoint /nfs/tangwenhao/lhp/outputs_teacher/checkpoints/teacher_epoch_0300.pt \
  --output-dir outputs/generated_eval_images \
  --num-images 10000 \
  --batch-size 100
```

这里的 `--batch-size` 是每个进程各自的采样 batch size，不是全局 batch size。

最后计算指标：

```bash
cd code
python3 eval_metrics.py \
  --real-dir outputs/real_test_images \
  --generated-dir /tangwenhao/lhp/SICLAB/code/consistency-models/outputs_teacher/eval_images \
  --batch-size 128 \
  --device cuda:0
```

Inception Score: 7.703035 ± 0.1818404
Frechet Inception Distance: 16.3495
inception_score_mean: 7.703034714135143
inception_score_std: 0.18184039969660493
frechet_inception_distance: 16.349504386597005

说明：

- `FID` 越低越好。
- `Inception Score` 越高越好。
- 更建议把 `test_batch` 作为真实集口径固定下来。
- 如果只想算 `FID`，加 `--no-isc`。

## 一步蒸馏

这套代码还额外提供了一版基于当前 teacher checkpoint 的像素级一步蒸馏骨架。它不是完整复刻 `dmd` 仓库，而是直接利用本项目已经训练好的 diffusion 模型做 teacher。

先确保你已经有 teacher checkpoint，例如：

```bash
code/outputs/checkpoints/checkpoint_epoch_0300.pt
```

训练一步生成器：

```bash
cd code
python3 train_distill.py --config distill_config.json
```

单机 4 卡：

```bash
cd code
torchrun --standalone --nproc_per_node=4 train_distill.py --config distill_config.json
```

采样：

```bash
cd code
python3 sample_distill.py \
  --checkpoint distill_outputs/checkpoints/distill_epoch_0100.pt \
  --output distill_outputs/sample.png
```

这版蒸馏的基本思路是：

- student 从随机噪声一步生成图像
- 对 student 图像重新加噪
- 用 teacher 预测噪声并反推出 `x0`
- 用 teacher 反推得到的 `x0` 去约束 student 输出

## 可调整项

- 如果 CPU 太慢，先把 `timesteps` 改小，比如 `200` 或 `500`。
- 如果显存不足，先把 `base_channels` 从 `128` 降到 `96` 或 `64`，再把 `batch_size` 下调。
- `attention_levels=[1, 2]` 表示在更低分辨率层启用 attention，这比全层 attention 更稳，也更接近 CIFAR-10 上常见配置。
- Linux 多卡环境下可以把 `num_workers` 调到 `8` 或 `16`；macOS 上先保持 `0` 更稳。
- 如果使用双卡或多卡，建议适当增大 `batch_size`，比如从 `64` 提到 `128` 或 `256`，不然加速不会太明显。
- 如果你想临时关闭 DDP，用普通方式直接跑 `python3 train.py --config train_config.json`，或者把 `train_config.json` 里的 `use_ddp` 改成 `false`。
