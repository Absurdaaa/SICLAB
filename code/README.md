# CIFAR-10 DDPM

这个目录提供了一个基于 PyTorch 的最小 DDPM 训练骨架，直接读取 `../data/cifar-10-batches-py` 下的 CIFAR-10 官方 Python 批文件。

## 目录

- `train.py`：训练入口。
- `sample.py`：从训练好的 checkpoint 采样并导出图片网格。
- `show_dataset.py`：导出数据集样本网格图，快速检查数据是否正常。
- `export_real_images.py`：导出 CIFAR-10 真实图片目录，供 FID/IS 使用。
- `generate_eval_images.py`：批量生成图片目录，供 FID/IS 使用。
- `eval_metrics.py`：计算 `FID` 和 `Inception Score`。
- `train_config.json`：训练配置。
- `ddpm_cifar/`：模型、扩散过程、数据集和工具函数。

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

如果你有多张 GPU，默认会在单机上自动启用 `DataParallel`。当前这套代码会把一个 batch 拆到多卡上并行训练。

## 训练

```bash
cd code
python3 train.py --config train_config.json
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

最后计算指标：

```bash
cd code
python3 eval_metrics.py \
  --real-dir outputs/real_test_images \
  --generated-dir outputs/generated_eval_images \
  --batch-size 64 \
  --device cuda:0
```

说明：

- `FID` 越低越好。
- `Inception Score` 越高越好。
- 更建议把 `test_batch` 作为真实集口径固定下来。
- 如果只想算 `FID`，加 `--no-isc`。

## 可调整项

- 如果 CPU 太慢，先把 `timesteps` 改小，比如 `200` 或 `500`。
- 可以把 `base_channels` 从 `64` 降到 `32`，把 `batch_size` 从 `64` 降到 `16` 或 `32`。
- `num_workers` 在 macOS 上先保持 `0`，更稳。
- 如果使用双卡或多卡，建议适当增大 `batch_size`，比如从 `64` 提到 `128` 或 `256`，不然加速不会太明显。
- 如果你想临时关闭多卡，把 `train_config.json` 里的 `use_data_parallel` 改成 `false`。
