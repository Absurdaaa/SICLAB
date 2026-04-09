# CIFAR-10 DDPM

这个目录提供了一个基于 PyTorch 的最小 DDPM 训练骨架，直接读取 `../data/cifar-10-batches-py` 下的 CIFAR-10 官方 Python 批文件。

## 目录

- `train.py`：训练入口。
- `sample.py`：从训练好的 checkpoint 采样并导出图片网格。
- `train_config.json`：训练配置。
- `ddpm_cifar/`：模型、扩散过程、数据集和工具函数。

## 环境

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

你的机器没有 GPU 也可以训练，只是会比较慢。默认 `device` 是 `auto`，没有 CUDA 时会自动回退到 CPU。

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

## 可调整项

- 如果 CPU 太慢，先把 `timesteps` 改小，比如 `200` 或 `500`。
- 可以把 `base_channels` 从 `64` 降到 `32`，把 `batch_size` 从 `64` 降到 `16` 或 `32`。
- `num_workers` 在 macOS 上先保持 `0`，更稳。
