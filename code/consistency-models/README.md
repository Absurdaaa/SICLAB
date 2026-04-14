# Consistency Distillation on CIFAR-10

这个目录实现了一条最小可跑的两阶段流程：

1. 用 `diffusers.UNet2DModel + DDPMScheduler` 训练一个 CIFAR-10 diffusion teacher
2. 用训练好的 teacher 做 consistency distillation，得到一步/少步采样模型

这版实现是“论文思想 + 工程可运行版本”，不是论文 `2303.01469` 的全量复刻。
它保留了核心结构：

- teacher: 标准噪声预测 diffusion
- student: consistency skip/out parameterization
- distillation: 用 teacher 把相邻时间点连接起来，再让 consistency model 保持输出一致

## 依赖

至少需要：

```bash
pip install diffusers
```

如果你的环境里已经有 `torch`、`tqdm`、`numpy`、`Pillow`，就够跑这套代码。

## 训练 teacher

```bash
cd /Users/linshangjin/Desktop/SICLAB/code/consistency-models
python3 teacher_train.py --config teacher_config.json
```

输出目录：

- `outputs_teacher/checkpoints/teacher_epoch_XXXX.pt`
- `outputs_teacher/samples/epoch_XXXX.png`

## 采样 teacher

```bash
python3 sample_teacher.py \
  --checkpoint outputs_teacher/checkpoints/teacher_epoch_0100.pt \
  --num-samples 16 \
  --output outputs_teacher/sample.png
```

## 一致性蒸馏

先确认 `consistency_config.json` 里的 `teacher_checkpoint` 指向已训练好的 teacher checkpoint。

```bash
cd /Users/linshangjin/Desktop/SICLAB/code/consistency-models
python3 distill_train.py --config consistency_config.json
```

输出目录：

- `outputs_consistency/checkpoints/consistency_epoch_XXXX.pt`
- `outputs_consistency/samples/epoch_XXXX.png`

## 采样 consistency model

一步采样：

```bash
python3 sample_consistency.py \
  --checkpoint outputs_consistency/checkpoints/consistency_epoch_0100.pt \
  --num-samples 16 \
  --steps 1 \
  --output outputs_consistency/sample_1step.png
```

少步采样：

```bash
python3 sample_consistency.py \
  --checkpoint outputs_consistency/checkpoints/consistency_epoch_0100.pt \
  --num-samples 16 \
  --steps 4 \
  --output outputs_consistency/sample_4step.png
```

## 实现说明

- `teacher_train.py`
  - 标准 DDPM 训练
  - teacher 预测噪声
- `distill_train.py`
  - 从 batch 图像构造 `x_t`
  - 用 teacher 的 DDIM-style deterministic step 得到相邻较小噪声点 `x_s`
  - 让 `student(x_t, sigma_t)` 拟合 `ema_student(x_s, sigma_s)`
- `modeling.py`
  - `ConsistencyModel` 的 skip/out 参数化
  - teacher 的 DDPM/DDIM scheduler

## 目前的边界

这版没有实现：

- 论文里的完整 continuous-time PF-ODE distillation
- FID/IS 评估脚本

如果后面需要，可以继续往这几个方向补。

## 单机多卡 DDP

teacher 训练：

```bash
torchrun --standalone --nproc_per_node=4 teacher_train.py --config teacher_config.json
```

consistency 蒸馏：

```bash
torchrun --standalone --nproc_per_node=4 distill_train.py --config consistency_config.json
```

## 导出评估图片

teacher：

```bash
torchrun --standalone --nproc_per_node=4 generate_eval_images.py \
  --model-type teacher \
  --checkpoint outputs_teacher/checkpoints/teacher_epoch_0100.pt \
  --output-dir outputs_teacher/eval_images \
  --num-images 10000 \
  --batch-size 100
```

consistency：

```bash
torchrun --standalone --nproc_per_node=4 generate_eval_images.py \
  --model-type consistency \
  --checkpoint outputs_consistency/checkpoints/consistency_epoch_0100.pt \
  --output-dir outputs_consistency/eval_images \
  --num-images 10000 \
  --batch-size 100 \
  --steps 1
```

导出后可以直接用仓库外层已有的 [eval_metrics.py](/Users/linshangjin/Desktop/SICLAB/code/eval_metrics.py) 计算 `FID` 和 `IS`。
