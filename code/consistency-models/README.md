# Consistency Distillation on CIFAR-10

这个目录实现了一条最小可跑的两阶段流程：

1. 用 `diffusers.UNet2DModel + DDPMScheduler` 训练一个 CIFAR-10 diffusion teacher
2. 用训练好的 teacher 做 consistency distillation，得到一步/少步采样模型

这版实现是“更接近论文 `2303.01469` 的工程版本”，仍然不是逐行复刻，但已经补上了原先最关键的几处缺口：

- teacher: 标准噪声预测 diffusion
- student: continuous-time consistency skip/out parameterization
- distillation: 用 Karras 风格 sigma 调度构造相邻噪声对，再让 consistency model 保持输出一致
- schedule: 近似论文的 `N(k)` / `mu(k)` 风格 bins 与 EMA 调度
- loss: 支持 `LPIPS + L1`

## 依赖

至少需要：

```bash
pip install diffusers torchmetrics
```

如果你的环境里已经有 `torch`、`tqdm`、`numpy`、`Pillow`，再补 `torchmetrics` 就能启用更接近论文的 LPIPS 蒸馏损失。没有这个依赖时，代码会自动退回像素损失。

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
  - 用 Karras 风格连续 sigma 网格构造相邻 `(sigma_t, sigma_s)`
  - 用 teacher 的 DDIM-style deterministic step 近似 PF-ODE 邻接点
  - 用论文风格 EMA 调度，让 `student(x_t, sigma_t)` 拟合 `ema_student(x_s, sigma_s)`
  - 默认使用 `LPIPS + L1`
- `modeling.py`
  - `ConsistencyModel` 的 skip/out 参数化
  - 连续 sigma 调度与 teacher timestep 映射

## 目前的边界

这版仍然没有实现：

- 真正连续 teacher solver 下的完整 PF-ODE 积分
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
  --checkpoint /nfs/tangwenhao/lhp/outputs_teacher/checkpoints/teacher_epoch_0300.pt \
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
