# CIFAR-10 一致性模型复现与条件生成微调

这个目录基于 OpenAI 的 Consistency Models CIFAR-10 JAX 实现整理，当前重点扩展了“从无条件 student 出发，做条件生成微调”的能力。现在支持三种类别条件注入方式，并提供了配套的训练、采样、评估和可视化脚本。

## 当前新增内容

本次实现主要新增了条件生成微调相关能力：

1. 三种条件注入方式

   - `adagn`：通过类嵌入调制归一化层。
   - `concat`：将类别特征映射到空间特征后，与主干特征拼接。
   - `cross_attn`：在残差块中引入类别条件的交叉注意力。
2. 三种微调范围

   - `full`：全量微调整个 student。
   - `conditional_only`：仅训练条件模块。
   - `conditional_plus_head`：训练条件模块和输出头。
3. 条件采样与评估工具

   - 按 CIFAR-10 十个类别批量采样。
   - 使用分类器评估条件准确率、混淆矩阵和 class-wise FID。
   - 生成按类别汇总的可视化网格图。

## 目录说明

- `configs/`：训练与采样配置。
- `jcm/`：核心训练、采样、模型与损失实现。
- `scripts/`：条件采样、条件评估、速度测试、分类器训练等辅助脚本。
- `run_student_conditional_ft.sh`：条件微调入口。
- `run_conditional_sampling.sh`：对单个 checkpoint 做十类批量条件采样。
- `evaluate_conditional_checkpoint.sh`：对单个 checkpoint 做条件生成评估。
- `make_conditional_class_grid.sh`：把十类样本汇总成网格图。
- `launch.sh`：保留原始 EDM / CD / CT / PD 示例命令。

## 环境安装

建议使用 Python 3.9。

先安装项目依赖：

```bash
pip install -e .
```

当前 `setup.py` 已经包含了 `jax`、`flax`、`dm-haiku`、`optax` 等 Python 依赖，因此不建议继续沿用旧版 README 中写死的 `jaxlib-0.4.7 + CUDA 11 + cuDNN 8.2` 手动安装命令。

更准确的使用方式是：

- 如果你使用 CPU 环境，通常执行 `pip install -e .` 即可。
- 如果你使用 GPU 环境，需要再根据你本机的 Python、CUDA、cuDNN 与驱动版本，额外安装匹配的 JAX GPU 轮子。

也就是说，旧说明中的这段命令：

```bash
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.7+cuda11.cudnn82-cp39-cp39-manylinux2014_x86_64.whl
```

只适用于当时特定的旧环境，不应直接照搬到当前仓库。请以你实际机器环境为准安装对应版本的 JAX。

## 条件生成微调

### 1. 配置文件

- `configs/cifar10_ve_cd_conditional_ft.py`
  - 在原 `cifar10_ve_cd.py` 基础上打开类别条件能力。
- `configs/cifar10_student_conditional_ft.py`
  - 用于“从已有无条件 CD student checkpoint 初始化，再做条件微调”。
  - 默认使用 `DSM` 损失，不依赖 teacher 条件监督。

### 2. 一键运行三种条件方式

最常用入口：

```bash
INIT_CKPT=/path/to/checkpoints/checkpoint_25 \
WORKDIR=/path/to/conditional_ft_runs \
CLASSIFIER_CKPT=/path/to/best_classifier.pt \
bash run_student_conditional_ft.sh
```

该脚本会依次运行：

- `adagn`
- `concat`
- `cross_attn`

若设置了 `WORKDIR=/path/to/root`，输出会自动写到：

- `/path/to/root/adagn`
- `/path/to/root/concat`
- `/path/to/root/cross_attn`

### 3. 常用可调参数

`run_student_conditional_ft.sh` 支持通过环境变量覆盖主要参数：

- `CONFIG`：默认 `configs/cifar10_student_conditional_ft.py`
- `INIT_CKPT`：无条件 student 初始化 checkpoint
- `GPUS`：例如 `0,1,2,3`
- `BATCH_SIZE`
- `LR`
- `N_ITERS`
- `SNAPSHOT_FREQ`
- `NUM_SAMPLES`
- `EVAL_BATCH_SIZE`
- `CLASSIFIER_CKPT`
- `FINETUNE_MODE`

例如只训练条件模块：

```bash
INIT_CKPT=/path/to/checkpoint_25 \
WORKDIR=/path/to/conditional_ft_runs \
FINETUNE_MODE=conditional_only \
bash run_student_conditional_ft.sh
```

或者训练条件模块加输出头：

```bash
INIT_CKPT=/path/to/checkpoint_25 \
WORKDIR=/path/to/conditional_ft_runs \
FINETUNE_MODE=conditional_plus_head \
bash run_student_conditional_ft.sh
```

## 单独训练某一种条件方式

如果你不想一次跑三种方式，也可以直接调用训练入口：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m jcm.main \
  --config configs/cifar10_student_conditional_ft.py \
  --workdir /path/to/adagn_run \
  --mode train \
  --config.training.init_ckpt=/path/to/checkpoint_25 \
  --config.training.finetune_mode=full \
  --config.training.batch_size=128 \
  --config.optim.lr=1e-5 \
  --config.training.n_iters=20000 \
  --config.model.class_conditional=True \
  --config.model.conditioning_type=adagn \
  --config.model.num_classes=10
```

把 `adagn` 改成 `concat` 或 `cross_attn` 即可切换条件方式。

## 条件采样

对某个训练结果做十类批量采样：

```bash
WORKDIR=/path/to/adagn_run \
OUTPUT_ROOT=/path/to/adagn_run/conditional_samples_ckpt_20000 \
CONFIG=configs/cifar10_student_conditional_ft.py \
CONDITIONING_TYPE=adagn \
CKPT=20000 \
NUM_SAMPLES=5000 \
EVAL_BATCH_SIZE=256 \
bash run_conditional_sampling.sh
```

输出目录结构大致为：

```text
conditional_samples_ckpt_20000/
  class_0/
  class_1/
  ...
  class_9/
```

每个类别目录下会保存按 host / round 切分的 `.npz` 样本文件。

## 条件生成评估

对某个 checkpoint 做条件评估：

```bash
WORKDIR=/path/to/adagn_run \
CLASSIFIER_CKPT=/path/to/best_classifier.pt \
CKPT=latest \
bash evaluate_conditional_checkpoint.sh
```

默认会自动查找：

- 样本目录：`WORKDIR/conditional_samples_ckpt_<ckpt>`
- 结果目录：`WORKDIR/conditional_metrics_ckpt_<ckpt>`

评估脚本会输出：

- 每类分类准确率
- 总体混淆矩阵
- 每类 FID
- 可视化网格图
- JSON / CSV 等汇总结果

如果训练脚本中传入了 `CLASSIFIER_CKPT`，`run_student_conditional_ft.sh` 在每个条件方式训练结束后会自动执行采样和评估。

## 条件样本网格图

将十类结果拼成一张总览图：

```bash
WORKDIR_ROOT=/path/to/conditional_ft_runs \
VARIANT=adagn \
CKPT=latest \
bash make_conditional_class_grid.sh
```

支持的 `VARIANT`：

- `adagn`
- `concat`
- `cross_attn`

## 其他保留脚本

- `compare_teacher_student_same_seed.sh`
  - 用相同随机种子对 teacher / student 采样，方便做可视化对比。
- `compare_teacher_student_speed.sh`
  - 测试 teacher 和 student 的采样速度。
- `run_student_4step_pd.sh`
  - Progressive Distillation 相关训练脚本。
- `launch.sh`
  - 保留原始仓库中的 EDM / CD / CT / Progressive Distillation 示例命令。

## 实现说明

条件生成相关实现主要分布在以下位置：

- `jcm/models/ncsnpp.py`
  - 增加 `class_conditional`、`conditioning_type`、`num_classes` 等配置读取。
- `jcm/models/layerspp.py`
  - 实现 `AdaGN / concat / cross_attn` 三种条件注入路径。
- `jcm/losses.py`
  - 增加 `finetune_mode`，支持按参数子集微调。
- `scripts/sample_conditional_all_classes.py`
  - 单次加载模型后，批量生成 10 个类别样本，减少重复加载开销。

## 原始能力

这个仓库仍保留原始一致性模型相关功能，包括：

- EDM 训练与评估
- Consistency Distillation
- Continuous-time Consistency Distillation
- Consistency Training
- Progressive Distillation
- 多步采样与编辑 notebook：`editing_multistep_sampling.ipynb`

## Citation

如果你使用了原始 Consistency Models 方法，请引用：

```bibtex
@article{song2023consistency,
  title={Consistency Models},
  author={Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2303.01469},
  year={2023},
}
```

如果你也参考了 `score_sde` 基础实现，请一并引用其原论文。
