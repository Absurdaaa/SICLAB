# SICLAB考核

这个仓库目前主要用于 CIFAR-10 生成模型实验，核心内容已经整理到 [code](/Users/linshangjin/Desktop/SICLAB/code/README.md) 目录。当前重点是基于 Consistency Models 做条件生成微调，并扩展了三种类别条件注入方式。

## 仓库结构

- [code](/Users/linshangjin/Desktop/SICLAB/code/README.md)
  - 主实验目录。
  - 基于 OpenAI Consistency Models 的 CIFAR-10 JAX 实现整理。
  - 目前包含条件生成微调、条件采样、条件评估、可视化等完整流程。
- [edm](/Users/linshangjin/Desktop/SICLAB/edm)
  - NVLabs 官方 EDM 代码。
  - 主要用于直接评测官方 `.pkl` 权重。
- [weight](/Users/linshangjin/Desktop/SICLAB/weight)
  - 本地保存的官方权重和相关文件。
- [code_old](/Users/linshangjin/Desktop/SICLAB/code_old)
  - 早期 CIFAR-10 diffusion 训练与评测脚本备份。
  - 仅作历史保留，不再作为当前主流程。

## 当前主要工作

当前主线实验在 [code](/Users/linshangjin/Desktop/SICLAB/code/README.md) 中，新增能力包括：

1. 三种条件生成方式
   - `adagn`
   - `concat`
   - `cross_attn`
2. 三种微调范围
   - `full`
   - `conditional_only`
   - `conditional_plus_head`
3. 配套工具链
   - 条件训练
   - 十类批量采样
   - 分类器条件评估
   - class-wise FID
   - 条件样本网格图生成

详细说明、训练命令和脚本入口请直接查看 [code/README.md](/Users/linshangjin/Desktop/SICLAB/code/README.md)。

## 推荐使用方式

如果你要继续做当前项目，建议只关注 `code/` 目录：

```bash
cd /Users/linshangjin/Desktop/SICLAB/code
```

然后参考 [code/README.md](/Users/linshangjin/Desktop/SICLAB/code/README.md) 中的说明进行：

- 环境安装
- 条件生成微调
- 条件采样
- 条件评估
- 网格图可视化

## 官方 EDM 权重评测

如果你的目标是评测 NVLabs 官方 EDM 权重，而不是跑当前 `code/` 里的 JAX 一致性模型流程，那么建议使用：

- 生成：`edm/generate.py`
- FID：`edm/fid.py`

权重文件位于 [weight](/Users/linshangjin/Desktop/SICLAB/weight) 目录，例如：

- `weight/edm-cifar10-32x32-uncond-vp.pkl`
- `weight/edm-cifar10-32x32-uncond-ve.pkl`
- `weight/baseline-cifar10-32x32-uncond-vp.pkl`
- `weight/baseline-cifar10-32x32-uncond-ve.pkl`

这部分属于独立流程，和 `code/` 目录里的条件生成微调实验不是同一套训练代码。

## 说明

- 根目录 README 只保留仓库总览和入口说明。
- 具体实验细节统一放在 [code/README.md](/Users/linshangjin/Desktop/SICLAB/code/README.md)。
- 如果后续目录继续调整，优先维护 `code/README.md`，根目录 README 只做导航。
