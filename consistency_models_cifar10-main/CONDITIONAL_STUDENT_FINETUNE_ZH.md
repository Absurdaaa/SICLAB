# 条件 Student 微调实现说明

这份文档说明当前仓库里新增的“**不依赖 teacher、直接微调 conditional student**”方案是怎么实现的，尽量落到具体代码层面，而不是泛泛讲思路。

这条路线的目标不是严格复现完整的 conditional consistency distillation，而是：

- 复用已有的无条件 `CD student` checkpoint
- 给模型加上类别条件输入
- 在尽量少改现有代码的前提下，继续训练 student

核心原则只有一个：

- **默认无条件流程不能被破坏**

也就是说，如果不显式开启 conditional 配置，原来的训练、采样、评测逻辑应该保持不变。

---

## 1. 整体方案

当前仓库原本的主线是：

- `teacher` 提供参考
- `student` 通过 consistency distillation 学习
- 模型本身只有噪声/时间条件，没有类别条件

现在新增的是一条扩展路线：

1. 保留原有无条件主线
2. 给模型增加**可选**类别条件接口
3. 给 loss 和 sampler 增加**可选**类别标签输入
4. 用现有无条件 student checkpoint 作为初始化
5. 用 `DSM` 继续训练 conditional student

这意味着：

- 这条新路线是 **student-only fine-tune**
- 训练时**不再依赖 teacher 蒸馏**
- 但依然复用了原 student 的参数作为初始化

---

## 2. 改了哪些文件

本次主要改了这些文件：

- `configs/default_cifar10_configs.py`
- `configs/cifar10_ve_cd_conditional_ft.py`
- `configs/cifar10_student_conditional_ft.py`
- `jcm/models/ncsnpp.py`
- `jcm/models/utils.py`
- `jcm/losses.py`
- `jcm/sampling.py`
- `jcm/train.py`
- `run_student_conditional_ft.sh`

下面按文件解释具体改动。

---

## 3. 配置层改动

### 3.1 `configs/default_cifar10_configs.py`

新增了 4 个默认字段：

```python
training.init_ckpt = ""
sampling.class_label = None
model.class_conditional = False
model.num_classes = 10
```

作用分别是：

- `training.init_ckpt`
  用于指定一个已有 checkpoint，作为 conditional student 的初始化来源

- `sampling.class_label`
  控制采样时使用哪个类别

- `model.class_conditional`
  是否启用类别条件

- `model.num_classes`
  类别总数，CIFAR-10 固定为 `10`

为什么要放在默认配置里：

- 这样所有 config 都自动有这几个字段
- 但默认值不会影响老代码

默认行为仍然是：

- `class_conditional=False`
- `class_label=None`
- `init_ckpt=""`

所以旧的无条件流程不会变。

### 3.2 `configs/cifar10_ve_cd_conditional_ft.py`

这份 config 是在原始 `cifar10_ve_cd.py` 基础上加了一层 conditional 开关：

```python
config.model.class_conditional = True
config.model.num_classes = 10
```

它的作用是：

- 保留原来的 CD 配置结构
- 但允许模型接收类别标签

### 3.3 `configs/cifar10_student_conditional_ft.py`

这份 config 是真正给“student-only fine-tune”用的。

它基于上面的 conditional config，再额外指定：

```python
config.training.loss = "dsm"
config.training.finetune = False
config.training.snapshot_sampling = False
```

这里最关键的是：

- `training.loss = "dsm"`

原因是当前仓库里的 `consistency` / `continuous consistency` 训练路径，在 `losses.py` 里仍然会去加载 `ref_model_path`，本质上还是 teacher-based。

如果我们想做真正的 student-only fine-tune，就不能再走那条 teacher 路。

所以这里选了：

- 用已有 student 权重初始化
- 再用 `DSM` 做 conditional 微调

这在方法上不是标准 conditional CD，但在工程上是自洽的，而且最省改动。

---

## 4. 模型改动

### 4.1 `jcm/models/ncsnpp.py`

原来 `NCSNpp` 的输入接口是：

```python
def __call__(self, x, time_cond, train=True):
```

现在改成了：

```python
def __call__(self, x, time_cond, class_labels=None, train=True):
```

注意这里 `class_labels` 是**可选参数**，默认是 `None`。

这保证了：

- 老代码继续调用 `model(x, time_cond, train=True)` 也不会坏
- 新代码可以调用 `model(x, time_cond, class_labels=..., train=True)`

### 4.2 条件注入方式

在 `NCSNpp.__call__()` 里，原来先构造噪声/时间 embedding：

```python
temb = ...
```

然后如果 `conditional=True`，继续走原来的时间嵌入 MLP：

```python
temb = nn.Dense(...)(temb)
temb = nn.Dense(...)(act(temb))
```

现在新增了类别条件逻辑：

```python
if class_conditional and temb is not None and class_labels is not None:
    class_emb = nn.Embed(
        num_embeddings=num_classes,
        features=nf * 4,
        embedding_init=default_initializer(),
    )(class_labels.astype(jnp.int32))
    temb = temb + class_emb
```

这段逻辑的含义是：

1. 先用原来的方式得到时间条件 embedding
2. 如果启用了类别条件，并且确实给了类别标签
3. 用 `nn.Embed` 把类别 id 映射到向量
4. 直接把类别 embedding 加到 `temb` 上

这样设计的好处是：

- 改动最小
- 不需要重写 UNet 主干
- 大部分无条件 student 参数都还能复用

### 4.3 `JointNCSNpp`

`JointNCSNpp` 也同步改了接口：

```python
def __call__(self, x, time_cond, class_labels=None, train=True):
```

然后把 `class_labels` 同时传给：

- `denoiser`
- `distiller`

虽然你当前主线大概率不会直接用到 `JointNCSNpp`，但这里一起改掉，可以避免后续接口不一致。

---

## 5. 模型工具层改动

### 5.1 `jcm/models/utils.py`

这个文件是“模型”和“训练/采样逻辑”之间的桥。

原来整个仓库只考虑了一种条件：

- 时间 / 噪声标签

现在要加类别条件，就必须把这一层补上。

### 5.2 `get_model_fn`

原来：

```python
def model_fn(x, labels, rng=None):
```

现在改成：

```python
def model_fn(x, labels, rng=None, class_labels=None):
```

这里 `labels` 还是原来的噪声/时间条件，`class_labels` 是新增的类别条件。

最关键的兼容策略是：

- 如果 `class_labels is None`
  就走原来的 `model.apply(...)`

- 如果 `class_labels` 不为空
  才显式传入：

```python
model.apply(..., x, labels, class_labels=class_labels, train=...)
```

这保证了老代码完全不需要同步改完才能继续运行。

### 5.3 其它工具函数

下面这些函数都加了 `class_labels` 支持：

- `get_denoiser_fn`
- `get_distiller_fn`
- `get_gaussianizer_fn`
- `get_score_fn`
- `get_denoiser_and_distiller_fn`

例如：

```python
def get_distiller_fn(..., class_labels=None):
```

返回的内部函数也支持：

```python
distiller_fn(x, t, rng=None, class_labels_override=None)
```

这样做的意义是：

- 可以在构造函数时绑定默认类别标签
- 也可以在调用时动态覆盖

这对后面训练和采样都比较方便。

### 5.4 `init_model`

这个函数原来只用 fake input 和 fake label 初始化模型：

```python
model.init(..., fake_input, fake_label)
```

现在如果开启了 `class_conditional=True`，会改成：

```python
model.init(..., fake_input, fake_label, class_labels=fake_label)
```

原因很直接：

- conditional 模型多了 `nn.Embed` 参数
- 初始化时必须走到这条路径，参数树里才会生成 class embedding 的权重

---

## 6. 训练损失改动

### 6.1 `jcm/losses.py`

先加了一个辅助函数：

```python
def _get_class_labels(batch):
    return batch["label"] if "label" in batch else None
```

这样就不需要在每个 loss 里硬编码：

- batch 一定有 label

因为这个仓库的数据 batch 本来就有：

- `image`
- `label`
- `mask`

但为了兼容旧路径，还是统一走这个 helper。

### 6.2 `get_consistency_loss_fn`

原来只取：

```python
x = batch["image"]
```

现在多取：

```python
class_labels = _get_class_labels(batch)
```

然后在 student 前向时：

```python
Ft, new_states = mutils.get_distiller_fn(
    ...,
    class_labels=class_labels,
)(...)
```

目标分支 `Ft2` 也同样传入：

```python
Ft2, new_states = mutils.get_distiller_fn(
    ...,
    class_labels=class_labels,
)(...)
```

这意味着：

- 同一个样本的类别标签，会同时作用在两条 consistency 分支上

### 6.3 `get_progressive_distillation_loss_fn`

同样补了：

```python
class_labels = _get_class_labels(batch)
```

然后传给：

- `get_denoiser_fn(...)`
- `target_denoiser_fn`

虽然你当前 student-only 路线不一定会直接用这条 loss，但保持接口一致更稳。

### 6.4 `get_continuous_consistency_loss_fn`

这里有一个细节：

原来函数一开始就创建了：

```python
score_fn = mutils.get_score_fn(...)
```

但 `class_labels` 是每个 batch 才知道的，所以不能在函数外层提前绑定。

现在改成：

1. 在 `loss_fn(...)` 里取：

```python
class_labels = _get_class_labels(batch)
```

2. 再在 batch 内部构造：

```python
score_fn = mutils.get_score_fn(..., class_labels=class_labels)
```

3. student 分支同样传入：

```python
mutils.get_distiller_fn(..., class_labels=class_labels)
```

这样 continuous consistency 路径也能吃类别标签。

---

## 7. 采样改动

### 7.1 `jcm/sampling.py`

新增了一个 helper：

```python
def _get_class_labels(config, rng, batch_size):
```

逻辑是：

- `config.sampling.class_label is None`
  返回 `None`

- `config.sampling.class_label == "random"`
  从 `0 ~ num_classes-1` 随机采样类别

- 否则
  生成一个固定类别 id 的 batch

例如固定生成 airplane：

```python
class_labels = jnp.full((batch_size,), 0, dtype=jnp.int32)
```

### 7.2 接到哪些 sampler 上

目前接到了：

- `get_onestep_sampler`
- `get_multistep_sampler`
- `get_seeded_sampler`

这是够用的，因为你当前 conditional student 最主要就是走这些 student sampler。

例如 `get_onestep_sampler` 现在会先取：

```python
class_labels = _get_class_labels(config, rng, shape[0])
```

再传给：

```python
mutils.get_distiller_fn(..., class_labels=class_labels)
```

这样采样时就能按类别生成。

### 7.3 为什么没继续大改 `heun/euler`

这次没有把所有 sampler 全部改成条件版本，原因是：

- 你当前扩展路线是 student-only fine-tune
- 主要关心的是 `onestep / multistep`
- 继续把整套 teacher sampler 全改一遍，会扩大改动面

所以这里只优先打通 student 路线。

---

## 8. 训练初始化改动

### 8.1 `jcm/train.py`

这里新增了两段逻辑：

- `_merge_compatible_tree`
- `_maybe_initialize_from_checkpoint`

### 8.2 为什么需要这一步

现在有一个结构不一致问题：

- 旧无条件 student checkpoint 里**没有** class embedding 参数
- 新 conditional model 里**有** class embedding 参数

如果直接完整 restore：

- 很容易 shape mismatch
- 或者参数树不匹配

### 8.3 怎么解决

做法是“**只加载兼容参数**”。

`_merge_compatible_tree` 的逻辑是：

1. 如果是字典节点，就递归看 key
2. 如果是张量节点，就比较 shape
3. shape 一致才用旧 checkpoint 的值覆盖
4. 否则保留新模型初始化出来的值

这正好适合当前场景：

- 原 student 主干参数大多能匹配，直接继承
- 新增的 class embedding 因为旧 checkpoint 没有，保留随机初始化

### 8.4 什么时候触发

只有满足两个条件才会触发：

```python
int(state.step) == 0 and config.training.init_ckpt
```

也就是说：

- 如果是一个全新 workdir，从 step 0 开始，并且显式给了 `init_ckpt`
  就会做 student 初始化

- 如果是已有训练中断后恢复
  仍然优先从 `checkpoints-meta` 正常续跑

这保证了：

- 老的恢复逻辑不被破坏
- 新的 conditional student 可以从无条件 student 接着起步

---

## 9. 新增配置和脚本

### 9.1 `configs/cifar10_ve_cd_conditional_ft.py`

这份配置只负责：

- 打开类别条件
- 保留原始 CD 配置的结构

适合作为 conditional 基础配置。

### 9.2 `configs/cifar10_student_conditional_ft.py`

这份配置是真正给“直接微调 student”用的。

它明确指定：

```python
config.training.loss = "dsm"
config.training.finetune = False
config.training.snapshot_sampling = False
```

这表示：

- 不再走 teacher-based consistency distillation
- 而是走 student-only conditional DSM 微调

### 9.3 `run_student_conditional_ft.sh`

新增了独立 bash 启动脚本：

- `run_student_conditional_ft.sh`

默认参数是：

```bash
WORKDIR=/nfs/tangwenhao/lhp/cd-conditional-student-ft
INIT_CKPT=/nfs/tangwenhao/lhp/cd-lpips/checkpoints/checkpoint_21
GPUS=0,1,2,3
BATCH_SIZE=128
LR=1e-5
N_ITERS=20000
```

它实际运行的是：

```bash
python -m jcm.main \
  --config configs/cifar10_student_conditional_ft.py \
  --workdir "${WORKDIR}" \
  --mode train \
  --config.training.init_ckpt="${INIT_CKPT}" \
  ...
```

也就是：

1. 初始化 conditional model
2. 从旧 student checkpoint 加载兼容权重
3. class embedding 保持随机初始化
4. 继续做 conditional DSM 微调

---

## 10. 这条路线和标准 conditional CD 的区别

这套实现不是标准的：

- conditional teacher -> conditional student distillation

而是：

- unconditional student checkpoint -> conditional student fine-tune

优点：

- 改动小
- 不需要重新训练 conditional teacher
- 更适合你现在的工程节奏

缺点：

- 方法上不如 teacher-based conditional CD 标准
- 条件可控性和生成质量不一定最好

所以要把它定位成：

- **工程上可行的扩展路线**

而不是：

- **严格论文复现**

---

## 11. 目前还差什么

到现在为止，已经打通了：

- 模型可选类别条件
- loss 可选读取 `batch["label"]`
- student sampler 可选按类别采样
- 从旧 student checkpoint 初始化 conditional student
- 独立 config 和 bash 训练入口

还可以继续补的内容有：

1. 条件采样脚本
   例如一次性生成 10 个类别的网格图

2. conditional eval 说明
   比如怎么指定：

   ```bash
   --config.sampling.class_label=3
   ```

3. 更完整的 student-only 训练监控
   比如把类别采样可视化固定到某几个 checkpoint

---

## 12. 总结

这次实现的核心思路是：

- **把 conditional 能力做成可选基础设施**
- **默认不影响现有无条件代码**
- **通过 `init_ckpt` 复用旧 student 权重**
- **通过 `DSM` 避免再次依赖 teacher**

如果只看工程实现链路，可以概括成：

1. `default config` 增加 conditional 相关开关
2. `NCSNpp` 增加 class embedding
3. `models/utils.py` 让 `class_labels` 能穿透到模型
4. `losses.py` 从 batch 里把 `label` 送进去
5. `sampling.py` 让采样器支持固定/随机类别
6. `train.py` 支持从无条件 student checkpoint 部分初始化
7. 新增 config 和 bash，形成可运行入口

这就是当前仓库里“直接微调 conditional student”的完整实现方案。
