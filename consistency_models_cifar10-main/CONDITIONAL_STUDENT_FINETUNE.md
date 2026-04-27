# Conditional Student Fine-Tune

This note documents the "teacher-free conditional student fine-tune" path that was added on top of the original CIFAR-10 consistency-model codebase.

The goal is not to reproduce the full conditional distillation pipeline. The goal is to reuse an existing unconditional CD student checkpoint, add class-conditioning with minimal code changes, and continue training the student directly.

## What This Path Does

The original codebase assumes:

- a diffusion/EDM teacher is available
- consistency distillation uses `training.ref_model_path`
- the default `ncsnpp` model is noise-conditioned, not class-conditioned

The new path adds:

- optional class-conditioning in the model
- optional class labels in model utility functions
- conditional labels flowing through the main student loss paths
- a student-only fine-tuning config and launcher
- partial checkpoint initialization from an unconditional student checkpoint

This keeps the original unconditional code path intact by default.

## Design Principle

The implementation was done with one constraint:

- if `class_conditional=False`, the original training/eval/sampling behavior should stay unchanged

Because of that, every new conditional input is optional.

## Files Changed

### 1. `configs/default_cifar10_configs.py`

Added new default fields:

```python
training.init_ckpt = ""
sampling.class_label = None
model.class_conditional = False
model.num_classes = 10
```

Purpose:

- `training.init_ckpt`
  allows initializing a new model from an existing student checkpoint
- `sampling.class_label`
  allows class-controlled sampling later
- `model.class_conditional`
  switches class-conditioning on or off
- `model.num_classes`
  defines CIFAR-10 class count

These defaults preserve unconditional behavior because:

- `class_conditional=False`
- `class_label=None`
- `init_ckpt=""`

### 2. `jcm/models/ncsnpp.py`

The core model signature was extended from:

```python
def __call__(self, x, time_cond, train=True):
```

to:

```python
def __call__(self, x, time_cond, class_labels=None, train=True):
```

The class-conditioning logic is intentionally minimal:

1. compute the original noise/time embedding `temb`
2. if `config.model.class_conditional` is enabled and `class_labels` is not `None`
3. create a class embedding with `nn.Embed(num_classes, nf * 4)`
4. add the class embedding to `temb`

Concrete code shape:

```python
class_emb = nn.Embed(
    num_embeddings=num_classes,
    features=nf * 4,
    embedding_init=default_initializer(),
)(class_labels.astype(jnp.int32))
temb = temb + class_emb
```

Why this design:

- it reuses the existing time-conditioning path
- it avoids invasive changes to residual blocks
- it keeps most pretrained student weights reusable

`JointNCSNpp` was updated the same way, forwarding `class_labels` into both submodels.

### 3. `jcm/models/utils.py`

This file is the interface layer between the model and the rest of the system.

The original code only handled one conditioning variable: time/noise labels.

The following functions were extended to accept optional `class_labels`:

- `get_model_fn`
- `get_denoiser_fn`
- `get_distiller_fn`
- `get_gaussianizer_fn`
- `get_score_fn`
- `get_denoiser_and_distiller_fn`

The key idea is:

- keep the old call pattern valid
- only pass `class_labels=...` to `model.apply(...)` when labels are explicitly provided

Example behavior in `get_model_fn`:

- old path:
  `model.apply(variables, x, labels, train=False, mutable=False)`
- new conditional path:
  `model.apply(variables, x, labels, class_labels=class_labels, train=False, mutable=False)`

This preserves backward compatibility for unconditional runs.

`init_model(...)` was also updated. When `class_conditional=True`, it initializes the model with:

```python
model.init(..., fake_input, fake_label, class_labels=fake_label)
```

This ensures the class embedding parameters exist at initialization time.

### 4. `jcm/losses.py`

This file is where labels actually enter training.

Added helper:

```python
def _get_class_labels(batch):
    return batch["label"] if "label" in batch else None
```

This avoids hard-coding conditional behavior into every batch path.

The following loss paths were updated to pass class labels into model utilities:

- `get_consistency_loss_fn`
- `get_progressive_distillation_loss_fn`
- `get_continuous_consistency_loss_fn`

For example, in `get_consistency_loss_fn`:

```python
class_labels = _get_class_labels(batch)
Ft, new_states = mutils.get_distiller_fn(
    ...,
    class_labels=class_labels,
)(x_t, t, rng=...)
```

and similarly for the target branch:

```python
Ft2, new_states = mutils.get_distiller_fn(
    ...,
    class_labels=class_labels,
)(x_t2, t2, rng=...)
```

Important implementation detail:

- if the batch has no `"label"` field, `class_labels` becomes `None`
- that means unconditional training still works without changes

### 5. `jcm/sampling.py`

Added:

```python
def _get_class_labels(config, rng, batch_size):
```

Behavior:

- `config.sampling.class_label is None`
  returns `None`, so sampling stays unconditional
- `config.sampling.class_label == "random"`
  samples class ids uniformly from `[0, num_classes)`
- otherwise
  uses a fixed integer class id for the whole batch

This helper was wired into:

- `get_onestep_sampler`
- `get_multistep_sampler`
- `get_seeded_sampler`

So the sampler can now create class-controlled student outputs without breaking the old default path.

### 6. `jcm/train.py`

This file now supports partial initialization from an existing student checkpoint.

Added helper functions:

- `_merge_compatible_tree(new_tree, old_tree)`
- `_maybe_initialize_from_checkpoint(state, init_ckpt)`

Problem being solved:

- the old unconditional student checkpoint does not contain class embedding weights
- the new conditional model does contain them
- full checkpoint restore would fail or mismatch

Solution:

- load raw checkpoint state with `checkpoints.restore_checkpoint(init_ckpt, None)`
- recursively copy only parameters whose tree keys and shapes are compatible
- keep unmatched new parameters, especially class embeddings, randomly initialized

This is only triggered when:

```python
int(state.step) == 0 and config.training.init_ckpt
```

So:

- resumed training from `checkpoints-meta` still behaves normally
- old unconditional workflows are not changed

## New Configs

### `configs/cifar10_ve_cd_conditional_ft.py`

This config enables class-conditioning while inheriting the original CD config:

```python
config.model.class_conditional = True
config.model.num_classes = 10
```

This config is useful as the base conditional config.

### `configs/cifar10_student_conditional_ft.py`

This config defines the student-only fine-tune route:

```python
config.training.loss = "dsm"
config.training.finetune = False
config.training.snapshot_sampling = False
```

Why `dsm` instead of `consistency`:

- `consistency` training in this repo still assumes a reference teacher path
- `dsm` allows direct student fine-tuning without teacher distillation
- this makes the "student-only" route actually independent from the teacher at train time

## New Launcher

Added:

- `run_student_conditional_ft.sh`

This launcher is intended for student-only conditional fine-tuning.

Default behavior:

```bash
WORKDIR=/nfs/tangwenhao/lhp/cd-conditional-student-ft
INIT_CKPT=/nfs/tangwenhao/lhp/cd-lpips/checkpoints/checkpoint_21
BATCH_SIZE=128
LR=1e-5
N_ITERS=20000
```

It runs:

```bash
python -m jcm.main \
  --config configs/cifar10_student_conditional_ft.py \
  --workdir "${WORKDIR}" \
  --mode train \
  --config.training.init_ckpt="${INIT_CKPT}" \
  ...
```

This means:

1. initialize the conditional student model
2. partially load compatible weights from the unconditional student checkpoint
3. keep class embedding randomly initialized
4. continue training with conditional DSM

## Why This Does Not Fully Match Conditional CD

This route is an engineering shortcut, not a strict conditional distillation pipeline.

It does **not** do:

- conditional teacher training
- conditional teacher-to-student consistency distillation

Instead, it does:

- unconditional student initialization
- conditional student fine-tuning

That is cheaper to implement, but methodologically weaker than a full teacher-based pipeline.

## What Still Remains

This implementation gives you the conditional fine-tuning backbone, but not a full end-to-end polished workflow.

The main remaining tasks are:

1. add explicit conditional sampling/eval commands to the project docs
2. add a helper script for generating all 10 CIFAR-10 classes in one run
3. verify that the partial checkpoint initialization works on the remote environment
4. run training and visually confirm class controllability

## Recommended Usage

1. keep existing unconditional experiments untouched
2. use a new workdir for conditional student fine-tuning
3. start from a known unconditional student checkpoint
4. use a small learning rate, for example `1e-5` or `5e-5`
5. first run short fine-tuning and inspect conditional samples before long training

## Summary

The implementation strategy is:

- add conditional capability as optional infrastructure
- preserve unconditional defaults
- add student-only initialization from an existing checkpoint
- avoid forcing a teacher-based retraining pipeline

This is the least disruptive way to extend the current repo toward class-conditional generation.
