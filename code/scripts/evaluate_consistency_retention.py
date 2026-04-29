#!/usr/bin/env python3
import copy
import functools
import json
import os
import sys
from pathlib import Path

import flax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from ml_collections.config_flags import config_flags


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jcm import checkpoints
from jcm import datasets
from jcm import losses
from jcm import sde_lib
from jcm.models import ddpm, ncsnv2, ncsnpp
from jcm.models import utils as mutils


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "model_config",
    None,
    "Model architecture/data config of the checkpoint being evaluated.",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "consistency_config",
    None,
    "Original consistency/distillation config used as the consistency metric definition.",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "baseline_model_config",
    None,
    "Optional model config for the baseline checkpoint. Defaults to --model_config.",
    lock_config=False,
)

flags.DEFINE_string("workdir", None, "Work directory of the checkpoint to evaluate.")
flags.DEFINE_string("ckpt", "latest", "Checkpoint id to evaluate, or 'latest'.")
flags.DEFINE_string(
    "baseline_workdir",
    "",
    "Optional baseline workdir for comparison, e.g. the original student.",
)
flags.DEFINE_string(
    "baseline_ckpt",
    "latest",
    "Optional baseline checkpoint id to evaluate, or 'latest'.",
)
flags.DEFINE_enum("split", "test", ["train", "test"], "Dataset split for evaluation.")
flags.DEFINE_integer("num_batches", 50, "Number of evaluation batches to average over.")
flags.DEFINE_integer(
    "batch_size",
    128,
    "Evaluation batch size. Must be divisible by the number of devices.",
)
flags.DEFINE_integer(
    "seed",
    1234,
    "Random seed for the consistency-loss evaluation noise/time sampling.",
)
flags.DEFINE_string(
    "output",
    "",
    "Optional JSON output path. If empty, only prints results.",
)

flags.mark_flags_as_required(["model_config", "consistency_config", "workdir"])


def _resolve_latest_checkpoint(ckpt_dir: Path) -> int:
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    ckpts = []
    for path in ckpt_dir.iterdir():
        if path.is_file() and path.name.startswith("checkpoint_"):
            try:
                ckpts.append(int(path.name.split("checkpoint_")[1]))
            except ValueError:
                continue
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint_* files found under {ckpt_dir}")
    return max(ckpts)


def _resolve_ckpt_arg(workdir: str, ckpt_arg: str) -> int:
    ckpt_dir = Path(workdir).expanduser().resolve() / "checkpoints"
    if ckpt_arg == "latest":
        return _resolve_latest_checkpoint(ckpt_dir)
    return int(ckpt_arg)


def _copy_config(config):
    return copy.deepcopy(config)


def build_eval_config(model_config, consistency_config, batch_size: int):
    config = _copy_config(model_config)
    with config.unlocked():
        config.training.loss = consistency_config.training.loss
        if hasattr(consistency_config.training, "loss_norm"):
            config.training.loss_norm = consistency_config.training.loss_norm
        if hasattr(consistency_config.training, "weighting"):
            config.training.weighting = consistency_config.training.weighting
        if hasattr(consistency_config.training, "stopgrad"):
            config.training.stopgrad = consistency_config.training.stopgrad
        if hasattr(consistency_config.training, "dsm_target"):
            config.training.dsm_target = consistency_config.training.dsm_target
        if hasattr(consistency_config.training, "solver"):
            config.training.solver = consistency_config.training.solver
        if hasattr(consistency_config.training, "ref_config"):
            config.training.ref_config = consistency_config.training.ref_config
        if hasattr(consistency_config.training, "ref_model_path"):
            config.training.ref_model_path = consistency_config.training.ref_model_path
        config.training.finetune = False
        config.eval.batch_size = batch_size
    return config


def init_state(config, rng):
    score_model, init_model_state, initial_params = mutils.init_model(next(rng), config)
    optimizer, optimize_fn = losses.get_optimizer(config)
    if config.training.loss.lower().endswith(("ema", "adaptive", "progressive_distillation")):
        state = mutils.StateWithTarget(
            step=0,
            lr=config.optim.lr,
            ema_rate=config.model.ema_rate,
            params=initial_params,
            target_params=initial_params,
            params_ema=initial_params,
            model_state=init_model_state,
            opt_state=optimizer.init(initial_params),
            rng_state=rng.internal_state,
        )
    else:
        state = mutils.State(
            step=0,
            lr=config.optim.lr,
            ema_rate=config.model.ema_rate,
            params=initial_params,
            params_ema=initial_params,
            model_state=init_model_state,
            opt_state=optimizer.init(initial_params),
            rng_state=rng.internal_state,
        )
    return score_model, state, optimize_fn


def evaluate_one(model_config, consistency_config, workdir: str, ckpt: int, split: str, num_batches: int, batch_size: int, seed: int):
    config = build_eval_config(model_config, consistency_config, batch_size=batch_size)
    rng = hk.PRNGSequence(seed)

    score_model, state, optimize_fn = init_state(config, rng)
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)

    sde = sde_lib.get_sde(config)
    train_ds, eval_ds = datasets.get_dataset(
        config,
        additional_dim=1,
        uniform_dequantization=config.data.uniform_dequantization,
        evaluation=True,
        drop_last=False,
    )
    ds = train_ds if split == "train" else eval_ds

    _, eval_loss_fn, state = losses.get_loss_fn(config, sde, score_model, state, next(rng))
    ema_scales_fn = losses.get_ema_scales_fn(config)
    eval_step = losses.get_step_fn(
        eval_loss_fn,
        train=False,
        optimize_fn=optimize_fn,
        ema_scales_fn=ema_scales_fn,
    )
    p_eval_step = jax.pmap(
        functools.partial(jax.lax.scan, eval_step),
        axis_name="batch",
    )
    pstate = flax.jax_utils.replicate(state)

    all_losses = []
    all_log_stats = {}
    ds_iter = iter(ds)
    for batch_idx in range(num_batches):
        try:
            batch = next(ds_iter)
        except StopIteration:
            break
        eval_batch = jax.tree_util.tree_map(lambda x: x.detach().cpu().numpy(), batch)
        next_rng = jnp.asarray(rng.take(jax.local_device_count()))
        (_, _), (p_eval_loss, p_eval_log_stats) = p_eval_step((next_rng, pstate), eval_batch)
        eval_loss = np.asarray(flax.jax_utils.unreplicate(p_eval_loss)).reshape(-1)
        all_losses.extend(eval_loss.tolist())
        log_stats = flax.jax_utils.unreplicate(p_eval_log_stats)
        for key, value in log_stats.items():
            all_log_stats.setdefault(key, []).extend(np.asarray(value).reshape(-1).tolist())

    if not all_losses:
        raise RuntimeError("No evaluation batches were processed.")

    result = {
        "workdir": os.path.abspath(workdir),
        "ckpt": int(ckpt),
        "split": split,
        "num_batches": len(all_losses),
        "mean_loss": float(np.mean(all_losses)),
        "std_loss": float(np.std(all_losses)),
        "min_loss": float(np.min(all_losses)),
        "max_loss": float(np.max(all_losses)),
        "loss_name": str(config.training.loss),
        "loss_norm": str(getattr(config.training, "loss_norm", "")),
    }
    if all_log_stats:
        result["log_stats_mean"] = {
            key: float(np.mean(values)) for key, values in sorted(all_log_stats.items())
        }
    return result


def main(argv):
    del argv
    target_ckpt = _resolve_ckpt_arg(FLAGS.workdir, FLAGS.ckpt)
    target_result = evaluate_one(
        FLAGS.model_config,
        FLAGS.consistency_config,
        FLAGS.workdir,
        target_ckpt,
        FLAGS.split,
        FLAGS.num_batches,
        FLAGS.batch_size,
        FLAGS.seed,
    )

    payload = {"target": target_result}
    print(
        f"[target] ckpt={target_result['ckpt']} "
        f"mean_{target_result['loss_name']}_loss={target_result['mean_loss']:.6f} "
        f"std={target_result['std_loss']:.6f}"
    )

    if FLAGS.baseline_workdir:
        baseline_config = (
            FLAGS.baseline_model_config
            if FLAGS.baseline_model_config is not None
            else FLAGS.model_config
        )
        baseline_ckpt = _resolve_ckpt_arg(FLAGS.baseline_workdir, FLAGS.baseline_ckpt)
        baseline_result = evaluate_one(
            baseline_config,
            FLAGS.consistency_config,
            FLAGS.baseline_workdir,
            baseline_ckpt,
            FLAGS.split,
            FLAGS.num_batches,
            FLAGS.batch_size,
            FLAGS.seed,
        )
        payload["baseline"] = baseline_result
        delta = target_result["mean_loss"] - baseline_result["mean_loss"]
        ratio = target_result["mean_loss"] / baseline_result["mean_loss"]
        payload["comparison"] = {
            "delta_mean_loss": float(delta),
            "ratio_mean_loss": float(ratio),
        }
        print(
            f"[baseline] ckpt={baseline_result['ckpt']} "
            f"mean_{baseline_result['loss_name']}_loss={baseline_result['mean_loss']:.6f} "
            f"std={baseline_result['std_loss']:.6f}"
        )
        print(f"[compare] delta={delta:.6f} ratio={ratio:.6f}")

    if FLAGS.output:
        output_path = Path(FLAGS.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"saved_json={output_path}")


if __name__ == "__main__":
    app.run(main)
