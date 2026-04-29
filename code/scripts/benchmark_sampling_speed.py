#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import os
import sys
import time

import flax
import flax.jax_utils as flax_utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from jcm import checkpoints
from jcm import losses
from jcm import sampling
from jcm import sde_lib
from jcm.models import ddpm, ncsnv2, ncsnpp  # noqa: F401
from jcm.models import utils as mutils


def _log(msg):
    print(msg, flush=True)


def _merge_compatible_tree(new_tree, old_tree):
    if isinstance(new_tree, dict) and isinstance(old_tree, dict):
        merged = dict(new_tree)
        for key, value in new_tree.items():
            if key in old_tree:
                merged[key] = _merge_compatible_tree(value, old_tree[key])
        return merged

    new_shape = getattr(new_tree, "shape", None)
    old_shape = getattr(old_tree, "shape", None)
    if new_shape is not None and old_shape is not None and new_shape == old_shape:
        return old_tree
    return new_tree


def _restore_state_compatibly(state, ckpt_path_or_dir, ckpt_step):
    raw_state = checkpoints.restore_checkpoint(ckpt_path_or_dir, None, step=ckpt_step)
    if raw_state is None:
        raise ValueError(f"Checkpoint not found: {ckpt_path_or_dir}, step={ckpt_step}")

    state_dict = flax.serialization.to_state_dict(state)
    for field in ("params", "params_ema", "target_params", "model_state", "step"):
        if field in raw_state and field in state_dict:
            state_dict[field] = _merge_compatible_tree(state_dict[field], raw_state[field])
    return flax.serialization.from_state_dict(state, state_dict)


def _load_config(config_path):
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    spec = importlib.util.spec_from_file_location("benchmark_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def _build_state(config, ckpt_path_or_dir, ckpt_step):
    rng = hk.PRNGSequence(config.seed + 1)
    model, init_model_state, initial_params = mutils.init_model(next(rng), config)
    optimizer, _ = losses.get_optimizer(config)

    if config.training.loss.lower().endswith(
        ("ema", "adaptive", "progressive_distillation")
    ):
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

    state = _restore_state_compatibly(state, ckpt_path_or_dir, ckpt_step)
    return model, state


def _to_scalar_int(x):
    return int(np.asarray(jax.device_get(x)).reshape(-1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--timed-iters", type=int, default=10)
    parser.add_argument("--sampling-method", default="")
    parser.add_argument("--sampling-steps", type=int, default=-1)
    parser.add_argument("--std", type=float, default=-1.0)
    parser.add_argument("--class-label", default="none")
    parser.add_argument("--class-conditional", action="store_true")
    parser.add_argument("--conditioning-type", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    config = _load_config(args.config)
    config.eval.batch_size = args.batch_size
    config.eval.enable_sampling = True
    config.eval.enable_loss = False
    config.eval.enable_bpd = False

    if args.sampling_method:
        config.sampling.method = args.sampling_method
    if args.sampling_steps >= 0:
        config.sampling.n_steps = args.sampling_steps
    if args.std > 0:
        config.sampling.std = args.std
    if args.conditioning_type:
        config.model.conditioning_type = args.conditioning_type
    if args.class_conditional:
        config.model.class_conditional = True
    config.sampling.class_label = args.class_label

    if config.eval.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"batch_size={config.eval.batch_size} must be divisible by device_count={jax.device_count()}"
        )

    ckpt_source = os.path.join(args.workdir, "checkpoints")
    _log(
        f"[{args.name}] building state "
        f"(config={args.config}, ckpt={args.ckpt}, batch_size={config.eval.batch_size})"
    )
    model, state = _build_state(config, ckpt_source, args.ckpt)
    sde = sde_lib.get_sde(config)
    sampling_shape = (
        config.eval.batch_size // jax.local_device_count(),
        config.data.image_size,
        config.data.image_size,
        config.data.num_channels,
    )
    sampling_fn = sampling.get_sampling_fn(config, sde, model, sampling_shape)
    pstate = flax_utils.replicate(state)

    rng = hk.PRNGSequence(jax.random.PRNGKey(config.seed + 2026))
    rng = hk.PRNGSequence(jax.random.fold_in(next(rng), jax.process_index()))

    total_samples_per_iter = config.eval.batch_size

    _log(f"[{args.name}] starting warmup ({args.warmup_iters} iters)")
    for _ in range(args.warmup_iters):
        sample_rng = jnp.asarray(rng.take(jax.local_device_count()))
        samples, n = sampling_fn(sample_rng, pstate)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), samples)
        _ = _to_scalar_int(n)
        if args.warmup_iters <= 10 or (_ + 1) % max(1, args.warmup_iters // 5) == 0:
            _log(f"[{args.name}] warmup {_ + 1}/{args.warmup_iters} done")

    timings = []
    eval_steps = None
    _log(f"[{args.name}] starting timed runs ({args.timed_iters} iters)")
    for _ in range(args.timed_iters):
        sample_rng = jnp.asarray(rng.take(jax.local_device_count()))
        t0 = time.perf_counter()
        samples, n = sampling_fn(sample_rng, pstate)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), samples)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        eval_steps = _to_scalar_int(n)
        if args.timed_iters <= 20 or (_ + 1) % max(1, args.timed_iters // 10) == 0:
            _log(f"[{args.name}] timed {_ + 1}/{args.timed_iters} done")

    mean_sec_per_iter = float(np.mean(timings))
    std_sec_per_iter = float(np.std(timings))
    samples_per_sec = float(total_samples_per_iter / mean_sec_per_iter)
    sec_per_eval_step = float(mean_sec_per_iter / max(eval_steps, 1))

    result = {
        "name": args.name,
        "config": args.config,
        "workdir": args.workdir,
        "ckpt": args.ckpt,
        "sampling_method": config.sampling.method,
        "sampling_steps": int(getattr(config.sampling, "n_steps", 1)),
        "class_label": args.class_label,
        "batch_size": config.eval.batch_size,
        "warmup_iters": args.warmup_iters,
        "timed_iters": args.timed_iters,
        "eval_steps_per_iter": eval_steps,
        "mean_sec_per_iter": mean_sec_per_iter,
        "std_sec_per_iter": std_sec_per_iter,
        "samples_per_sec": samples_per_sec,
        "sec_per_eval_step": sec_per_eval_step,
    }

    _log(f"[{args.name}] finished")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fout:
            json.dump(result, fout, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
