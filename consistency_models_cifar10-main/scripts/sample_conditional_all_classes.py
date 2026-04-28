#!/usr/bin/env python3
import argparse
import importlib.util
import io
import logging
import math
import os
import sys

import blobfile
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


def _load_config(config_path):
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    spec = importlib.util.spec_from_file_location("sampling_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def _restore_state(config, workdir, ckpt):
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

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
    return model, state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("--conditioning-type", default="adagn")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--class-start", type=int, default=0)
    parser.add_argument("--class-end", type=int, default=9)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
    )

    config = _load_config(args.config)
    config.eval.batch_size = args.batch_size
    config.eval.num_samples = args.num_samples
    config.eval.enable_loss = False
    config.eval.enable_bpd = False
    config.eval.enable_sampling = True
    config.model.class_conditional = True
    config.model.conditioning_type = args.conditioning_type
    config.model.num_classes = 10

    if config.eval.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"eval batch size {config.eval.batch_size} must be divisible by "
            f"device count {jax.device_count()}"
        )

    model, state = _restore_state(config, args.workdir, args.ckpt)
    sde = sde_lib.get_sde(config)
    sampling_shape = (
        config.eval.batch_size // jax.local_device_count(),
        config.data.image_size,
        config.data.image_size,
        config.data.num_channels,
    )
    pstate = flax_utils.replicate(state)

    rng = hk.PRNGSequence(jax.random.PRNGKey(config.seed + 1234))
    rng = hk.PRNGSequence(jax.random.fold_in(next(rng), jax.process_index()))

    num_rounds = int(math.ceil(config.eval.num_samples / config.eval.batch_size))

    for class_id in range(args.class_start, args.class_end + 1):
        class_dir = os.path.join(args.output_root, f"class_{class_id}")
        host_dir = os.path.join(class_dir, f"ckpt_{args.ckpt}_host_{jax.process_index()}")
        blobfile.makedirs(host_dir)

        config.sampling.class_label = str(class_id)
        sampling_fn = sampling.get_sampling_fn(config, sde, model, sampling_shape)

        logging.info("Sampling class %d into %s", class_id, host_dir)
        for r in range(num_rounds):
            sample_rng = jnp.asarray(rng.take(jax.local_device_count()))
            samples, n = sampling_fn(sample_rng, pstate)
            samples = jax.tree_util.tree_map(lambda x: x.block_until_ready(), samples)
            num_eval_steps = int(np.asarray(jax.device_get(n)).reshape(-1)[0])

            samples = (samples + 1.0) / 2.0
            samples = np.clip(samples * 255.0, 0, 255).astype(np.uint8)
            samples = samples.reshape(
                (
                    -1,
                    config.data.image_size,
                    config.data.image_size,
                    config.data.num_channels,
                )
            )

            with blobfile.BlobFile(
                os.path.join(host_dir, f"samples_{r}.npz"), "wb"
            ) as fout:
                io_buffer = io.BytesIO()
                np.savez(io_buffer, samples=samples)
                fout.write(io_buffer.getvalue())

            logging.info(
                "class %d round %d: saved %d samples (%d eval steps)",
                class_id,
                r,
                samples.shape[0],
                num_eval_steps,
            )


if __name__ == "__main__":
    main()
