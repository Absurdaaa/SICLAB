from configs.cifar10_ve_cd_conditional_ft import get_config as get_base_config


def get_config():
    config = get_base_config()

    # Student-only conditional fine-tuning:
    # initialize from an existing unconditional student checkpoint and adapt it
    # with score matching. We intentionally avoid teacher-based consistency
    # losses here because an unconditional teacher would not provide a useful
    # conditional target.
    config.training.loss = "dsm"
    config.training.finetune = False
    config.training.snapshot_sampling = False
    config.training.ref_model_path = ""

    return config
