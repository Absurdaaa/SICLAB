from configs.cifar10_ve_cd_conditional_ft import get_config as get_base_config


def get_config():
    config = get_base_config()

    # Teacher-free fine-tuning route: initialize from an existing CD student
    # checkpoint and continue training with score matching.
    config.training.loss = "dsm"
    config.training.finetune = False
    config.training.snapshot_sampling = False

    return config
