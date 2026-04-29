from configs.cifar10_ve_cd import get_config as get_base_config


def get_config():
    config = get_base_config()

    # Enable optional class-conditioning while reusing the existing CD setup.
    config.model.class_conditional = True
    config.model.num_classes = 10

    return config
