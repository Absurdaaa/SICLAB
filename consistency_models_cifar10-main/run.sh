  python -m jcm.main \
    --config configs/cifar10_ve_cd.py \
    --workdir /nfs/tangwenhao/lhp/cd-lpips \
    --mode train \
    --config.training.ref_model_path=checkpoints/edm_cifar10_ema \
    --config.optim.lr=0.0004 \
    --config.training.loss_norm='lpips'