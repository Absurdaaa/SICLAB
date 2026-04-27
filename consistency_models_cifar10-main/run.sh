#   python -m jcm.main \
#     --config configs/cifar10_ve_cd.py \
#     --workdir /nfs/tangwenhao/lhp/cd-lpips \
#     --mode train \
#     --config.training.ref_model_path=checkpoints/edm_cifar10_ema \
#     --config.optim.lr=0.0004 \
#     --config.training.loss_norm='lpips'
    
    
# stu
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m jcm.main \
--config configs/cifar10_ve_cd.py \
--workdir /nfs/tangwenhao/lhp/cd-lpips \
--mode eval \
--eval_folder eval_student_50k_1step \
--config.eval.begin_ckpt=25 \
--config.eval.end_ckpt=25 \
--config.eval.num_samples=50000 \
--config.eval.batch_size=512 \
--config.eval.enable_loss=False \
--config.eval.enable_bpd=False \
--config.eval.enable_sampling=True

# 1step
# fid 5.2251193333027
# torchfid 5.222541041249542
# is_mean 9.184435462955332
# is_std 0.12045264797134327

CUDA_VISIBLE_DEVICES=0 \
python -m jcm.main \
--config configs/cifar10_ve_cd.py \
--workdir /nfs/tangwenhao/lhp/cd-lpips \
--mode metrics \
--eval_folder eval_student_50k_1step \
--config.eval.begin_ckpt=25 \
--config.eval.end_ckpt=25 \
--config.eval.batch_size=256 \
--config.eval.num_samples=50000