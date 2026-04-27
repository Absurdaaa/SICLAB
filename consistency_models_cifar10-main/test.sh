# stu
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m jcm.main \
--config configs/cifar10_ve_cd.py \
--workdir /nfs/tangwenhao/lhp/cd-lpips \
--mode eval \
--eval_folder eval_student_40k_1step \
--config.eval.begin_ckpt=21 \
--config.eval.end_ckpt=21 \
--config.eval.num_samples=50000 \
--config.eval.batch_size=256 \
--config.eval.enable_loss=False \
--config.eval.enable_bpd=False \
--config.eval.enable_sampling=True

# 1step
fid 5.504918662306181
torchfid 5.50185727861691
is_mean 9.159789286154586
is_std 0.09658235910327317

CUDA_VISIBLE_DEVICES=0 \
python -m jcm.main \
--config configs/cifar10_ve_cd.py \
--workdir /nfs/tangwenhao/lhp/cd-lpips \
--mode metrics \
--eval_folder eval_student_40k_1step \
--config.eval.begin_ckpt=21 \
--config.eval.end_ckpt=21 \
--config.eval.batch_size=128 \
--config.eval.num_samples=50000

# 4step
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m jcm.main \
--config configs/cifar10_ve_cd.py \
--workdir /nfs/tangwenhao/lhp/cd-lpips \
--mode eval \
--eval_folder eval_student_40k_1step \
--config.eval.begin_ckpt=21 \
--config.eval.end_ckpt=21 \
--config.eval.num_samples=50000 \
--config.eval.batch_size=256 \
--config.eval.enable_loss=False \
--config.eval.enable_bpd=False \
--config.eval.enable_sampling=True \
--config.sampling.method=multistep \
--config.sampling.n_steps=4

CUDA_VISIBLE_DEVICES=0 \
python -m jcm.main \
--config configs/cifar10_ve_cd.py \
--workdir /nfs/tangwenhao/lhp/cd-lpips \
--mode metrics \
--eval_folder eval_student_40k_4step \
--config.eval.begin_ckpt=21 \
--config.eval.end_ckpt=21 \
--config.eval.batch_size=128 \
--config.eval.num_samples=50000 

python - <<'PY'
import numpy as np
path = "/nfs/tangwenhao/lhp/cd-lpips/eval_student_40k_1step/metrics_21.npz"
data = np.load(path)
print(data.files)
for k in data.files:
    print(k, data[k])
PY


## teacher

mkdir -p /nfs/tangwenhao/lhp/teacher_eval/checkpoints
ln -sf /tangwenhao/lhp/SICLAB/consistency_models_cifar10-main/checkpoints/edm_cifar10_ema /nfs/tangwenhao/lhp/teacher_eval/checkpoints/checkpoint_1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m jcm.main \
--config configs/cifar10_k_ve.py \
--workdir /nfs/tangwenhao/lhp/teacher_eval \
--mode eval \
--eval_folder eval_teacher \
--config.eval.begin_ckpt=1 \
--config.eval.end_ckpt=1 \
--config.eval.num_samples=50000 \
--config.eval.batch_size=128 \
--config.eval.enable_loss=False \
--config.eval.enable_bpd=False \
--config.eval.enable_sampling=True

CUDA_VISIBLE_DEVICES=0 \
python -m jcm.main \
--config configs/cifar10_k_ve.py \
--workdir /nfs/tangwenhao/lhp/teacher_eval \
--mode metrics \
--eval_folder eval_teacher \
--config.eval.begin_ckpt=21 \
--config.eval.end_ckpt=21 \
--config.eval.batch_size=128 \
--config.eval.num_samples=50000


python - <<'PY'
import numpy as np
x = np.load('/nfs/tangwenhao/lhp/teacher_eval/eval_teacher/metrics_1.npz')
for k in x.files:
    print(k, x[k])
PY
