#!/bin/bash --login

export MPLBACKEND="agg"

# flags:
## add flag --cuda to run script on GPUs, otherwise do not add flag

mkdir -p logs
#python train.py settings/gwae.yaml | tee logs/log_celeba_rasgw.txt
#python train.py settings/gwae.yaml 2>&1 | tee logs_round_3/log_shapes3d_rasgw.txt

python3 main_gwgan_cnn.py --data fmnist --num_epochs 100 --beta 35  --cuda
#python3 main_gwgan_cnn.py --data fmnist --num_epochs 100 --beta 35  --cuda --n_channels 1 2>&1 | tee logs/log_fmnist_sgw.txt
#python3 main_gwgan_cnn.py --data cifar_gray --num_epochs 100 --beta 40  --cuda
#python3 main_gwgan_cnn.py --data mnist --num_epochs 100 --beta 32 --cuda  --n_channels 1
# (beta for MNIST: 32, for gray-scale CIFAR: 40)
