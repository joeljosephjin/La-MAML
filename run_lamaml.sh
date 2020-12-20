#!/bin/bash

ROT="--n_layers 2 --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 1000 --dataset mnist_rotations    --cuda --log_dir logs/"
PERM="--n_layers 2 --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 1000 --dataset mnist_permutations --cuda --log_dir logs/"
MANY="--n_layers 2 --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 200 --dataset mnist_manypermutations --cuda --log_dir logs/"
CIFAR='--data_path data/ --log_every 100 --dataset cifar100 --cuda --log_dir logs/'
IMGNET='--data_path data/tiny-imagenet-200/ --log_every 100 --dataset tinyimagenet --log_dir logs/'
SEED=0

##### La-MAML #####
python3 main.py $IMGNET --model lamaml_cifar --expt_name lamaml --memories 400 --batch_size 10 --replay_batch_size 10 --n_epochs 10 \
                    --opt_lr 0.4 --alpha_init 0.1 --opt_wt 0.1 --glances 1 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 5 --learn_lr --log_every 3125 --second_order --class_order random \
                    --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.1


##### La-MAML #####
python3 main.py $IMGNET --model lamaml_cifar --expt_name lamaml --memories 400 --batch_size 10 --replay_batch_size 10 --n_epochs 1 \
                    --opt_lr 0.4 --alpha_init 0.1 --opt_wt 0.1 --glances 2 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 5 --learn_lr --log_every 3125 --second_order --class_order random \
                    --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.1