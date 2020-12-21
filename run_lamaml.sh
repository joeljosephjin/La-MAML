#!/bin/bash

ROT="--n_layers 2 --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 1000 --dataset mnist_rotations    --cuda --log_dir logs/"
PERM="--n_layers 2 --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 1000 --dataset mnist_permutations --cuda --log_dir logs/"
MANY="--n_layers 2 --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 200 --dataset mnist_manypermutations --cuda --log_dir logs/"

CIFAR='--data_path data/ --log_every 100 --dataset cifar100 --cuda --log_dir logs/'
IMGNET='--data_path data/tiny-imagenet-200/ --log_every 100 --dataset tinyimagenet --cuda --log_dir logs/'

SEED=0

#1. lamaml ROTATION MNIST DATASETS
python3 main.py $ROT --model lamaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 5 --opt_lr 0.3 \
                    --alpha_init 0.15 --learn_lr --use_old_task_memory --seed $SEED

#2. lamaml PERMUTATION MNIST DATASETS
python3 main.py $PERM --model lamaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 5 --opt_lr 0.3 \
                    --alpha_init 0.15 --learn_lr --use_old_task_memory --seed $SEED

#3. lamaml MANY MNIST DATASETS
python3 main.py $MANY --model lamaml --memories 500 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 10 --opt_lr 0.1 \
                    --alpha_init 0.1 --learn_lr --use_old_task_memory --seed $SEED

#4. La-MAML CIFAR DATASET Multi-Pass
python3 main.py $CIFAR --model lamaml_cifar --expt_name lamaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 10 \
                    --opt_lr 0.25 --alpha_init 0.1 --opt_wt 0.1 --glances 1 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 5 --learn_lr --log_every 3125 --second_order --class_order random \
                    --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.1

#5. La-MAML CIFAR DATASET Single-Pass
python3 main.py $CIFAR --model lamaml_cifar --expt_name lamaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 \
                    --opt_lr 0.25 --alpha_init 0.1 --opt_wt 0.1 --glances 10 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 5 --learn_lr --log_every 3125 --second_order --class_order random \
                    --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.1

#6. La-MAML TinyImageNet Dataset Multi-Pass
python3 main.py $IMGNET --model lamaml_cifar --expt_name lamaml --memories 400 --batch_size 10 --replay_batch_size 10 --n_epochs 10 \
                    --opt_lr 0.4 --alpha_init 0.1 --opt_wt 0.1 --glances 1 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 5 --learn_lr --log_every 3125 --second_order --class_order random \
                    --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.1

#7. La-MAML TinyImageNet Dataset Single-Pass
python3 main.py $IMGNET --model lamaml_cifar --expt_name lamaml --memories 400 --batch_size 10 --replay_batch_size 10 --n_epochs 1 \
                    --opt_lr 0.4 --alpha_init 0.1 --opt_wt 0.1 --glances 2 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 5 --learn_lr --log_every 3125 --second_order --class_order random \
                    --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.1