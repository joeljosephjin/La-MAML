##### ER ##### CIFAR DATASET Multi-Pass 
python3 mainwb.py $CIFAR --model eralg4 --expt_name eralg4 --memories 200 --batch_size 10 --replay_batch_size 1 --n_epochs 10 \
                     --lr 0.03 --glances 1 --loader class_incremental_loader  --increment 5 \
                    --arch "pc_cnn" --log_every 3125 --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.1

DIFF: n_epochs, glances

##### ER #####  CIFAR DATASET Single-Pass
python3 mainwb.py $CIFAR --model eralg4 --expt_name eralg4 --memories 200 --batch_size 10 --replay_batch_size 1 --n_epochs 1 \
                     --lr 0.03 --glances 10 --loader class_incremental_loader  --increment 5 \
                    --arch "pc_cnn" --log_every 3125 --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.1



##### La-MAML ##### TinyImageNet Dataset Multi-Pass
python3 mainwb.py $IMGNET --model lamaml_cifar --expt_name lamaml --memories 400 --batch_size 10 --replay_batch_size 10 --n_epochs 10 \
                    --opt_lr 0.4 --alpha_init 0.1 --opt_wt 0.1 --glances 1 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 5 --learn_lr --log_every 3125 --second_order --class_order random \
                    --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.1

DIFF: n_epochs, glances, 

##### La-MAML ##### TinyImageNet Dataset Single-Pass
python3 mainwb.py $IMGNET --model lamaml_cifar --expt_name lamaml --memories 400 --batch_size 10 --replay_batch_size 10 --n_epochs 1 \
                    --opt_lr 0.4 --alpha_init 0.1 --opt_wt 0.1 --glances 2 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 5 --learn_lr --log_every 3125 --second_order --class_order random \
                    --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.1