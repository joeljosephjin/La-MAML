IMGNET='--data_path data/tiny-imagenet-200/ --log_every 100 --dataset tinyimagenet --cuda --log_dir logs/'

SEED=0

#6. La-MAML TinyImageNet Dataset Multi-Pass
python3 mainwb.py $IMGNET --model lamaml_cifar --expt_name lamaml --memories 400 --batch_size 10 --replay_batch_size 10 --n_epochs 10 \
                    --opt_lr 0.4 --alpha_init 0.1 --opt_wt 0.1 --glances 1 --loader class_incremental_loader --increment 5 \
                    --arch "pc_cnn" --cifar_batches 5 --learn_lr --log_every 3125 --second_order --class_order random \
                    --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.1