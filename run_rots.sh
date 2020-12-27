ROT="--n_layers 2 --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 1000 --dataset mnist_rotations    --cuda --log_dir logs/"

#cmaml ROTATION MNIST DATASETS
python3 mainwb.py $ROT --model lamaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 5 --opt_lr 0.1 \
                    --opt_wt 0.1 --alpha_init 0.1 --sync_update --use_old_task_memory --seed $SEED

#sync ROTATION MNIST DATASETS
python3 mainwb.py $ROT --model lamaml --memories 200 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 5 --opt_lr 0.1 \
                    --opt_wt 0.3 --alpha_init 0.15 --learn_lr --sync_update --use_old_task_memory --seed $SEED