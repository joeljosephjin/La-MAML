MANY="--n_layers 2 --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 200 --dataset mnist_manypermutations --cuda --log_dir logs/"

SEED=0

#cmaml MANY MNIST DATASETS
python3 mainwb.py $MANY --model lamaml --memories 500 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 5 --opt_wt 0.15 \
                    --alpha_init 0.03 --sync_update --use_old_task_memory --seed $SEED

# #sync MANY MNIST DATASETS
# python3 mainwb.py $MANY --model lamaml --memories 500 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 10 --opt_lr 0.1 \
#                     --opt_wt 0.03  --alpha_init 0.03 --learn_lr --sync_update --use_old_task_memory --seed $SEED

# #lamaml MANY MNIST DATASETS
# python3 mainwb.py $MANY --model lamaml --memories 500 --batch_size 10 --replay_batch_size 10 --n_epochs 1 --glances 10 --opt_lr 0.1 \
#                     --alpha_init 0.1 --learn_lr --use_old_task_memory --seed $SEED