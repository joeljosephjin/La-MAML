apt-get update

apt-get install wget

apt-get install python3-pip # terminal

pip3 install ipdb

conda install matplotlib numpy pillow torchvision urllib3 scipy --yes # terminal

conda update -n base -c defaults conda --yes #terminal

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch --yes # terminal

# cd La-MAML

git pull https://github.com/joeljosephjin/La-MAML

conda install -c conda-forge wandb --yes

wandb login 665a5d573c302c27f7dab355484da17a460e6759

chmod +x run_many.sh
./run_many.sh