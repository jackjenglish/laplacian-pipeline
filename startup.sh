# Use docker image: https://hub.docker.com/r/continuumio/anaconda3/
# TODO XFORMERS
eval "$(conda shell.bash hook)"
conda activate base

touch .no_auto_tmux

apt update && apt install -y libsm6 libxext6
sudo apt-get -y install libsm6 libxrender1 libfontc
sudo apt-get -y install libxrender1
sudo apt -y install zip
apt-get update && apt-get -y install libgl1

# curl https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh --output anaconda.sh
# sha256sum anaconda.sh
# bash anaconda.sh -b -p ~/anaconda3
# eval "$(conda shell.bash hook)"
# source ~/.bashrc

git clone https://github.com/future-promise/stable-diffusion-webui.git

cd stable-diffusion-webui/models/Stable-diffusion
wget https://huggingface.co/Dunkindont/Foto-Assisted-Diffusion-FAD_V0/resolve/main/FAD-foto-assisted-diffusion_V0.ckpt
mv FAD-foto-assisted-diffusion_V0.ckpt model.ckpt

cd ~/stable-diffusion-webui/extensions
git clone https://github.com/future-promise/sd-webui-controlnet.git

cd ~/stable-diffusion-webui/extensions/sd-webui-controlnet/models
wget https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_canny-fp16.safetensors

cd ~/control-gen-server
pip install -r requirements.txt

cd ~/stable-diffusion-webui
pip install -r requirements.txt

cd ~/stable-diffusion-webui/extensions/sd-webui-controlnet/
pip install -r requirements.txt

cd ~/
# Launch (misleading name) Installs some necessary deps, webui.py runs actual server 
python3 stable-diffusion-webui/launch.py --api --disable-safe-unpickle --enable-insecure-extension-access --no-download-sd-model --no-half-vae --ckpt ./stable-diffusion-webui/models/Stable-diffusion/model.ckpt --enable-console-prompts

bash run.sh


wget https://mirror-personal-models.s3.us-west-2.amazonaws.com/control_laplace-fp16.safetensors