## Install wsl on windows to use the GPU with torch and tensor flow
- Installation Command
wsl -- install

- Activation command
wsll.exe -d ubuntu

# Install CUDA compatible with the ubuntu wsl
- Cuda 11.8 version

- Commanda

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

** Note - Run the 5th command before the fourth one **


- Since some packages are depriciated use the following to manually install them before hitting install cuda

wget http://launchpadlibrarian.net/371711898/libtinfo5_6.1-1ubuntu1.18.04_amd64.deb
sudo dpkg -i libtinfo5_6.1-1ubuntu1.18.04_amd64.deb


- create a vistual environment to instal the torch
sudo apt install python3-venv
python3 -m venv .venv 
source .venv/bin/activate 
pip install requests 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## Resource
1. https://www.youtube.com/watch?v=R4m8YEixidI
2. https://pytorch.org/
3. https://askubuntu.com/questions/1491254/installing-cuda-on-ubuntu-23-10-libt5info-not-installable#:~:text=The%20libtinfo5%20package,while%20installing%20CUDA
4. https://askubuntu.com/questions/1465218/pip-error-on-ubuntu-externally-managed-environment-%C3%97-this-environment-is-extern

