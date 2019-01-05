# https://docs.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup
set -e

URL_PREFIX=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64
CUDA_REPO_PKG=cuda-repo-ubuntu1604_9.2.88-1_amd64.deb

wget -O /tmp/${CUDA_REPO_PKG} ${URL_PREFIX}/${CUDA_REPO_PKG}
sudo dpkg -i /tmp/${CUDA_REPO_PKG}
sudo apt-key adv --fetch-keys ${URL_PREFIX}/7fa2af80.pub

sudo apt update
sudo apt install -y cuda-drivers cuda cuda-9-0 cuda-9-2

sudo apt install -y nvidia-cuda-dev # for cuda_runtime.h
