set -e

URL_PREFIX=http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/
CUDNN_REPO_PKG=nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb

wget -O /tmp/${CUDNN_REPO_PKG} ${URL_PREFIX}/${CUDNN_REPO_PKG}
sudo dpkg -i /tmp/${CUDNN_REPO_PKG}
sudo apt-key adv --fetch-keys ${URL_PREFIX}/7fa2af80.pub

sudo apt update
sudo apt install -y libcudnn7 libcudnn7-dev
