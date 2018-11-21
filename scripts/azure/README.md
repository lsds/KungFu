# azure scripts

Collection of scripts for various operations.

## Usage

```bash
./cloud/gpu-machine.sh # create a new GPU VM
# you can select the machine type by editing the IMAGE variable

./cloud/gpu-machine.sh ssh # log into the VM

# inside the VM
./gpu-machine/init.sh
sudo reboot -h 0

./cloud/gpu-machine.sh ssh # log into the VM again
./gpu-machine/test-tf-gpu.py # check if tensorflow, cuda, are installed prpoerly

./experiments/run-tensorflow-benchmark.sh # download an run an official benchmark
```
