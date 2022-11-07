# rotten-detect-resnet
A rotten fruit detection algorithm based on resnet
## Installation
```shell
# First make sure conda is installed on your mac
conda create -n torch-gpu python=3.9
conda activate torch-gpu
# MPS acceleration is available on MacOS 12.3+
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
# Or install pytorch without MPS acceleration
conda install pytorch torchvision torchaudio -c pytorch
```

