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

## Dataset Setup
- [Fruit rotten detection dataset](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)

Dataset used in the original paper, in the `./datasets/Fruit3`

- [Fruit and vegetables detection dataset](https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset)

Supplementary dataset that contains both fruit and vegetables, in the `./datasets/FruitVege`

## Quickstart
```shell
python resnet.py
```

## Experiment Logs

### 6-Class classification of rotten or fresh apples, oranges, and bananas

Accuracy for validation
- Inference without training: 18.6%
- Inference after training 1 epoch: 98.3%
- Inference after training 6 epoch: 99.7%


