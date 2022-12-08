import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from data.single_rotten import SingleRotten
from data.all_fruit import AllFruit
from data.all_fruit_vege import AllFruitVege
from resnet_model import Resnet50


batch_size = 10
random_seed = 42
num_cls = 2

model = Resnet50(cls=num_cls)
model.load_state_dict(torch.load("./model.pth"))

preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

# dataset = SingleRotten(category="Apple", preprocess=preprocess)
dataset = AllFruit(preprocess=preprocess, mode="test", binary=True)
# dataset = AllFruitVege(preprocess=preprocess, binary=True)

dataset_size = len(dataset)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# move the input and model to GPU for speed if available
device = torch.device('mps' if torch.has_mps else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)



correct_preds = 0
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        if i % 10 == 9:
            print("Number of batch ", i+1, ':')
        outputs = model(inputs)
        # probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, preds = torch.max(outputs, 1)
        for pred, label in zip(preds, labels):
            if pred == label:
                correct_preds += 1

    acc = 100 * correct_preds / dataset_size
    print(f'Accuracy for validation set is {acc:.1f} %')
