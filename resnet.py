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
validation_split = .3
shuffle_dataset = True
random_seed = 19
num_cls = 2

model = Resnet50(cls=num_cls)

"""preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])"""

preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),  # 0.5
            transforms.RandomVerticalFlip(p=0.5),  # 0.5
            transforms.RandomRotation([-90, 90]),
            transforms.ToTensor(),
            transforms.RandomErasing(p = 0.8, scale=(0.02, 0.33)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# dataset = SingleRotten(kind="Vegetables", category="Tomato", preprocess=preprocess)
# dataset = AllFruit(preprocess=preprocess, mode="train", binary=True)
dataset = AllFruitVege(preprocess=preprocess, binary=True)

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler, drop_last=True)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler, drop_last=True)

# move the input and model to GPU for speed if available
device = torch.device('mps' if torch.has_mps else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


for epoch in range(6):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 30 == 29:
            print(f'[epoch {epoch + 1}, batch {i + 1:3d}] loss: {running_loss / 30:.3f}')
            running_loss = 0.0

print('Finished Training')

torch.save(model.state_dict(), "./model.pth")


correct_preds = 0
model.eval()
with torch.no_grad():
    for i, data in enumerate(validation_loader):
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

    acc = 100 * correct_preds / split
    print(f'Accuracy for validation set is {acc:.1f} %')
