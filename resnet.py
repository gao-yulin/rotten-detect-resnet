import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from data.single_rotten import SingleRotten
from binary_resnet import binaryResnet50

model = binaryResnet50()

child_counter = 0
"""for child in model.children():
   print(" child", child_counter, "is:")
   print(child)
   child_counter += 1"""

preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

batch_size = 4
validation_split = .2
shuffle_dataset = True
random_seed = 42

dataset = SingleRotten(preprocess=preprocess)

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

# move the input and model to GPU for speed if available
device = torch.device('mps' if torch.has_mps else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


for epoch in range(8):  # loop over the dataset multiple times

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
        if i % 50 == 49:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')



categories = ["Fresh Apple", "Rotten Apple"]

correct_preds = 0
model.eval()
with torch.no_grad():
    for i, data in enumerate(validation_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        print("Number of batch ", i, ':')
        outputs = model(inputs)
        # probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, preds = torch.max(outputs, 1)
        print([categories[index] for index in preds])
        for pred, label in zip(preds, labels):
            if pred == label:
                correct_preds += 1

    acc = 100 * correct_preds / split
    print(f'Accuracy for validation set is {acc:.1f} %')