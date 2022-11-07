from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import glob
import os.path as osp

class SingleRotten(Dataset):
    def __init__(self, root='datasets', kind="Fruits", category="Apple", preprocess=None):
        image_dir = osp.join(root, kind)
        fresh_food_dir = osp.join(image_dir, "Fresh" + category)
        rotten_food_dir = osp.join(image_dir, "Rotten" + category)
        fresh_images = self.process_dir(fresh_food_dir)
        rotten_images = self.process_dir(rotten_food_dir)
        images = fresh_images + rotten_images
        labels = [0 for i in range(len(fresh_images))]
        labels = labels + [1 for i in range(len(rotten_images))]
        if preprocess is None:
            preprocess = transforms.ToTensor()
        self.preprocess = preprocess
        self.data = images
        self.label = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        img = self.preprocess(img)
        label = torch.tensor(label)
        return img, label

    def process_dir(self, dir):
        imgs_path = glob.glob(osp.join(dir, "*"))
        data = []
        for img_path in imgs_path:
            image = Image.open(img_path).convert("RGB")
            data.append(image)

        return data

