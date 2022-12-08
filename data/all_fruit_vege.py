from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import glob
import os.path as osp


class AllFruitVege(Dataset):
    def __init__(self, root='datasets', preprocess=None, binary=False):
        image_dir = osp.join(osp.join(root, "FruitVege"), "*")
        food_dirs = glob.glob(osp.join(image_dir, "*"))
        data_list = [self.process_dir(dir_i) for dir_i in food_dirs]
        data = sum(data_list, [])
        # label_list = [[1 for _ in range(self.size)] if "rotten" in osp.basename(fdir) else [0 for _ in range(self.size)] for fdir in fruit_dirs]
        if binary:
            label_list = [[0 if "Fresh" in food_dirs[i] else 1 for _ in range(len(data_list[i]))] for i in range(len(food_dirs))]
        else:
            label_list = [[i for _ in range(len(data_list[i]))] for i in range(len(food_dirs))]
        label = sum(label_list, [])
        if preprocess is None:
            preprocess = transforms.ToTensor()
        self.preprocess = preprocess
        self.data = data
        self.label = label
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
