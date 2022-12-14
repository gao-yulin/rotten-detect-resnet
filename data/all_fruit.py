from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import glob
import os.path as osp


class AllFruit(Dataset):
    def __init__(self, root='datasets', mode="train", preprocess=None, binary=False):
        self.mode = mode
        self.size = 500
        root_dir = osp.join(osp.join(root, "Fruit3"), mode)
        fruit_dirs = glob.glob(osp.join(root_dir, "*"))
        data_list = [self.process_dir(fruit_dir) for fruit_dir in fruit_dirs]
        data = sum(data_list, [])
        # label_list = [[1 for _ in range(self.size)] if "rotten" in osp.basename(fdir) else [0 for _ in range(self.size)] for fdir in fruit_dirs]
        if binary:
            label_list = [[0 if "fresh" in fruit_dirs[i] else 1 for _ in range(len(data_list[i]))] for i in range(len(fruit_dirs))]
        else:
            label_list = [[i for _ in range(len(data_list[i]))] for i in range(len(fruit_dirs))]
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
        image_cnt = 0
        for img_path in imgs_path:
            image_cnt += 1
            image = Image.open(img_path).convert("RGB")
            data.append(image)
            if image_cnt == self.size:
                break
        if self.mode == "train": assert len(data) == self.size
        return data
