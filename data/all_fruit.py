from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import glob
import os.path as osp


class AllFruit(Dataset):
    def __init__(self, root='datasets', preprocess=None):
        self.size = 500
        root_dir = osp.join(osp.join(root, "Fruit3"), "train")
        fruit_dirs = glob.glob(osp.join(root_dir, "*"))
        data_list = [self.process_dir(fruit_dir) for fruit_dir in fruit_dirs]
        data = sum(data_list, [])
        # label_list = [[1 for _ in range(self.size)] if "rotten" in osp.basename(fdir) else [0 for _ in range(self.size)] for fdir in fruit_dirs]
        label_list = [[i for _ in range(self.size)] for i in range(len(fruit_dirs))]
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
        assert len(data) == self.size
        return data
