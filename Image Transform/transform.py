import torchvision
from PIL import Image
from torchvision import transforms

img_path = "./mango.jpg"
img = Image.open(img_path)

img_pure = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)])(img)

img_pure.save("mango_pure.jpg")

img_flip = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=1), # 0.5
    transforms.RandomVerticalFlip(p=1), # 0.5
    transforms.RandomRotation([-90,90])])(img)

img_flip.save("mango_flip.jpg")

img_color = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)])(img)

img_color.save("mango_color.jpg")

img_erase = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.RandomErasing(p = 0.8, scale=(0.02, 0.33)),
    transforms.ToPILImage()])(img)

img_erase.save("mango_erase.jpg")

img_sharp = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.RandomAdjustSharpness(30,1), # 0.5
    transforms.ToPILImage()])(img)

img_sharp.save("mango_sharp.jpg")


# change preprocess as follows

preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5), # 0.5
            transforms.RandomVerticalFlip(p=0.5), # 0.5
            transforms.RandomRotation([-90,90]),
            transforms.ToTensor(),
            transforms.RandomErasing(p = 0.8, scale=(0.02, 0.33)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])