from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from PIL import Image, UnidentifiedImageError
import os
from kornia.color import rgb_to_lab
import warnings


class RGB2LAB:
    def __call__(self, img) -> Image.Image:
        # 确保图像为 RGB 三通道（灰度图需转换）
        if img.mode != "RGB":
            img = img.convert("RGB")
        imgTensor = ToTensor()(img).unsqueeze(0)
        # converting image to tensor

        lab = rgb_to_lab(imgTensor).squeeze(0)

        lab[0] /= 50
        lab[1:] /= 128
        lab[0] -= 1

        return lab

class ImageDataset(Dataset):
    def __init__(self, targetFolder:str,transforms:Compose=None) -> None:
        super().__init__()
        self.targetFolder = targetFolder
        self.files = os.listdir(self.targetFolder)
        self.transforms = transforms

        if self.transforms == None:
            self.transforms = Compose(
                [
                    Resize((256, 256)),
                    RGB2LAB()
                    ]
                )

        # 预先找一张有效图片作为损坏文件的回退目标
        self._safe_fallback = None
        for f in self.files:
            try:
                img = Image.open(os.path.join(self.targetFolder, f))
                img.verify()
                self._safe_fallback = f
                break
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index):
        image = self.files[index]
        imagePath = os.path.join(self.targetFolder, image)

        for attempt in range(2):
            try:
                img = Image.open(imagePath)
                img.verify()
                img = Image.open(imagePath)
                lab = self.transforms(img)
                return lab[0].unsqueeze(0), lab[1:]
            except (UnidentifiedImageError, OSError, SyntaxError) as e:
                if attempt == 0 and self._safe_fallback:
                    warnings.warn(f"跳过损坏的图片: {imagePath} ({e})")
                    image = self._safe_fallback
                    imagePath = os.path.join(self.targetFolder, image)
                else:
                    raise RuntimeError(f"无法读取图片且无有效回退: {imagePath}")
    
def getDataLoader(targetFolder, transforms:Compose=None, batchSize=64, numWorks=4):
    return DataLoader(ImageDataset(targetFolder, transforms), batch_size=batchSize, shuffle=True, num_workers=numWorks)


if __name__ == "__main__":
    import torch
    import time
    from matplotlib import pyplot as plt
    import numpy as np
    from torchvision.utils import save_image

    from kornia.color import lab_to_rgb

    folder = os.path.join("data", "test")
    dataLoader = getDataLoader(folder, numWorks=6)

    start = time.time()
    
    for i, (l, ab) in enumerate(dataLoader):
        print(torch.max(l), torch.max(ab))
        print(torch.min(l), torch.min(ab))
        print(l.shape, ab.shape)
        print("----------------------------------------------------------------")

    elapsed = time.time() - start
    print(f"Time took: {elapsed}, avgRunTime: {elapsed/(i+1)}")