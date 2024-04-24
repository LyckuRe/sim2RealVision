from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np

class EncoderDataset(Dataset):
    def __init__(self, root_dir, transform=None, dataset_size=256):
        """
        Args:
            root_dir (string): Directory with all the images and masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # List image files
        self.image_files = [f"line_image_{i}.jpeg" for i in range(dataset_size)]
        self.mask_files = [f"mask_image_{i}.jpeg" for i in range(dataset_size)]
        self.label_files = [f"point_label_{i}.txt" for i in range(dataset_size)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_name = os.path.join(self.root_dir, "images", self.image_files[idx])
        image = Image.open(img_name)
        image_np = np.array(image)
        image_np = np.transpose(image_np, (2, 0, 1))

        # Load mask
        mask_name = os.path.join(self.root_dir, "masks", self.mask_files[idx])
        mask = Image.open(mask_name)
        mask_np = np.array(mask)

        # Load label
        label_name = os.path.join(self.root_dir, "labels", self.label_files[idx])
        with open(label_name, 'r') as f:
            data = f.read()[1:-1]
        label_np = np.array([float(i) for i in data.split(', ')], dtype=np.float32)

        # Convert to torch tensors with dtype=torch.uint8
        image_tensor = torch.tensor(image_np, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32)
        label_tensor = torch.tensor(label_np, dtype=torch.float32)

        sample = {'image': image_tensor, 'mask': mask_tensor, 'label': label_tensor}

        if self.transform:
            sample = self.transform(sample)

        return sample