import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from glob import glob
import cv2
import os
import numpy as np

from basicsr import models

class TestDataset(Dataset):
    def __init__(self, path, transforms=ToTensor()):
        self.paths = glob(path + "/*")
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])
        img = self.transforms(img)
        return img
    
test_path = "competition datasets/test_scan"
test_data = TestDataset(test_path)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Load model

model = models.archs.kbnet_s_arch.KBNet_s(
    width=16,
    middle_blk_num=7,
    lightweight=True
)

DEVICE = torch.device("cuda")
model.to(DEVICE)

model_id = "net_g_1000"
model_path = "results/3.1" 

params = torch.load(f"{model_path}/{model_id}.pth")

model.load_state_dict(params['params'])
model.eval()

test_output_path = model_path + "/test_output"
os.makedirs(test_output_path, exist_ok=True)

i = 0
for test_data in test_loader:
    test_data = test_data.to(DEVICE)
    pred = model(test_data)
    for p in pred:
        np.save(f"{test_output_path}/{i}", p.cpu().detach().numpy())
        # cv2.imwrite(f"{test_output_path}/{i}.png", p)
        i += 1
