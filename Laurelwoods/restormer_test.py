import torch
from torchinfo import summary
from glob import glob
import cv2

from basicsr import models

# Load model

model = models.archs.restormer_arch.Restormer(
    inp_channels=3, 
    out_channels=3, 
    dim=32,
    num_blocks=[1, 2, 4, 7], 
    num_refinement_blocks=4,
    heads=[1, 2, 4, 8], 
    ffn_expansion_factor=2.66, 
    bias=False, 
    LayerNorm_type='BiasFree', 
    dual_pixel_task=False)

path = r'F:\Laurelwoods IDE\Univ-ML-in-practice-competition\Laurelwoods\experiments\Restormer\models\net_g_latest.pth'
params = torch.load(path)

model.load_state_dict(params['params'])
model.eval()

# Model summary
summary(model)


# Run test

test_path = r'F:\Laurelwoods IDE\Univ-ML-in-practice-competition\competition datasets\test_scan'

# for img in glob(test_path + '/*'):
#     img = cv2.imread(img)
#     pred = model(img)
#     print(pred)

#     break

sample = cv2.imread(glob(test_path + '/*')[0])
pred = model(sample)
print(pred)
