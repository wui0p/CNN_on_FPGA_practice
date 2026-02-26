# import torch

# model_weights = torch.load("models/fashion_model_5.pth", weights_only=True)

# for layer_name in model_weights.keys():
#     print(layer_name)


import torch
from torch import nn
import numpy as np
from tqdm import tqdm

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

import torch.nn.functional as F

W_SCALE = 2**14 #Q2.14
B_SCALE = [2**28,
           2**28,
           2**28,
           2**28,
           2**28]   #classifier
Q_SHIFT = [14, 14, 14, 14]
MAX_VAL = 32767
W_LENGTH = 16

def get_folded_params(conv_name, bn_name, sd, b_sc):
    W = sd[f'{conv_name}.weight']
    B = sd[f'{conv_name}.bias']
    rm = sd[f'{bn_name}.running_mean']
    rv = sd[f'{bn_name}.running_var']
    gamma = sd[f'{bn_name}.weight']
    beta = sd[f'{bn_name}.bias']
    eps = 1e-5

    denom = torch.sqrt(rv + eps)
    fold_multi = gamma / denom
    W_folded = W * fold_multi.view(-1, 1, 1, 1)
    B_folded = ((B - rm) * fold_multi) + beta

    W_q = torch.round(W_folded * W_SCALE).to(torch.int32).clamp(-MAX_VAL, MAX_VAL)
    B_q = torch.round(B_folded * b_sc).to(torch.int32)
    return W_q , B_q

def save_hex(tensor, filename, width):
    with open(f"hardware/model_5/{filename}", "w") as f:
        for val in tensor.view(-1):
            v = val.item()
            if v < 0:
                v = (1 << width) + v
            hex_val = f"{v:0{width//4}x}"
            f.write(f"{hex_val}\n")


sd = torch.load("models/fashion_model_5.pth", weights_only=True)
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)
test_loader = DataLoader(test_data,
                         batch_size=32,
                         shuffle=False)
class_names = test_data.classes
image, label = test_data[2]
image_q = torch.round(image * W_SCALE).to(torch.int32)
save_hex(image_q, "test_image_2.hex", W_LENGTH)
print(f"Image 1 (Label: {label}) saved to hex!")