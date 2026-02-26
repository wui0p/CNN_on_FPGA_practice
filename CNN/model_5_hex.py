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
image, label = test_data[0]
image_q = torch.round(image * W_SCALE).to(torch.int32)
save_hex(image_q, "test_image_0.hex", W_LENGTH)
print(f"Image 0 (Label: {label}) saved to hex!")


# FOLDING BLOCKS
W_b1, B_b1 = get_folded_params('conv_b1.0', 'conv_b1.1', sd, B_SCALE[0])
save_hex(W_b1, "conv_b1_w.hex", W_LENGTH)
save_hex(B_b1, "conv_b1_b.hex", 32)
W_b2, B_b2 = get_folded_params('conv_b2.0', 'conv_b2.1', sd, B_SCALE[1])
save_hex(W_b2, "conv_b2_w.hex", W_LENGTH)
save_hex(B_b2, "conv_b2_b.hex", 32)

W_b3, B_b3 = get_folded_params('conv_b3.0', 'conv_b3.1', sd, B_SCALE[2])
save_hex(W_b3, "conv_b3_w.hex", W_LENGTH)
save_hex(B_b3, "conv_b3_b.hex", 32)
W_b4, B_b4 = get_folded_params('conv_b4.0', 'conv_b4.1', sd, B_SCALE[3])
save_hex(W_b4, "conv_b4_w.hex", W_LENGTH)
save_hex(B_b4, "conv_b4_b.hex", 32)


# GOLDEN PATTERN
# with torch.inference_mode():

#     x = torch.round(image * W_SCALE).to(torch.int32)
        
#     x = F.conv2d(x.double(), W_b1.double(), B_b1.double(), padding=1)
#     print(f"B1 - Max after Conv: {x.max().item()}, Min: {x.min().item()}")
#     x = x.to(torch.int32) >> Q_SHIFT[0] 
#     x = torch.clamp(F.relu(x), 0, MAX_VAL)
#     print(f"B1 - Max after Shift/ReLU: {x.max().item()}, Min: {x.min().item()}")
#     save_hex(x.to(torch.int32), "golden_b1.hex", W_LENGTH)
    
#     x = F.conv2d(x.double(), W_b2.double(), B_b2.double(), padding=1)
#     print(f"B2 - Max after Conv: {x.max().item()}, Min: {x.min().item()}")
#     x = x.to(torch.int32) >> Q_SHIFT[1]
#     x = torch.clamp(F.relu(x), 0, MAX_VAL)
#     save_hex(x.to(torch.int32), "golden_b2_pre.hex", W_LENGTH)
#     x = F.max_pool2d(x.double(), kernel_size=2)
#     print(f"B2 - Max after Shift/ReLU: {x.max().item()}, Min: {x.min().item()}")
#     save_hex(x.to(torch.int32), "golden_b2.hex", W_LENGTH)
    
#     x = F.conv2d(x.double(), W_b3.double(), B_b3.double(), padding=1)
#     print(f"B3 - Max after Conv: {x.max().item()}, Min: {x.min().item()}")
#     x = x.to(torch.int32) >> Q_SHIFT[2]
#     x = torch.clamp(F.relu(x), 0, MAX_VAL)
#     print(f"B3 - Max after Shift/ReLU: {x.max().item()}, Min: {x.min().item()}")
#     save_hex(x.to(torch.int32), "golden_b3.hex", W_LENGTH)
    
#     x = F.conv2d(x.double(), W_b4.double(), B_b4.double(), padding=1)
#     print(f"B4 - Max after Conv: {x.max().item()}, Min: {x.min().item()}")
#     x = x.to(torch.int32) >> Q_SHIFT[3]
#     x = torch.clamp(F.relu(x), 0, MAX_VAL)
#     x = F.max_pool2d(x.double(), kernel_size=2)
#     print(f"B4 - Max after Shift/ReLU: {x.max().item()}, Min: {x.min().item()}")
#     save_hex(x.to(torch.int32), "golden_b4.hex", W_LENGTH)

#     print("Golden outputs generated successfully!")

#     # Extract Classifier Params (No folding needed)
# W_cls = sd['classifier.1.weight']
# B_cls = sd['classifier.1.bias']

# W_cls_q = torch.round(W_cls * W_SCALE).to(torch.int32).clamp(-MAX_VAL, MAX_VAL)
# B_cls_q = torch.round(B_cls * B_SCALE[4]).to(torch.int32)

# save_hex(W_cls_q, "classifier_w.hex", W_LENGTH)
# save_hex(B_cls_q, "classifier_b.hex", 32)

# # Run Classifier math
# x_flat = x.view(1, -1)
# logits = torch.matmul(x_flat.double(), W_cls_q.double().t()) + B_cls_q.double()
# prediction = torch.argmax(logits)

# # inside CLASSIFIER CHECK
# print("--- FINAL SCORES ---")
# for i in range(len(class_names)):
#     score = logits[0][i].item()
#     marker = "  <-- WINNER" if i == torch.argmax(logits) else ""
#     print(f"Class {i} ({class_names[i]}): {score:.0f}{marker}")

# print(f"--- QUANTIZED MODEL RESULT ---")
# print(f"Predicted Class: {prediction.item()}")
# print(f"Actual Label: {label}")

W_cls = sd['classifier.1.weight']
B_cls = sd['classifier.1.bias']
W_cls_q = torch.round(W_cls * W_SCALE).to(torch.int32).clamp(-MAX_VAL, MAX_VAL)
B_cls_q = torch.round(B_cls * B_SCALE[4]).to(torch.int32)

print(f"\n--- STARTING FULL ACCURACY CHECK ---")
correct = 0
total = 0

with torch.inference_mode():
    # 1. Loop through the batches
    for images, labels in tqdm(test_loader):
        
        # FIX #1: Use 'images' (the batch), not 'image_q' (the single image)
        # We also need to quantize this batch on the fly.
        x = torch.round(images * W_SCALE).to(torch.int32)
        
        # BLOCK 1
        x = F.conv2d(x.double(), W_b1.double(), B_b1.double(), padding=1)
        x = x.to(torch.int32) >> Q_SHIFT[0] 
        x = torch.clamp(F.relu(x), 0, MAX_VAL) 
    
        x = F.conv2d(x.double(), W_b2.double(), B_b2.double(), padding=1)
        x = x.to(torch.int32) >> Q_SHIFT[1]
        x = torch.clamp(F.relu(x), 0, MAX_VAL)
        x = F.max_pool2d(x.double(), kernel_size=2)
    

        # BLOCK 2
        x = F.conv2d(x.double(), W_b3.double(), B_b3.double(), padding=1)
        x = x.to(torch.int32) >> Q_SHIFT[2]
        x = torch.clamp(F.relu(x), 0, MAX_VAL)
    
        x = F.conv2d(x.double(), W_b4.double(), B_b4.double(), padding=1)
        x = x.to(torch.int32) >> Q_SHIFT[3]
        x = torch.clamp(F.relu(x), 0, MAX_VAL)
        x = F.max_pool2d(x.double(), kernel_size=2)

        # FIX #2: Flatten based on Batch Size (x.size(0)), not '1'
        # view(1, -1) collapses the whole batch into one giant vector.
        x_flat = x.view(x.size(0), -1)
        
        logits = torch.matmul(x_flat.double(), W_cls_q.double().t()) + B_cls_q.double()
        
        # FIX #3: Argmax across dimension 1 (Classes), not default flat
        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

final_acc = 100 * correct / total
print(f"Final Accuracy: {final_acc:.2f}%")
# Final Accuracy: 87.63%