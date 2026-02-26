import sys
import torch
from torch import nn
import numpy as np

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

import matplotlib

from tqdm import tqdm
from timeit import default_timer as Timer

import requests

from pathlib import Path
from helper_functions import accuracy_fn

train_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform = None
)
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

BATCH = 32
train_dataloader = DataLoader(dataset = train_data,
                              batch_size = BATCH,
                              shuffle = True)
test_dataloader = DataLoader(dataset = test_data,
                             batch_size = BATCH,
                             shuffle = False)
class_names = train_data.classes

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               acc_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += acc_fn(y_true = y,
                            y_pred = y_pred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train Loss: {train_loss:.4f}, Train acc: {train_acc:.2f}%")

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              acc_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += acc_fn(y_true = y,
                               y_pred = test_pred.argmax(dim=1))
        
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test Loss: {test_loss:.4f}, Test acc: {test_acc:.2f}%\n")

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
  total_time = end - start
  print(f"Train time on {device}: {total_time:.3f} seconds")
  print(f"Train time on {device}: {total_time/60:.1f} minutes")
  return total_time


class FashionModelV1(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units_1: int,
                 hidden_units_2: int,
                 output_shape: int):
        super().__init__()
        self.conv_b1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units_1,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units_1),
            nn.Hardtanh(min_val=0.0, max_val=1.9999)
        )
        self.conv_b2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units_1,
                      out_channels=hidden_units_2,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units_2),
            nn.Hardtanh(min_val=0.0, max_val=1.9999),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_b3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units_2,
                      out_channels=hidden_units_2,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units_2),
            nn.Hardtanh(min_val=0.0, max_val=1.9999)
        )
        self.conv_b4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units_2,
                      out_channels=hidden_units_2,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units_2),
            nn.Hardtanh(min_val=0.0, max_val=1.9999),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_b5 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units_2,
                      out_channels=hidden_units_2,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units_2),
            nn.Hardtanh(min_val=0.0, max_val=1.9999)
        )
        self.conv_b6 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units_2,
                      out_channels=hidden_units_2,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units_2),
            nn.Hardtanh(min_val=0.0, max_val=1.9999)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units_2*7*7,
                      out_features=output_shape)
        )
    def forward(self, x):
        return self.classifier(self.conv_b6(self.conv_b5(self.conv_b4(self.conv_b3(self.conv_b2(self.conv_b1(x)))))))

model_3 = FashionModelV1(input_shape = 1,
                         hidden_units_1 = 64,
                         hidden_units_2 = 128,
                         output_shape = len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer_3 = torch.optim.SGD(params=model_3.parameters(),
                            lr = 0.1)

time_start = Timer()
epochs = 10
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(model = model_3,
               data_loader = train_dataloader,
               loss_fn = loss_fn,
               optimizer = optimizer_3,
               acc_fn = accuracy_fn,
               device = device)
    test_step(model = model_3,
              data_loader = test_dataloader,
              loss_fn = loss_fn,
              acc_fn = accuracy_fn,
              device = device)
time_end = Timer()
total_time = print_train_time(start = time_start,
                              end = time_end,
                              device = device)

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)
MODEL_NAME = "fashion_model_3.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_3.state_dict(),
           f=MODEL_SAVE_PATH)

# With hidden units = 64 -> 128
# Train Loss: 0.0901, Train acc: 96.72%
# Test Loss: 0.1970, Test acc: 93.51%


'''
model_load_3 = FashionModelV1(input_shape = 1,
                              hidden_units_1 = 64,
                              hidden_units_2 = 128,
                              output_shape = len(class_names)).to(device)
model_load_3.load_state_dict(torch.load("models/fashion_model_3.pth"))
model_load_3.to(device)

image, label = test_data[0]
input_tensor = image.unsqueeze(dim=0).to(device)
model_3.eval() # Set model to evaluation mode (important for BatchNorm!)
with torch.inference_mode():
    # Pass the image through the model
    logits = model_load_3(input_tensor)
    
    # Convert logits to probabilities (optional) and get the predicted label
    pred_probs = torch.softmax(logits, dim=1)
    pred_label = torch.argmax(pred_probs, dim=1)

print(f"Predicted label: {pred_label.item()} ({class_names[pred_label]})")
print(f"Actual label: {label} ({class_names[label]})")
'''