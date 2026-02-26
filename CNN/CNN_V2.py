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
  return total_time

class FashionModelV2(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.conv_b1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.Hardtanh(min_val=0.0, max_val=1.9999)
        )
        self.conv_b2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.Hardtanh(min_val=0.0, max_val=1.9999),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_b3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.Hardtanh(min_val=0.0, max_val=1.9999)
        )
        self.conv_b4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.Hardtanh(min_val=0.0, max_val=1.9999),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )
    def forward(self, x):
        x = self.conv_b1(x)
        x = self.conv_b2(x)
        x = self.conv_b3(x)
        x = self.conv_b4(x)
        x = self.classifier(x)
        return x

model_4 = FashionModelV2(input_shape = 1,
                        hidden_units = 8,
                        output_shape = len(class_names)).to(device)
model_5 = FashionModelV2(input_shape = 1,
                         hidden_units = 4,
                         output_shape = len(class_names)).to(device)
                        

loss_fn = nn.CrossEntropyLoss()
optimizer_4 = torch.optim.SGD(params=model_4.parameters(),
                              lr = 0.1)
optimizer_5 = torch.optim.SGD(params=model_5.parameters(),
                              lr = 0.1)

# time_start = Timer()
# epochs = 10
# for epoch in tqdm(range(epochs)):
#     print(f"Epoch: {epoch}\n---------")
#     train_step(model = model_4,
#                data_loader = train_dataloader,
#                loss_fn = loss_fn,
#                optimizer = optimizer_4,
#                acc_fn = accuracy_fn,
#                device = device)
#     test_step(model = model_4,
#               data_loader = test_dataloader,
#               loss_fn = loss_fn,
#               acc_fn = accuracy_fn,
#               device = device)
# time_end = Timer()
# total_time = print_train_time(start = time_start,
#                               end = time_end,
#                               device = device)

# MODEL_PATH = Path("models")
# MODEL_PATH.mkdir(parents=True,
#                  exist_ok=True)
# MODEL_NAME = "fashion_model_4.pth"
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# print(f"Saving model to: {MODEL_SAVE_PATH}")
# torch.save(obj=model_4.state_dict(),
#            f=MODEL_SAVE_PATH)

# # With hidden units = 8
# # Train Loss: 0.2241, Train acc: 91.83%
# # Test Loss: 0.2641, Test acc: 90.57%

time_start = Timer()
epochs = 10
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(model = model_5,
               data_loader = train_dataloader,
               loss_fn = loss_fn,
               optimizer = optimizer_5,
               acc_fn = accuracy_fn,
               device = device)
    test_step(model = model_5,
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
MODEL_NAME = "fashion_model_5.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_5.state_dict(),
           f=MODEL_SAVE_PATH)

# With hidden units = 8
# Train Loss: 0.2856, Train acc: 89.64%
# Test Loss: 0.3123, Test acc: 89.01%