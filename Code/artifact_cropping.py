######################################
## V26B.ART Image Cropping DL Model ##
######################################
## Developed by M. Chaney, 2026 ##
##################################

# imports
import os
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# folder paths
dataset_path = r'C:/Users/Owner/Documents/Lagoon_Public_Art_V26B/Photos/'

# training directory
train_dir = os.path.join(dataset_path, 'Train')

# file names
classes = os.listdir(train_dir)

transform = transforms.Compose([
	transforms.Resize((3008,2000)),
	transforms.ToTensor(),
	transforms.Normalize([0.485,0.465,0.406],
		[0.229,0.224,0.225])
])

train_dataset = ImageFolder(os.path.join(dataset_path, 'Train'), transform = transform)
val_dataset = ImageFolder(os.path.join(dataset_path, 'Val'), transform = transform)
test_dataset = ImageFolder(os.path.join(dataset_path, 'Test'), transform = transform)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 0)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # works with nvidia gpus

model = models.resnet18(pretrained = True)

for param in model.parameters():
	param.requires_grad = False

num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

# function for training an epoch
def train_one_epoch(model, loader):
	model.train()
	running_loss = 0

	for images, labels in loader:
		images, labels = images.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()

	return running_loss / len(loader)

def evaluate(model, loader): 
	model.eval()
	correct = 0
	total = 0

	with torch.no_grad():
		for images, labels in loader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	return correct / total

epochs = 5

for epoch in range(epochs):
	train_loss = train_one_epoch(model, train_loader)
	val_acc = evaluate(model, val_loader)
	print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

