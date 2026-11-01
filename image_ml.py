#####################################################
## V26B.ART: MACHINE LEARNING ARTIFACT RECOGNITION ##
#####################################################

# imports
import numpy as np 
import torch
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.io import decode_image

# datasets
datafolder = "C:/Users/mchan/Lagoon_Public_Art/Photos/"
burano_data = datafolder + "Burano_Images"
mazzorbo_data = datafolder + "Mazzorbo_Images"
from training_data.csv import *

# labelling
labels_map = {
	0: "Coat of Arms",
	1: "Patere",
	2: "Formelle",
	3: "Cross",
	4: "Confraternity Symbol",
	5: "Fragment"
}

print(burano_data)

kill
# test visuals
# figure = plt.figure(figsize=(8,8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
# 	sample_idx = torch.randint(len(burano_data), size=(1,)).item()
# 	img, label = burano_data[sample_idx]
# 	figure.add_subplot(rows, cols, i)
# 	plt.title(labels_map[label])
# 	plt.axis("off")
# 	plt.imshow(img.squeeze())
# plt.show()

# class CustumImageDataset(Dataset):
# 	def __init__(self, csv_data, img_dir, transform=None, target_transform=None):
# 		self.img_labels = pd.read_csv(csv_data)
# 		self.img_dir = img_dir
# 		self.transform = transform
# 		self.target_transform = target_transform

# 	def __len__(self):
# 		return len(self.img_labels)

# 	def __getitem__(self, idx):
# 		img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#  		image = decode_image(img_path)
# 		label = self.img_labels.iloc[idx, 1]
# 		if self.transform:
# 		image = self.transform(image)
# 		if self.target_transform:
# 		label = self.target_transform(label)
# 		return image, label