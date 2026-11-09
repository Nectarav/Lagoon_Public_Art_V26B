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

# folder paths
data_folder = 'C:/Users/Owner/Documents/Lagoon_Public_Art_V26B/Photos/'
coats_of_arms_path = data_folder + 'Coats_of_Arms'

# training directory
train_dir = os.path.join(coats_of_arms_path, coats_of_arms_path + '/Train')

# file names
classes = os.listdir(train_dir)

# show images
def show_sample_images(num_classes = 4):
	fig, axes = plt.subplots(num_classes, 5, figsize = (15, 3 * num_classes))
	for i, class_name in enumerate(random.sample(classes, num_classes)):
		class_path = os.path.join(train_dir, train_dir + '/' + class_name)
		images = os.listdir(class_path)

		for j in range(5):
			img_path = os.path.join(class_path, class_path + '/' + images[j])
			img = Image.open(img_path)
			axes[i,j].imshow(img)
			axes[i,j].set_title(class_name)
			axes[i,j].axis('off')
	plt.tight_layout()

show_sample_images()