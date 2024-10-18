import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2

import os
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset


def create_dataloader(resize_size, batch_size=4,transform_mean=[0.485, 0.456, 0.406],transform_std=[0.229, 0.224, 0.225]):

    # Define transformations
    image_transform = transforms.Compose([
        transforms.Resize(resize_size),  # Resize images to 256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_mean, std=transform_std)
    ])

    mask_transform = transforms.Compose([
        transforms.Resize(resize_size),  # Resize masks to 256x256
        transforms.ToTensor()
    ])
    # Load COCO annotations for training and validation sets
    train_json_path = 'train/_annotations.coco.json'
    val_json_path = 'valid/_annotations.coco.json'
    train_img_dir = 'train'
    val_img_dir = 'valid'

    train_images, train_annotations, train_categories = load_coco_annotations(train_json_path)
    val_images, val_annotations, val_categories = load_coco_annotations(val_json_path)


    # Create datasets and dataloaders
    train_dataset = BrainData(train_images, train_annotations, train_img_dir, image_transform=image_transform,mask_transform=mask_transform) 
    val_dataset = BrainData(val_images, val_annotations, val_img_dir, image_transform=image_transform,mask_transform=mask_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader

def load_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    return images, annotations, categories


class BrainData(Dataset):
    def __init__(self, images, annotations, img_dir, image_transform=None, mask_transform=None):
        self.images = images
        self.annotations = annotations
        self.img_dir = img_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_ids = list(images.keys())
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
   # Create an empty mask
        mask = Image.new('L', (img_info['width'], img_info['height']), 0)

        draw = ImageDraw.Draw(mask)
        
        # Fill the mask with the segmentation annotations
        for ann in self.annotations:
            if ann['image_id'] == img_id:
                segmentation = ann['segmentation']
                for seg in segmentation:
                    poly = np.array(seg).reshape((len(seg) // 2, 2))
                    draw.polygon([tuple(p) for p in poly], outline=1, fill=1)
        
        # Convert the mask to a NumPy array
        mask = np.array(mask)
        mask[mask > 0] = 255

        mask = Image.fromarray(mask.astype(np.uint8))
        if self.image_transform:
            image = self.image_transform(image)
        if  self.mask_transform:
            mask = self.mask_transform(mask)
            mask = mask # Remove the channel dimension and convert to long
        
        return image, mask
    
    def get_img_info(self, idx):
        img_id = self.image_ids[idx]
        return self.images[img_id]
    def get_bounding_boxes(self, idx):
        img_id = self.image_ids[idx]
        return [ann for ann in self.annotations if ann['image_id'] == img_id]
    def plot(self, idx):
        img, mask = self[idx]
        # Denormalize the image for plotting
        transform_mean = [0.485, 0.456, 0.406]
        transform_std = [0.229, 0.224, 0.225]
        img = img.numpy().transpose(1, 2, 0)
        img = img * transform_std + transform_mean
        img = (img * 255).astype(np.uint8)

        mask = mask.numpy().squeeze()
        y_indices, x_indices = np.where(mask == 1)
        if len(y_indices) > 0 and len(x_indices) > 0:
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            # Draw the bounding box on the image
            img_with_box = img.copy()
            cv2.rectangle(img_with_box, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Red box
            
            # Overlay the mask on the image
            red_overlay = np.zeros_like(img)
            red_overlay[..., 0] = 255  # Red channel
            alpha = np.zeros_like(mask, dtype=np.float32)
            alpha[mask > 0] = 0.3  # Adjust transparency here (0.3 for 30% transparency)
            
            # Plot the image with the red overlay and bounding box
            plt.figure(figsize=(10, 5))
            plt.imshow(img_with_box)
            plt.axis('off')
            plt.show()
        else:
            print("No ones found in the mask.")








# Example usage
if __name__ == "__main__":
    # Define dummy data for testing
    images = {
        1: {'file_name': 'dummy_image.png', 'width': 256, 'height': 256}
    }
    annotations = [
        {'image_id': 1, 'segmentation': [[50, 50, 200, 50, 200, 200, 50, 200]]}
    ]
    img_dir = 'dummy_dir'
    
    # Create a dummy image file for testing
    os.makedirs(img_dir, exist_ok=True)
    dummy_image_path = os.path.join(img_dir, 'dummy_image.png')
    dummy_image = Image.new('RGB', (256, 256), color='white')
    dummy_image.save(dummy_image_path)
    
    # Define transformations
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Create an instance of the dataset
    dataset = BrainData(images, annotations, img_dir, image_transform=image_transform, mask_transform=mask_transform)
    
    # Get the first item from the dataset
    image, mask = dataset[0]
    
    # Plot the image and mask
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image.permute(1, 2, 0).numpy())
    ax[0].set_title('Image')
    ax[1].imshow(mask.squeeze().numpy(), cmap='gray')
    ax[1].set_title('Mask')
    plt.show()