import os
import torch
import numpy as np
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms

from dataset import get_data_transforms
from resnet_TTA import  wide_resnet50_2
from de_resnet import  de_wide_resnet50_2
from dataset import VisADataset, VisADatasetOOD
from test import  evaluation_ATTA


import torch
import numpy as np
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

# Anomaly map calculation
def cal_anomaly_map(fs_list, ft_list, out_size=224):
    anomaly_map = np.ones([out_size, out_size])
    print(fs_list[0].shape)
    for i in range(len(ft_list)):
        fs = fs_list[i]  # fs: [batch_size, num_channels, height, width]
        ft = ft_list[i]  # ft: [batch_size, num_channels, height, width]
        
        # Min-max normalization
        fs_min, fs_max = fs.min(), fs.max()
        ft_min, ft_max = ft.min(), ft.max()
        fs_norm = (fs - fs_min) / (fs_max - fs_min + 1e-8)
        ft_norm = (ft - ft_min) / (ft_max - ft_min + 1e-8)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        fs_norm = fs_norm + epsilon
        ft_norm = ft_norm + epsilon
        
        # Compute KL divergence
        kl_div = torch.sum(ft_norm * (torch.log(ft_norm) - torch.log(fs_norm)), dim=1)
        
        # Resize KL divergence map
        kl_div = torch.unsqueeze(kl_div, dim=1)  # Add channel dimension back for interpolation
        kl_div = F.interpolate(kl_div, size=out_size, mode='bilinear', align_corners=True)
        kl_div = kl_div[0, 0, :, :].to('cpu').detach().numpy()
        
        # Normalize KL divergence map between 0 and 1
        kl_div_min, kl_div_max = kl_div.min(), kl_div.max()
        kl_div = (kl_div - kl_div_min) / (kl_div_max - kl_div_min + 1e-8)
        
        # Update anomaly map
        anomaly_map *= kl_div
    
    return anomaly_map


# Min-max normalization function
def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

# Function to convert anomaly map to a heatmap
def anomaly_map_to_heatmap(anomaly_map, threshold=0.6):
    normed_map = min_max_norm(anomaly_map)
    normed_map[normed_map < threshold] = 0  # Apply thresholding
    heatmap = cv2.applyColorMap(np.uint8(255 * normed_map), cv2.COLORMAP_JET)
    return heatmap

# Function to overlay heatmap on image
def overlay_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlay

# Main function for inference and visualization
def predict_anomaly(encoder, bn, decoder, image_path, normal_image_path,device, img_size=224, lamda=0.5, threshold=0.6):
    bn.eval()
    decoder.eval()

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    normal_image = Image.open(normal_image_path).convert("RGB")
    trans = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = trans(img).unsqueeze(0).to(device)
    normal_image = trans(normal_image).unsqueeze(0).to(device)
    #normal_image = normal_image.to(device)
    # Get normal image (if needed, use actual normal image logic)
    #normal_image = img_tensor.clone().to(device)

    with torch.no_grad():
        inputs = encoder(img_tensor, normal_image, "test", lamda=lamda)
        outputs = decoder(bn(inputs))
        anomaly_map = cal_anomaly_map(inputs, outputs, img_size)
        heatmap = anomaly_map_to_heatmap(anomaly_map, threshold)

    # Convert tensor image to numpy array
    img_np = np.array(img)

    # Overlay heatmap on original image
    overlayed_image = overlay_heatmap(img_np, heatmap)

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img_np)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Anomaly Map')
    plt.imshow(overlayed_image)
    plt.axis('off')

    plt.show()

    return np.max(anomaly_map)  # Predicted anomaly score

#load model
_class_ = 'pcb1'
ckp_path = './checkpoints/' + 'visa_DINL_' + str(_class_) + '_1.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder, bn = wide_resnet50_2(pretrained=True)
encoder = encoder.to(device)
bn = bn.to(device)
encoder.eval()
decoder = de_wide_resnet50_2(pretrained=False)
decoder = decoder.to(device)

#load checkpoint
ckp = torch.load(ckp_path)
for k, v in list(ckp['bn'].items()):
    if 'memory' in k:
        ckp['bn'].pop(k)
decoder.load_state_dict(ckp['decoder'])
bn.load_state_dict(ckp['bn'])

lamda = 0.5

folder_path = f'./visa/1cls/{_class_}/test/bad'  # Update the path as needed
im_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) ]

folder_path_normal = f'./visa/1cls/{_class_}/train/good'  # Update the path as needed
im_paths_normal = [os.path.join(folder_path_normal, file) for file in os.listdir(folder_path_normal) ]
image_path_normal = im_paths_normal[0]

image_path = im_paths[0] # Set your image path
anomaly_score = predict_anomaly(encoder, bn, decoder, image_path, image_path_normal, device)
print("Anomaly score:", anomaly_score)
