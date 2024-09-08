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


# Anomaly map calculation
def cal_anomaly_map(fs_list, ft_list, out_size=224):
    anomaly_map = np.ones([out_size, out_size])
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        anomaly_map *= a_map
    return anomaly_map

# Main function for inference
def predict_anomaly(encoder, bn, decoder, image_path, device, img_size=224, lamda=0.5):
    bn.eval()
    decoder.eval()

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    trans = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = trans(img).unsqueeze(0).to(device)

    # Get normal image (if needed, use actual normal image logic)
    normal_image = img.clone().to(device)

    with torch.no_grad():
        inputs = encoder(img, normal_image, "test", lamda=lamda)
        outputs = decoder(bn(inputs))
        anomaly_map = cal_anomaly_map(inputs, outputs, img.shape[-1])
        return np.max(anomaly_map)  # Predicted anomaly score


#load model
_class_ = 'candle'
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

folder_path = './visa/1cls/candle/test/bad'  # Update the path as needed
im_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) ]


image_path = im_paths[0] # Set your image path
anomaly_score = predict_anomaly(encoder, bn, decoder, image_path, device)
print("Anomaly score:", anomaly_score)
