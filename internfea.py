import os
import shutil
from PIL import Image
import torch
from transformers import AutoModel, CLIPImageProcessor

image_processor = CLIPImageProcessor.from_pretrained("InternViT-6B-448px-V1-5")
path = "InternViT-6B-448px-V1-5"
base_model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).cuda().eval()

def f(image_path):
    image = Image.open(image_path).convert('RGB')
    # image = image.transpose(Image.FLIP_LEFT_RIGHT)
    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
    fea = base_model(pixel_values.to(torch.bfloat16).cuda())
    return fea

def process_images(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith('.jpg'):
                fea = f(file_path)
                output_file_path = os.path.join(output_path, file)
                torch.save(fea[1].detach().cpu(), output_file_path)

input_dir = ['/Ego4D_LookAtMe/face_imgs', '/Ego4D_LookAtMe/videos_challenge']
output_dir = ['/Ego4D_LookAtMe/internvitfea', '/Ego4D_LookAtMe/internvitfea_test']
for i in range(len(input_dir)):
    process_images(input_dir[i], output_dir[i])
