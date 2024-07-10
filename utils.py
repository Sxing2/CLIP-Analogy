import os
import pickle

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

model2vecsize = {
    'ViT-B/32': 512,
    'ViT-B/16': 512,
    'ViT-L/14': 768,
    'RN50': 1024,
}

def clip_normalize(image,device):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def file2path(data_path, file:str, cls_str:str) -> str:
    path = os.path.join(
        data_path, cls_str, file
    )
    return path

def path2file(path:str) -> str:
    file = path.split("/")[-1]
    return file

def flatten_tuple(tuple_:tuple) -> tuple:
    # tuple_: (cls, (i,j)) -> output tuple: (cls, i, j)
    return (tuple_[0], tuple_[1][0], tuple_[1][1])

def pickle_dump(obj, dest_file):
    f = open(dest_file, "wb")
    pickle.dump(obj, f)
    f.close()
    
def pickle_load(read_file):
    f = open(read_file, "rb")
    obj = pickle.load(f)
    f.close()
    return obj

def load_image2(img_path, img_height=None,img_width =None):
    image = Image.open(img_path)
    if img_width is not None:
        image = image.resize((img_width, img_height))  # change image size to (3, img_size, img_size)
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        ])   
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image

def write_text(text:str, file:str, mode:str='a'):
    with open(file, mode) as write_file:
        write_file.write(text)

def write_config_file(args, config_file, config_of_interest=None):
    if config_of_interest is None:
        list_of_configs = [f'{key} : {vars(args)[key]}' for key in vars(args)]
    else:
        list_of_configs = [f'{key} : {vars(args)[key]}' for key in vars(args) \
                           if key in config_of_interest and vars(args)[key] is not None]
    with open(config_file, 'a') as write_file:
        [ write_file.write(f'{st}\n') for st in list_of_configs ]

def print_architecture(net:nn.Module, write_file:str):
    write_text("\nArchitecture:\n", write_file, 'a')
    write_text(str(net), write_file, 'a')
    return None

def print_parameters(net:nn.Module, write_file:str):
    with open(write_file, 'a') as file:
        file.write('\nParameters:\n')
        for name, param in net.named_parameters():
            text = name + "\t" + str(list(param.shape)) + "\t" + f"requires_grad = {param.requires_grad}\n"
            file.write(text)

def print_cls_onto_image(image, text:str):
    with image.convert("RGBA") as base:
        txt = Image.new("RGBA", base.size, (255, 255, 255, 0))
        fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 15)
        d = ImageDraw.Draw(txt)
        d.text((10, 10), text, font=fnt, fill=(255, 0, 0, 255)) # red colour text
        out = Image.alpha_composite(base, txt).convert("RGB")
    return out

def freeze(x):
    if hasattr(x, 'parameters'):
        for param in x.parameters():
            param.requires_grad = False
    else:
        x.requires_grad = False
    return None

def unfreeze(x):
    if hasattr(x, 'parameters'):
        for param in x.parameters():
            param.requires_grad = True
    else:
        x.requires_grad = True
    return None
