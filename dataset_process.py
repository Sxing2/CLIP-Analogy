import os
from tqdm import tqdm
from PIL import Image
import clip
import torch
from nltk.corpus import wordnet as wn
from torch.utils.data import Dataset, DataLoader
from utils import file2path, pickle_dump, pickle_load, freeze

from arguments import define_arguments
parser, args = define_arguments()

device = torch.device(args.device)
model, preprocess = clip.load(args.clip_vision_model, device, jit=False)
freeze(model)

def process_raw_dataset(dataset:str, dataset_path:str, split:str='train'):

    stored_path = f"./stored_files/{args.clip_vision_model.replace('/', '')}/{dataset.lower()}"
    if not os.path.exists(stored_path):
        os.makedirs(stored_path)

    # Specify output paths
    if dataset.lower() == 'imagenet-21k':

        train_cls2files_path = os.path.join(stored_path, "cls2files.pkl")
        train_cls2vecs_path = os.path.join(stored_path, "cls2vecs.pkl")
        train_cls2vecs_flattened_path = os.path.join(stored_path, "cls2vecs_flattened.pkl")

        val_cls2files_path, val_cls2vecs_path, val_cls2vecs_flattened_path = None, None, None

    else:
        train_cls2files_path = os.path.join(stored_path, "train_cls2files.pkl")
        train_cls2vecs_path = os.path.join(stored_path, "train_cls2vecs.pkl")
        train_cls2vecs_flattened_path = os.path.join(stored_path, "train_cls2vecs_flattened.pkl")

        val_cls2files_path = os.path.join(stored_path, "val_cls2files.pkl")
        val_cls2vecs_path = os.path.join(stored_path, "val_cls2vecs.pkl")
        val_cls2vecs_flattened_path = os.path.join(stored_path, "val_cls2vecs_flattened.pkl")

    def clean_class_name(name_str:str) -> str:
        if dataset.lower() in ['imagenet-21k', 'imagenet']:
            synset = wn.synset_from_pos_and_offset('n', int(name_str.split('(')[0].lstrip('n0')))
            clean_name = synset._name.split('.')[0].replace('_', ' ')
        else:
            clean_name = name_str.split("(")[-1].rstrip(")").replace("_", " ")
        return clean_name

    def get_class_names(train_path:str):
        if dataset.lower() == 'imagenet':
            mapping_file = os.path.join(dataset_path, 'classnames.txt')
            assert os.path.exists(mapping_file)

            str2clean, clean2str, categories = {}, {}, []
            with open(mapping_file, 'r') as file:
                for line in file.readlines():
                    str_name, class_name = line.split(' ', maxsplit=1)
                    class_name = class_name.split('\n')[0]
                    str2clean[str_name] = class_name
                    clean2str[class_name] = str_name
                    categories.append(class_name)
            
            print("print 10 class names and their synset idx:")
            for class_name in categories[:10]:
                print(f"{class_name}: {clean2str[class_name]}")
            
            class_dirs = os.listdir(train_path)
            for class_dir in class_dirs:
                assert class_dir in str2clean

            return str2clean, clean2str, categories
        
        elif dataset.lower() == 'imagenet-100':
            class_dirs = os.listdir(train_path)
            str2clean, clean2str, categories = {}, {}, []
            for class_dir in class_dirs:
                # class_dir e.g. n01558993(robin)
                class_name = class_dir.split('(')[-1].rstrip(')').replace('_', ' ') # e.g., 'robin'
                str2clean[class_dir] = class_name
                clean2str[class_name] = class_dir
                categories.append(class_name)
            return str2clean, clean2str, categories

    def get_cls2files(data_path:str, clean2str:dict):
        save_path = train_cls2files_path if split == "train" else val_cls2files_path
        if os.path.exists(save_path):
            cls2files = pickle_load(save_path)
        else:
            cls2files = {}
            for class_name in clean2str:
                cls_path = os.path.join(data_path, clean2str[class_name])
                files = os.listdir(cls_path)
                cls2files[class_name] = dict(
                    zip(
                        range(len(files)), # key: sample id ([0, num_images_per_cls])
                        list(
                            map(
                                lambda x: os.path.join(cls_path, x),
                                files
                            )
                        ) # value: path of an image file
                    )
                )
            pickle_dump(cls2files, save_path)
        return cls2files

    def get_cls2vecs(cls2files:dict):
        save_path = train_cls2vecs_path if split == "train" else val_cls2vecs_path

        class imagenet_cls(Dataset):
            def __init__(self, cls:str, id2file:dict, preprocess=None):
                self.id2file = id2file # a dict mapping from instance id to filepath
                self.preprocess = preprocess
                self.cls = cls # class name

            def __len__(self):
                return len(self.id2file)

            def __getitem__(self, idx):
                img_name = self.id2file[idx].split('/')[-1]
                img = Image.open(self.id2file[idx])
                image = self.preprocess(img) # [3, 224, 224]
                return img_name, image

        if not os.path.exists(save_path):
            print("Generating CLIP encodings for images.")
            cls2vecs = {} # key: class name, value: a dict mapping from image id to image embedding
            with torch.no_grad():
                for cls in tqdm(cls2files):
                    cls2vecs[cls] = {}
                    cls_dataset = imagenet_cls(cls, cls2files[cls], preprocess)
                    cls_loader = DataLoader(cls_dataset, batch_size=1024, shuffle=False)
                    batch_size = cls_loader.batch_size
                    for batch_id, (_, batch_imgs) in enumerate(cls_loader):
                        # batch_imgs: [bs, 3, 224, 224]
                        batch_img_embeds = model.encode_image(batch_imgs.to(device))
                        cls2vecs[cls].update(
                            dict(
                                zip(
                                    range(batch_id*batch_size, batch_id*batch_size + batch_img_embeds.shape[0]),
                                    batch_img_embeds.cpu().numpy()
                                )
                            )
                        )
            pickle_dump(cls2vecs, save_path)
        else:
            cls2vecs = pickle_load(save_path)
        return cls2vecs

    def flatten_cls2vecs(cls2vecs:dict):
        save_path = train_cls2vecs_flattened_path if split == "train" else val_cls2vecs_flattened_path
        if not os.path.exists(save_path):
            cls2vecs_flattened = {}
            for cls_name in cls2vecs:
                for img_id, img_vec in cls2vecs[cls_name].items():
                    key = (cls_name, img_id)
                    cls2vecs_flattened[key] = img_vec
            pickle_dump(cls2vecs_flattened, save_path)
        else:
            cls2vecs_flattened = pickle_load(save_path)
        return cls2vecs_flattened

    if dataset.lower() in ["imagenet-100", "imagenet"]:

        train_path = os.path.join(dataset_path, "train")
        val_path = os.path.join(dataset_path, "val")
        
        data_path = train_path if split == "train" else val_path
        
        str2clean, clean2str, categories = get_class_names(train_path)

        cls2files = get_cls2files(data_path, clean2str)
        cls2vecs = get_cls2vecs(cls2files)
        cls2vecs_flattened = flatten_cls2vecs(cls2vecs)

        return str2clean, clean2str, categories, cls2files, cls2vecs, cls2vecs_flattened

    elif dataset.lower() in ['imagenet-21k']:

        train_path = '/nfs/data_lambda/datasets/winter21_whole.tar.gz' # path to tar file without decompressing it
        val_path = None

        str2clean, clean2str, categories = get_class_names(train_path)
        cls2files = get_cls2files(categories, train_path, clean2str)
        cls2vecs = get_cls2vecs(cls2files)
        cls2vecs_flattened = None

        return str2clean, clean2str, categories, cls2files, cls2vecs, cls2vecs_flattened

if __name__ == "__main__":
    pass
