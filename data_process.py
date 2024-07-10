import os
import random
import time
from tqdm import tqdm
import math
import numpy as np
import torch
import clip
import PIL
from PIL import Image

from utils import pickle_dump, pickle_load, freeze

from arguments import define_arguments
_, args = define_arguments()

device = torch.device(args.device)
clip_model, preprocess = clip.load(args.clip_vision_model, device, jit=False)
freeze(clip_model)

def vector_normalize(X:np.ndarray, dim=-1):
    # X: [dim] or [N, dim]
    if len(X.shape)==1:
        X_normalized = X / np.linalg.norm(X, 2, dim)
    else:
        X_normalized = X / np.expand_dims((X**2).sum(dim)**(1/2), axis=-1)
    return X_normalized

def np_detect_anamoly(array):
    return np.isnan(array).any() or np.isinf(array).any()

def generate_triplet_pool(cls2vecs:dict, symmetric_pairs=True, sample_pool_size=None, cls2thres=None,
                          normalize_encoding=False, normalize_diff=False,
                          record_runtime=True, print_info=True):
    """
    Generate triplets within each image class using image encodings
    Parameters:
      -- cls2vecs (dict): key: name of class (str), value: a dict mapping from idx to image encoding
      -- symmetric_pairs (bool): generate both z_(i,j) and z_(j,i). Useful only when sample is False
      -- sample_pool_size (int): Useful only when we want to sample a triplet pool
    """
    if record_runtime:
        start = time.time()

    triplet_pool = {}
    
    if sample_pool_size is None:
        for cls in tqdm(cls2vecs):
            for i, z_i in cls2vecs[cls].items():
                if normalize_encoding: 
                    z_i = vector_normalize(z_i)
                js = filter(lambda x:x!=i, cls2vecs[cls].keys()) if symmetric_pairs \
                    else filter(lambda x:x>i, cls2vecs[cls].keys())
                for j in js:
                    z_j = cls2vecs[cls][j]
                    if normalize_encoding: 
                        z_j = vector_normalize(z_j)
                    diff = z_j - z_i
                    if cls2thres is not None:
                        if np.linalg.norm(diff) < cls2thres[cls]: continue
                    if normalize_diff: 
                        diff = vector_normalize(diff)
                    triplet_pool[(cls,i,j)] = diff
    else:
        n_cls = len(cls2vecs)
        if sample_pool_size >= n_cls:
            n_pairs_cls = sample_pool_size // n_cls
            for cls, id2vecs in tqdm(cls2vecs.items()):
                m = len(id2vecs)
                perm = np.random.permutation(m*m) # [m^2] each item falls into [0, m**2-1]
                cnt = 0
                for v in perm:
                    i = (v // m).item()
                    j = (v % m).item() # 0 <= i,j <= m-1
                    if i==j:
                        continue
                    z_i = id2vecs[i]
                    z_j = id2vecs[j]
                    delta = z_j - z_i

                    if cls2thres is not None:
                        delta_norm = np.linalg.norm(delta, 2)
                        if delta_norm <= cls2thres[cls]: continue

                    if normalize_encoding:
                        z_i = vector_normalize(z_i)
                        z_j = vector_normalize(z_j)
                    if normalize_diff:
                        delta = vector_normalize(delta)
                    if np_detect_anamoly(delta): continue

                    triplet_pool[(cls,i,j)] = delta
                    cnt += 1

                    if cnt >= n_pairs_cls: break
            """ # The old version of generating image pairs on the fly, less efficient
            cls_to_samples = random.choices(list(cls2vecs.keys()), k=sample_pool_size)
            for cls in tqdm(cls_to_samples):
                KEEP_SAMPLING = True
                while KEEP_SAMPLING:
                    image_index_pairs = random.sample(list(cls2vecs[cls].keys()), k=2)
                    i, j = image_index_pairs[0], image_index_pairs[1]
                    KEEP_SAMPLING = ((cls,i,j) in triplet_pool)
                    if cls2thres is not None:
                        delta = cls2vecs[cls][i]-cls2vecs[cls][j]
                        KEEP_SAMPLING = KEEP_SAMPLING or (np.linalg.norm(delta,2).item() < cls2thres[cls])
                z_i = cls2vecs[cls][i]
                z_j = cls2vecs[cls][j]
                if normalize_encoding:
                    z_i = vector_normalize(z_i)
                    z_j = vector_normalize(z_j)
                diff = z_j - z_i
                if normalize_diff:
                    diff = vector_normalize(diff)
                triplet_pool[(cls,i,j)] = diff
            """
        else:
            sel_class = random.choices(list(cls2vecs.keys()), k=sample_pool_size)
            for class_name in sel_class:
                index_pairs = random.sample(list(cls2vecs[class_name].keys()), k=2)
                i, j = index_pairs[0], index_pairs[1]
                z_i = cls2vecs[class_name][i]
                z_j = cls2vecs[class_name][j]
                if normalize_encoding:
                    z_i = vector_normalize(z_i)
                    z_j = vector_normalize(z_j)

                diff = z_j - z_i
                if normalize_diff:
                    diff = vector_normalize(diff)
                triplet_pool[(class_name, i ,j)] = diff

    if record_runtime:
        end = time.time()
        info = "Done generating triplets. Runtime: {}s\n".format(end-start)
    else:
        info = "Done generating triplets."
    if print_info: print(info)

    return triplet_pool

def generate_class_threshold(cls2vecs:dict, outfile=None):
    print("Generate thresholds of image pair distance for each class..")
    class2thres = {}
    for cls in tqdm(cls2vecs):
        image_pair = []
        for i, z_i in cls2vecs[cls].items():
            for j in filter(lambda x:x>i, cls2vecs[cls].keys()):
                z_j = cls2vecs[cls][j]
                image_pair.append(z_j - z_i)
        image_pair = np.stack(image_pair, 0) # [N_i, 512]
        class2thres[cls] = ((image_pair**2).sum(1)**(1/2)).mean().item()
    
    if outfile is not None:
        pickle_dump(class2thres, outfile)

    return class2thres

def search_nearest_images(input_vecs:torch.Tensor, search_space:dict):
    # input_vecs [bs, 512]
    # search_space: the dict cls2vecs_flattened to search
    search_vecs = torch.from_numpy(
        np.stack(list(search_space.values()), 0)
    ).to(device) # [N, 512]
    
    # for limited memory
    if input_vecs.shape[0] > 5:
        input_vecs_chunks = input_vecs.split(split_size=5, dim=0)
        NNs_dist, NNs_ids = [], []
        for input_vecs_chunk in input_vecs_chunks:
            dist_chunk = input_vecs_chunk.to(device).unsqueeze(-1) - search_vecs.t().unsqueeze(0) # [5, 512, N]
            dist_chunk = ((dist_chunk**2).sum(1))**(1/2) # [5, N]
            NNs_dist_chunk, NNs_ids_chunk = dist_chunk.min(dim=-1)
            NNs_dist.extend(NNs_dist_chunk.tolist())
            NNs_ids.extend(NNs_ids_chunk.tolist())
    else:
        dist = input_vecs.to(device).unsqueeze(-1) - search_vecs.t().unsqueeze(0) # [bs, 512, N]
        dist = ((dist**2).sum(1))**(1/2) # [bs, N]
        NNs_dist, NNs_ids = dist.min(dim=-1) # [bs]
        NNs_dist, NNs_ids = NNs_dist.tolist(), NNs_ids.tolist()
    
    return NNs_dist, NNs_ids

def get_number_possible_pairs(train_cls2vecs:dict, asymmetric:bool=False):
    
    n_img_pairs_per_class = {}
    for class_name in train_cls2vecs:
        dict_per_class = train_cls2vecs[class_name]
        n_imgs = len(dict_per_class)
        n_img_pairs_per_class[class_name] = n_imgs * (n_imgs - 1) // 2 if asymmetric else n_imgs * (n_imgs - 1)

    total_n_pairs = sum(n_img_pairs_per_class.values())

    return total_n_pairs

def compute_dist(vecs1:torch.Tensor, vecs2:torch.Tensor, chunk:bool=True):
    # Assume vecs1 contains more vectors (N1 >> N2)
    # vecs1 [N1, vec_size]
    # vecs2 [N2, vec_size]
    print(f'vecs1 size: {list(vecs1.shape)}')
    print(f"vecs2 size: {list(vecs2.shape)}")
    n_samples = vecs1.shape[0]
    if n_samples > 50000 and chunk:
        chunk_size = 2048
        vecs1_chunks = vecs1.split(split_size=chunk_size, dim=0)
        dist = []
        for vecs1_chunk in vecs1_chunks:
            # vecs1_chunk [chunk_size, vec_size]
            diff = vecs1_chunk.unsqueeze(1) - vecs2.unsqueeze(0) # [chunk_size, N2, vec_size]
            dist_chunk = torch.norm(diff, dim=-1) # [chunk_size, N2]
            dist.append(dist_chunk)
        dist = torch.cat(dist, dim=0) # [N1, N2]
        return dist

    else:
        diff = vecs1.unsqueeze(1) - vecs2.unsqueeze(0) # [N1, N2, vec_size]
        dist = torch.norm(diff, dim=-1) # [N1, N2]
        return dist


if __name__ == '__main__':
    pass
