import os
import random
import time
import torch
import numpy as np
import shutil
from cluster import clusteringEMA_torch, clusteringEMA_v2_torch
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.pipeline import Pipeline
from PIL import Image, ImageDraw, ImageFont

from arguments import define_arguments
_, args = define_arguments()

from utils import write_config_file
from const import VISUALIZATION_CONFIG

def visualize_learned_triplets(triplet_pool:dict, Cluster, cls2files:dict, sample_top=False, sample_bottom=False, 
                        num_pairs_per_cluster=5, show_diff_cls=False, cluster_to_visualize=-1, record_config=True,
                        combine_image_groups=False, print_cls=False, record_runtime=True, print_info=True,
                        split_pairs=False, keep_orig_img=False):

    if print_info:
        print("Assinging all triplets in the sample pool to nearest clusters..")
    if record_runtime:
        start = time.time()

    outdir = "./cluster_visualization/{}/run_{}".format(args.cluster_type, args.run)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if record_config:
        config_file = os.path.join(outdir, "config.txt")
        write_config_file(args, config_file, VISUALIZATION_CONFIG)

    preprocessor = None
    if type(Cluster) in [clusteringEMA_torch, clusteringEMA_v2_torch]:
        weight = Cluster.weight.cpu() # [512, 25]
    elif type(Cluster) in [MiniBatchKMeans, KMeans]:
        weight = Cluster.cluster_centers_
        weight = torch.from_numpy(weight).t() # [512, 25]
    elif type(Cluster) == Pipeline:
        preprocessor = Cluster['preprocessor']
        weight = torch.from_numpy(Cluster['Cluster'].cluster_centers_).t() # [512 ,25]

    sample_delta_array = np.stack(list(triplet_pool.values()), 0)
    if preprocessor is not None:
        sample_delta_array = preprocessor.fit_transform(sample_delta_array)

    sample_delta_array_torch = torch.from_numpy(sample_delta_array) # [10000, 64]
    triplet_keys = list(triplet_pool.keys())
    
    from data_process import compute_dist
    dist = compute_dist(sample_delta_array_torch, weight.t()) # [n_samples, n_clusters]
    dist, cluster_indexes = dist.min(-1) # dist [n_samples], cluster_indexes [n_samples]

    if print_info and record_runtime:
        print("Done assinging all transformations in the sample pool. Runtime: {}s".format(time.time()-start))
    
    def triplet_idx_to_cls(idx:int) -> str:
        cls_name = triplet_keys[idx][0]
        return cls_name

    def diff_list_items(list_:list) -> bool:
        # check if all items are different
        diff = True
        for item in list_:
            diff = (diff and list_.count(item) == 1)
        return diff

    def print_cls_onto_image(image, text:str):
        with image.convert("RGBA") as base:
            txt = Image.new("RGBA", base.size, (255, 255, 255, 0))
            fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 15)
            d = ImageDraw.Draw(txt)
            d.text((10, 10), text, font=fnt, fill=(0, 0, 255, 255))
            out = Image.alpha_composite(base, txt).convert("RGB")
        return out

    def sample_and_show_pairs_torch(cluster_id:int):
        # cluster_id (int): the cluster to sample pairs from

        if print_info:
            print("Sample image pairs for Cluster {}".format(cluster_id))

        def show_samples_images(samples_dict):

            if split_pairs:
                cluster_img_dir = os.path.join(outdir, f'Cluster_{cluster_id}')
                os.makedirs(cluster_img_dir, exist_ok=True)
                all_samples = []
                for _, samples in samples_dict.items():
                    if samples is not None:
                        all_samples.extend(samples)
                for _, sample_index in enumerate(all_samples):
                    cls, i, j = triplet_keys[sample_index]
                    result = Image.new("RGB", (224*2, 224))
                    img_i_path, img_j_path = cls2files[cls][i], cls2files[cls][j]
                    img_i = Image.open(img_i_path).resize((224,224))
                    img_j = Image.open(img_j_path).resize((224,224))
                    if print_cls:
                        img_i = print_cls_onto_image(img_i, cls)
                    result.paste(img_i, box=(0,0))
                    result.paste(img_j, box=(224,0))
                    outfile = cluster_img_dir + f"/{cls}_{i}_{j}.png"
                    result.save(outfile)

            elif keep_orig_img:
                
                cluster_img_dir = os.path.join(outdir, f'Cluster_{cluster_id}')
                os.makedirs(cluster_img_dir, exist_ok=True)

                for group_name, samples in samples_dict.items(): # subdir: top, bottom, random
                    subdir = os.path.join(cluster_img_dir, group_name)
                    os.makedirs(cluster_img_dir, exist_ok=True)

                    for pair_idx, sample_index in enumerate(samples):
                        # pair_idx: int, sample_index: int (the order of the sample pool)
                        img_pair_dir = os.path.join(subdir, f"pair_{pair_idx + 1}")
                        os.makedirs(img_pair_dir, exist_ok=True)
                        cls, i, j = triplet_keys[sample_index]
                        img_i_path, img_j_path = cls2files[cls][i], cls2files[cls][j]
                        shutil.copy(img_i_path, os.path.join(img_pair_dir, "img1.jpeg"))
                        shutil.copy(img_j_path, os.path.join(img_pair_dir, "img2.jpeg"))
                """
                all_samples = []
                for _, samples in samples_dict.items():
                    if samples is not None:
                        all_samples.extend(samples)
                for pair_idx, sample_index in enumerate(all_samples):
                    img_pair_dir = os.path.join(cluster_img_dir, f"pair_{pair_idx+1}")
                    os.makedirs(img_pair_dir, exist_ok=True)
                    cls, i, j = triplet_keys[sample_index]
                    img_i_path, img_j_path = cls2files[cls][i], cls2files[cls][j]
                    shutil.copy(img_i_path, os.path.join(img_pair_dir, f"img1_{cls.replace('/','')}_{i}.jpeg"))
                    shutil.copy(img_j_path, os.path.join(img_pair_dir, f"img2_{cls.replace('/','')}_{j}.jpeg"))
                """
            else:
                row_gap_pixels = 1
                if combine_image_groups:
                    gap_pixels = 100
                    
                    n_groups = len([samples for samples in samples_dict.values() if samples is not None])

                    combined_result = Image.new("RGB", 
                        (224*2*n_groups+gap_pixels*(n_groups-1), 224*num_pairs_per_cluster+(num_pairs_per_cluster-1)*row_gap_pixels)
                    )
                    for group_idx, (key, samples) in enumerate(samples_dict.items()):
                        if samples is None: pass
                        else:
                            for row_id in range(num_pairs_per_cluster):
                                cls, i, j = triplet_keys[samples[row_id]]
                                img_i_path, img_j_path = cls2files[cls][i], cls2files[cls][j]
                                img_i = Image.open(img_i_path).resize((224,224))
                                img_j = Image.open(img_j_path).resize((224,224))
                                #img_i = Image.open(img_i_path).resize((224,224), Image.ANTIALIAS)
                                #img_j = Image.open(img_j_path).resize((224,224), Image.ANTIALIAS)
                                if print_cls:
                                    img_i = print_cls_onto_image(img_i, cls)
                                    img_j = print_cls_onto_image(img_j, cls)
                                combined_result.paste(img_i, box=(group_idx*(224*2+gap_pixels), row_id*(224+row_gap_pixels)))
                                combined_result.paste(img_j, box=(group_idx*(224*2+gap_pixels)+224, row_id*(224+row_gap_pixels)))

                    outfile = outdir + "/Cluster_{}.png".format(cluster_id)
                    combined_result.save(outfile)

                else:
                    for key, samples in samples_dict.items():
                        if samples is None: pass
                        else:
                            result = Image.new("RGB", (224*2, 224*num_pairs_per_cluster+(num_pairs_per_cluster-1)*row_gap_pixels))
                            for row_id in range(num_pairs_per_cluster):
                                cls, i, j = triplet_keys[samples[row_id]] # cls: name of class (str), i,j: image index(int)
                                img_i_path, img_j_path = cls2files[cls][i], cls2files[cls][j]
                                img_i = Image.open(img_i_path).resize((224,224), Image.ANTIALIAS)
                                img_j = Image.open(img_j_path).resize((224,224), Image.ANTIALIAS)
                                if print_cls:
                                    img_i = print_cls_onto_image(img_i, cls)
                                    # img_j = print_cls_onto_image(img_j, cls)
                                result.paste(img_i, box=(0, row_id*(224+row_gap_pixels)))
                                result.paste(img_j, box=(224, row_id*(224+row_gap_pixels)))

                            dest_dir = os.path.join(outdir, key)
                            if not os.path.exists(dest_dir): os.mkdir(dest_dir)
                            outfile = dest_dir + "/Cluster_{}.png".format(cluster_id)
                            result.save(outfile)

        indices = torch.where(cluster_indexes==cluster_id)[0]
        dist2centroid = dist[indices] # [n]
        dist2centroid_sorted, sort_index = torch.sort(dist2centroid, 0)
        indices_sorted = indices[sort_index] # [n]

        top_samples, bottom_samples, random_samples = None, None, None
        if len(indices) >= num_pairs_per_cluster:
            if sample_top:
                top_samples = []
                if show_diff_cls:
                    existing_cls = []
                    for triplet_index in indices_sorted.tolist():
                        image_pair_cls = triplet_idx_to_cls(triplet_index)
                        if image_pair_cls not in existing_cls:
                            existing_cls.append(image_pair_cls)
                            top_samples.append(triplet_index)
                            if len(top_samples) == num_pairs_per_cluster: break
                else:
                    top_samples = indices_sorted[:num_pairs_per_cluster].tolist()

            if sample_bottom:
                bottom_samples = []
                if show_diff_cls:
                    existing_cls = []
                    for triplet_index in indices_sorted.flip([0]).tolist():
                        image_pair_cls = triplet_idx_to_cls(triplet_index)
                        if image_pair_cls not in existing_cls:
                            existing_cls.append(image_pair_cls)
                            bottom_samples.append(triplet_index)
                            if len(bottom_samples) == num_pairs_per_cluster: break
                else:
                    bottom_samples = indices_sorted.flip([0])[:num_pairs_per_cluster].tolist()

            if show_diff_cls:
                diff_cls = False
                while not diff_cls:
                    random_samples = random.sample(indices.tolist(), k=num_pairs_per_cluster)
                    diff_cls = diff_list_items(
                        list(map(triplet_idx_to_cls, random_samples))
                    )
            else:
                random_samples = random.sample(indices.tolist(), k=num_pairs_per_cluster)
        
        else:
            print("\t-- Only {} triplets can be found in Cluster {}. Not enough.".format(len(indices), cluster_id))
            return None
        
        samples_dict = {'random':random_samples, 'top':top_samples, 'bottom':bottom_samples}
        show_samples_images(samples_dict)

    num_cluster = getattr(Cluster, "num_cluster", None) \
               or getattr(Cluster, "n_clusters", None) \
               or getattr(Cluster["Cluster"], "n_clusters", None)

    if isinstance(cluster_to_visualize, int):
        if cluster_to_visualize <= 0 or cluster_to_visualize >= num_cluster:
            cluster_to_visualize = num_cluster
        cluster_to_visualize = list(range(cluster_to_visualize))

    for cluster_id in cluster_to_visualize:
        sample_and_show_pairs_torch(cluster_id=cluster_id)

    if print_info:
        info = "Done sampling visualization for each cluster."
        if record_runtime:
            info += f" Runtime:{round(time.time()-start, 2)}s"
        print(info)
