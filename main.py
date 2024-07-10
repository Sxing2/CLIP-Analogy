import os
from tqdm import tqdm
import clip
import torch.nn as nn

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from cluster import clusteringEMA_torch, clusteringEMA_v2_torch, run_cluster_algorithm

from arguments import define_arguments
from visualize import visualize_learned_triplets
from data_process import generate_triplet_pool, generate_class_threshold, convert_flattened_cls2vecs
from dataset_process import process_raw_dataset
from utils import *

# Set seeds
import torch
import numpy as np
import random

parser, args = define_arguments()

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# hyperperameters and experiment settings
device = torch.device(args.device)
vec_size = model2vecsize[args.clip_vision_model]

stored_path = f"./stored_files/{args.clip_vision_model.replace('/', '')}/{args.dataset.lower()}"
if not os.path.exists(stored_path):
    os.makedirs(stored_path)

clustering_methods = {'EMA': clusteringEMA_torch, 'minibatch_kmeans': MiniBatchKMeans, 
                      'kmeans': KMeans}

# data path
dataset = args.dataset.lower()
if dataset in ['imagenet-21k']:
    dataset_path = '/nas/imagenet-21k'
else:
    dataset_path = os.path.join(args.dataset_loc, dataset)
print("current path:", os.getcwd())
print("Dataset:", dataset)
print("dataset path:", dataset_path)

_, _, categories, train_cls2files, train_cls2vecs, train_flattened_cls2vecs = process_raw_dataset(
    dataset, dataset_path, "train"
)

#print("All classes:", categories)
categories = list(set(categories))
print("number of classes:", len(categories))
print("\n")

# Step One: Learn clusters
cluster_dir = f"./learned_clusters/{args.clip_vision_model.replace('/', '')}/{args.cluster_type}/cluster_run_{args.run}"
if args.train_cluster:
    if not args.cluster_from_checkpoint or not os.path.exists(cluster_dir):
        print("Train clusters from scratch.")
        if args.cluster_type == 'EMA':
            clustering_kwargs = {
                'batch_size': args.minibatch_size,
                'num_cluster': args.num_cluster,
                'num_step': args.max_step,
                'gamma': args.gamma,
                'eps': 1e-5,
                'thres': args.thres,
                'device': device,
            }
            Cluster = clusteringEMA_torch(**clustering_kwargs).to(device)

        elif args.cluster_type == 'EMA_v2':
            clustering_kwargs = {
                'batch_size': args.minibatch_size,
                'num_cluster': args.num_cluster,
                'max_iter': args.max_iter,
                'gamma': args.gamma,
                'eps': 1e-5,
                'thres': args.thres,
                'entropy_thres_low': args.entropy_thres_low,
                'entropy_thres_up': args.entropy_thres_up,
                'cardinality_thres': args.cardinality_thres,
                'check_interval': args.check_interval,
                'device': device,
            }
            Cluster = clusteringEMA_v2_torch(**clustering_kwargs).to(device)
        
        elif args.cluster_type == 'minibatch_kmeans':
            clustering_kwargs ={
                'n_clusters': args.num_cluster,
                'max_iter': args.max_iter,
                'batch_size': args.minibatch_size,
                'reassignment_ratio': args.reassignment_ratio if args.reassignment_ratio is not None else 0.01,
                'compute_labels': False,
                'tol': args.thres if args.thres is not None else 0.0,
            }
            Cluster = MiniBatchKMeans(**clustering_kwargs)
            if args.reduced_dim is not None:
                Cluster = Pipeline(
                    [
                        ('preprocessor', PCA(n_components=args.reduced_dim, random_state=42)),
                        ('Cluster', Cluster)
                    ]
                )

        elif args.cluster_type == "kmeans":
            clustering_kwargs = {
                'n_clusters': args.num_cluster,
                'max_iter': args.max_iter,
            }
            Cluster = KMeans(**clustering_kwargs)
            if args.reduced_dim is not None:
                Cluster = Pipeline(
                    [
                        ('preprocessor', PCA(n_components=args.reduced_dim, random_state=42)),
                        ('Cluster', Cluster)
                    ]
                )
    else:
        print("Train clusters from checkpoints.")
        outfile = os.path.join(cluster_dir, "Cluster.pkl")
        Cluster = pickle_load(outfile)
        if type(Cluster) == Pipeline:
            setattr(Cluster['Cluster'], 'max_iter', args.max_iter)
        else:
            setattr(Cluster, 'max_iter', args.max_iter)

    Cluster = run_cluster_algorithm(
        Cluster = Cluster, 
        cls2vecs = train_cls2vecs, 
        on_the_fly = (args.on_the_fly==1), 
        symmetric_pairs = (args.symmetric_pairs==1), 
        prefilter_image_pairs = (args.prefilter_image_pairs==1),
        normalize_encoding = (args.normalize_encoding==1),
        normalize_diff = (args.normalize_diff==1),
        outdir = cluster_dir,
        record_centroid_change=True,
        record_runtime=True, 
        print_info=True
    )
else:
    print("Load trained clusters without further training.")
    outfile = os.path.join(cluster_dir, "Cluster.pkl")
    Cluster = pickle_load(outfile)

if isinstance(Cluster, nn.Module):
    n_clusters = Cluster.num_cluster
else:
    n_clusters = Cluster.n_clusters

# (optional) Visualize the learned clusters
if args.show_images:
    cls2thres = None
    if args.prefilter_image_pairs:
        thres_file = os.path.join(stored_path, "cls2thres.pkl")
        if os.path.exists(thres_file):
            cls2thres = pickle_load(thres_file)
        else:
            cls2thres = generate_class_threshold(train_cls2vecs, outfile=thres_file)

    sample_triplet_pool = generate_triplet_pool(train_cls2vecs, sample_pool_size=131072, cls2thres=cls2thres,
                                normalize_encoding=(args.normalize_encoding==1),
                                normalize_diff=(args.normalize_diff==1),
                                record_runtime=True, print_info=True)

    visualize_learned_triplets(sample_triplet_pool, Cluster, train_cls2files,
                                sample_top=(args.sample_top==1),
                                sample_bottom=(args.sample_bottom==1),
                                num_pairs_per_cluster=args.num_images_shown,
                                show_diff_cls=(args.show_diff_cls==1),
                                cluster_to_visualize=-1,
                                record_config=True,
                                combine_image_groups=True,
                                print_cls=True,
                                record_runtime=True, 
                                print_info=True,
                                split_pairs=False,
                                keep_orig_img=False,
                            )
