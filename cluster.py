import os
import time
import random
import copy
import warnings
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from utils import write_config_file, pickle_dump, pickle_load, write_text
from const import CLUSTER_CONFIG
from data_process import generate_triplet_pool, generate_class_threshold, get_number_possible_pairs

from dataset_process import process_raw_dataset
from arguments import define_arguments
_, args = define_arguments()

stored_path = f"./stored_files/{args.dataset.lower()}/{args.clip_vision_model.replace('/', '')}"

class clusteringEMA(object):
    def __init__(self, num_cluster:int, num_step:int, gamma:float, batch_size:int, thres=None, **kwargs):
        self.num_cluster = num_cluster
        self.num_step = num_step
        self.gamma = gamma
        self.batch_size = batch_size
        self.thres = thres
        
    def _initialise(self, vecs_pool:list):
        self.num_vecs, self.vec_size = (len(vecs_pool), vecs_pool[0].shape)
        self.c = random.sample(vecs_pool, k=self.num_cluster) # list
        self.N = [1] * self.num_cluster
        self.m = copy.deepcopy(self.c)
        self.cluster_batch_vecs = {} # a dict to store all vectors assigned to a cluster at one minibatch
        for cluster_idx in range(self.num_cluster):
            self.cluster_batch_vecs[cluster_idx] = []
        self.steps = 0
        
    @staticmethod
    def dist(vec1:np.ndarray, vec2:np.ndarray):
        return np.linalg.norm(vec1-vec2, 2)
    @staticmethod
    def argmin(list_:list):
        return list_.index(
            min(list_)
        )
    def minibatch_update(self, vectors:list):
        # vectors: a list of vectors that are sampled from the vector pool
        for vector in vectors:
            cluster_idx = self.argmin([self.dist(vector, c) for c in self.c])
            self.cluster_batch_vecs[cluster_idx].append(vector)
        for i in range(self.num_cluster):
            self.N[i] = self.N[i] * self.gamma + len(self.cluster_batch_vecs[i]) * (1-self.gamma)
            self.m[i] = self.m[i] * self.gamma + sum(self.cluster_batch_vecs[i]) * (1-self.gamma)
            self.c[i] = self.m[i] / self.N[i]
            self.cluster_batch_vecs[i] = []
    
    def _check_stopping_criterion(self, last_c):
        delta = np.stack(self.c, axis=0) - np.stack(last_c, axis=0) # [num_cluster, vec_size]
        vecs_norm = np.sum(np.abs(delta)**2, axis=-1)**(1./2)
        sum_change = np.sum(vecs_norm).item()
        return (sum_change < self.thres)
    
    def __call__(self, delta_array:list):
        self._initialise(delta_array)
        for i in tqdm(range(self.num_step)):
            batch_vectors = random.sample(delta_array, self.batch_size)
            curr_centroids = self.c
            self.minibatch_update(batch_vectors)
            self.steps += 1
            if self.thres is not None and self._check_stopping_criterion(curr_centroids): 
                break
        #print("Online EMA clustering done.")

class clusteringEMA_torch(nn.Module):
    def __init__(self, batch_size, num_cluster, num_step, 
                 gamma=0.99, eps=1e-5, thres=None, device=torch.device("cuda:0"), data=None,):
        super(clusteringEMA_torch, self).__init__()
        self.batch_size = batch_size
        self.num_cluster = num_cluster
        self.eps = eps
        self.num_step = num_step
        self.gamma = gamma
        self.thres = thres
        self.device = device
        if data is not None:
            self.initialize_centroids(data)
    
    def initialize_centroids(self, data):
        # data: list of np.ndarray
        self.data = data
        embed = torch.from_numpy(
            np.stack(random.sample(self.data, self.num_cluster), 0)
        ).t().to(self.device)
        self.register_buffer('weight', embed)
        self.register_buffer('cluster_size', torch.ones(self.num_cluster).to(self.device))
        self.register_buffer('embed_avg', embed.clone())
    
    def find_nearest_centroids(self, x):
        """
        Input:
        x - (batch_size, vec_size)
        ---------
        Output:
        result - (batch_size, vec_size): the centroids for vectors in the minibatch
        argmin - (batch_size): the indexes of centroids for vectors in the minibatch
        """
        x_expanded = x.unsqueeze(-1)
        emb_expanded = self.weight
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        result = self.weight.t().index_select(0, argmin.view(-1))
        return result, argmin
    
    def minibatch_update(self, batch_vectors):
        """
        Input: batch_vectors - (batch_size, vec_size)
        """
        _, argmin = self.find_nearest_centroids(batch_vectors)
        latent_indices = torch.arange(self.num_cluster).type_as(argmin)
        emb_onehot = (argmin.view(-1, 1)
                      == latent_indices.view(1, -1)).type_as(batch_vectors.data) # [batch_size, num_cluster]
        n_idx_choice = emb_onehot.sum(0) # [num_cluster]
        n_idx_choice[n_idx_choice==0] = 1
        flatten = batch_vectors.permute(1,0) # [vec_size, batch_size]
        self.cluster_size.data.mul_(self.gamma).add_(
            1-self.gamma, n_idx_choice
        )
        embed_sum = flatten @ emb_onehot # [vec_size, num_cluster]
        self.embed_avg.data.mul_(self.gamma).add_(
            1 - self.gamma, embed_sum
        )
        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.eps) / (n + self.num_cluster * self.eps) * n
        )
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
        self.weight.data.copy_(embed_normalized)

    def _check_stopping_criterion(self, last_c):
        delta = self.weight - last_c # [vec_size, num_cluster]
        #sum_change = torch.norm(delta, 2, 0).sum().item()
        sum_change = (delta**2).sum(0)**(1/2).sum(0).item()
        return (sum_change < self.thres)
    
    def forward(self,):
        """
        input_vectors - a list of ndarray vectors, with the length of ~80 million
        """
        with torch.no_grad():
            for i in tqdm(range(self.num_step)):
                batch_vectors = random.sample(self.data, self.batch_size)
                last_c = self.weight # [vec_size, num_cluster]
                self.minibatch_update(
                    torch.from_numpy(
                        np.stack(batch_vectors, 0)
                    ).to(self.device)
                )
                if self.thres is not None and self._check_stopping_criterion(last_c):
                    break
        #print("Online EMA clustering done.")

class clusteringEMA_v2_torch(nn.Module):
    def __init__(self, batch_size, num_cluster, max_iter, 
                 gamma=0.99, eps=1e-5, thres=None, entropy_thres_low:float=None, entropy_thres_up:float=None,
                 cardinality_thres:float=None, check_interval:int=None, device=torch.device("cuda:0")):
        super(clusteringEMA_v2_torch, self).__init__()
        self.batch_size = batch_size
        self.num_cluster = num_cluster
        self.eps = eps
        self.max_iter = max_iter
        self.gamma = gamma
        self.thres = thres
        self.entropy_thres_low = entropy_thres_low
        self.entropy_thres_up = entropy_thres_up
        self.cardinality_thres = cardinality_thres
        self.check_interval = check_interval if check_interval is not None else 50
        self.device = device
    
    def initialize_centroids(self, cls2vecs:dict):
        # data: list of np.ndarray
        print("Initialize cluster centroids.")
        self.class_names = list(cls2vecs.keys())
        self.n_class = len(self.class_names)
        init_cluster_class = random.choices(self.class_names, k=self.num_cluster)
        init_cluster_class_id = [self.class_names.index(class_name) for class_name in init_cluster_class]
        embed = []
        for class_name in init_cluster_class:
            i, j = random.sample(list(range(len(cls2vecs[class_name]))), k=2)
            embed.append(cls2vecs[class_name][i] - cls2vecs[class_name][j])
        embed = torch.from_numpy(
            np.stack(embed, 0)
        ).t().to(self.device)
        hist = torch.zeros((self.num_cluster, len(self.class_names))).to(self.device)
        for q_id, class_id in zip(range(self.num_cluster), init_cluster_class_id):
            hist[q_id][class_id] = 1

        self.register_buffer('weight', embed)
        self.register_buffer('cluster_size', torch.ones(self.num_cluster).to(self.device))
        self.register_buffer('embed_avg', embed.clone())
        self.register_buffer('hist', hist)
    
    def find_nearest_centroids(self, x):
        """
        Input:
        x - (batch_size, vec_size)
        ---------
        Output:
        result - (batch_size, vec_size): the centroids for vectors in the minibatch
        argmin - (batch_size): the indexes of centroids for vectors in the minibatch
        """
        x_expanded = x.unsqueeze(-1)
        emb_expanded = self.weight
        #dist = x_expanded - emb_expanded # [bs, vec_size, n_clusters]
        dist = (((x_expanded - emb_expanded)**2).sum(1))**(0.5) # [bs, n_lusters]
        #dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        result = self.weight.t().index_select(0, argmin.view(-1))
        return result, argmin

    def process_batch_pool(self, batch_pool:dict):
        # batch_pool {(cls, i, j): np.ndarray}
        batch_keys = list(map(lambda key:key[0], batch_pool.keys()))
        batch_keys_id = np.array(list(map(lambda class_name:self.class_names.index(class_name), batch_keys))) # [bs]
        batch_vectors = np.stack(list(batch_pool.values()), 0) # [bs, 512]
        return batch_keys_id, batch_vectors

    def update_hist(self, argmin, batch_keys_id):
        # argmin [bs]
        # batch_keys_id [bs]
        batch_hist = torch.zeros((self.num_cluster, self.n_class)).to(self.device) # [n_cluster, n_class]
        for q_idx in range(self.num_cluster):
            sel_idx = torch.where(argmin==q_idx)[0] # [n]
            sel_cls = batch_keys_id[sel_idx] # [n]
            batch_hist[q_idx] = torch.bincount(sel_cls, minlength=self.n_class).type_as(batch_hist)
        self.hist.data.mul_(self.gamma).add_(
            1-self.gamma, batch_hist
        )

    def minibatch_update(self, batch_vectors, batch_keys_id):
        """
        Input: 
        -- batch_vectors (torch.Tensor) [batch_size, vec_size]
        -- batch_keys_id (torch.Tensor) [batch_size] 
        """
        _, argmin = self.find_nearest_centroids(batch_vectors)
        latent_indices = torch.arange(self.num_cluster).type_as(argmin)
        emb_onehot = (argmin.view(-1, 1)
                      == latent_indices.view(1, -1)).type_as(batch_vectors.data) # [batch_size, num_cluster]
        n_idx_choice = emb_onehot.sum(0) # [num_cluster]
        n_idx_choice[n_idx_choice==0] = 1
        flatten = batch_vectors.permute(1,0) # [vec_size, batch_size]
        self.cluster_size.data.mul_(self.gamma).add_(
            1-self.gamma, n_idx_choice
        )
        embed_sum = flatten @ emb_onehot # [vec_size, num_cluster]
        self.embed_avg.data.mul_(self.gamma).add_(
            1 - self.gamma, embed_sum
        )
        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.eps) / (n + self.num_cluster * self.eps) * n
        )
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
        self.weight.data.copy_(embed_normalized)

        self.update_hist(argmin, batch_keys_id)

    def _reassign_centroids(self, q_idx, train_cls2vecs):
        # q_idx [n] indexes of clusters that should be reinitialised
        n_reassign = len(q_idx)
        reassign_pool = generate_triplet_pool(
            train_cls2vecs, sample_pool_size=n_reassign, cls2thres=None, 
            record_runtime=False, print_info=False
        )
        vecs = torch.from_numpy(
            np.stack(list(reassign_pool.values()), 0)
        ).t().to(self.device)
        keys = list(
            map(lambda x:self.class_names.index(x), [key[0] for key in reassign_pool.keys()])
        )
        self.weight.data[:, list(q_idx)] = vecs
        self.cluster_size.data[list(q_idx)] = 1
        self.embed_avg.data[:, list(q_idx)] = vecs.clone()
        
        self.hist.data[list(q_idx)] = torch.zeros(n_reassign, self.n_class).to(self.device)
        for q, key in zip(q_idx, keys):
            self.hist[q, key] = 1
        
        print(f"{len(q_idx)} clusters have been reinitialized:\n{q_idx}")

    def _check_clusters(self,):
        hist_normalized = self.hist / self.cluster_size.unsqueeze(1) # [n_cluster, n_class]
        hist_normalized[hist_normalized==0.0] = 1
        entropy = (-1 * hist_normalized * hist_normalized.log()).sum(1) # [n_cluster]
        print(f"Entropy of {len(entropy)} clusters:\n{entropy.tolist()}")

        if self.entropy_thres_low is not None:
            q_entropy_low_idx = torch.where(entropy < self.entropy_thres_low)[0]
        else:
            q_entropy_low_idx = torch.tensor([])

        if self.entropy_thres_up is not None:
            q_entropy_up_idx = torch.where(entropy > self.entropy_thres_up)[0]
        else:
            q_entropy_up_idx = torch.tensor([])
        
        if self.cardinality_thres is not None:
            q_cardinality_idx = torch.where(self.cluster_size < self.cardinality_thres)[0]
        else:
            q_cardinality_idx = torch.tensor([])
        
        q_reassign_idx = set(q_entropy_low_idx.tolist() + q_entropy_up_idx.tolist() + q_cardinality_idx.tolist())
        return q_reassign_idx

    def _sum_change(self, last_weight:torch.Tensor):
        # last_weight [vec_size, n_clusters]
        change = self.weight - last_weight # [vec_size, n_clusters]
        sum_change = (((change**2).sum(0))**(0.5)).sum(0).item()
        return sum_change

    def _check_stopping_criterion(self, last_c):
        delta = self.weight - last_c # [vec_size, num_cluster]
        #sum_change = torch.norm(delta, 2, 0).sum().item()
        sum_change = (delta**2).sum(0)**(1/2).sum(0).item()
        return (sum_change < self.thres)

    def forward(self, train_cls2vecs:dict, outfile:str, change_log:str=None):

        if not hasattr(self, 'weight'):
            self.initialize_centroids(train_cls2vecs)

        n_img_pairs = get_number_possible_pairs(train_cls2vecs)
        n_iter_steps = self.max_iter * n_img_pairs // self.batch_size

        with torch.no_grad():

            for i in tqdm(range(1, n_iter_steps+1)):
                batch_pool = generate_triplet_pool(
                    train_cls2vecs, sample_pool_size=self.batch_size, cls2thres=None,  
                    record_runtime=False, print_info=False
                )
                batch_keys_id, batch_vectors = self.process_batch_pool(batch_pool) # [bs] [bs, 512]
                del batch_pool

                last_weight = self.weight.clone() # [vec_size, n_clusters]
                self.minibatch_update(
                    batch_vectors = torch.from_numpy(batch_vectors).to(self.device),
                    batch_keys_id = torch.from_numpy(batch_keys_id).to(self.device),
                )
                sum_change = self._sum_change(last_weight)
                if change_log is not None:
                    write_text(f"Iteration {i}: {sum_change}\n", change_log, 'a')

                if i % 5 == 0 or i == n_iter_steps:
                    pickle_dump(self, outfile)

                if i % self.check_interval == 0:
                    print(f"Check clusters at Epoch {i}")
                    q_reassign_idx = self._check_clusters()
                    if len(q_reassign_idx) > 0:
                        self._reassign_centroids(q_reassign_idx, train_cls2vecs)

                last_c = self.weight # [vec_size, num_cluster]
                if self.thres is not None and self._check_stopping_criterion(last_c):
                    break
        #print("Online EMA clustering done.")

def sample_batch_on_the_fly(cls2vecs:dict, batch_size:int, cls2thres=None,
                            normalize_encoding=False, normalize_diff=False, return_tensor=False):
                            
    batch_triplet_pool = generate_triplet_pool(
                            cls2vecs=cls2vecs, 
                            sample_pool_size=batch_size, 
                            cls2thres=cls2thres,
                            normalize_encoding=normalize_encoding, 
                            normalize_diff=normalize_diff,
                            record_runtime=False, 
                            print_info=False
                        )

    batch_triplets = list(batch_triplet_pool.values())
    if return_tensor:
        return torch.from_numpy(
            np.stack(batch_triplets, 0)
        )
    else:
        return batch_triplets

def run_cluster_algorithm(Cluster, cls2vecs:dict, on_the_fly=True, symmetric_pairs=False, prefilter_image_pairs=False,
                          normalize_encoding=False, normalize_diff=False, outdir=None, record_centroid_change=False,
                          record_runtime=False, print_info=True):
    """
    Run the clustering algorithm
    Parameters:
       -- Cluster: a cluster algorithm object, can be an object of MiniBatchKmeans, KMeans or EMA
       -- cls2vecs (dict): key: name of class (str), value: a dict mapping from idx to image encoding]
       -- on_the_fly (bool): Generate triplets on the fly when doing clustering; triplets are computed
                             beforehand if set to False
       -- symmetric_pairs (bool): whether to generate symmetric triplets z_(i,j) and z_(j,i). Useful
                                  only when on_the_fly is False
    """
    if outdir is not None:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outfile = os.path.join(outdir, "Cluster.pkl")
        config_file = os.path.join(outdir, "config.txt")
        if not os.path.exists(config_file):
            write_config_file(args, config_file, CLUSTER_CONFIG)

        if record_centroid_change:
            centroid_change_logfile = os.path.join(outdir, "centroid_change.txt")

    cls2thres = None
    if prefilter_image_pairs:
        thres_file = os.path.join(stored_path, "cls2thres.pkl")
        if not os.path.exists(thres_file):
            cls2thres = generate_class_threshold(cls2vecs, thres_file)
        else:
            cls2thres = pickle_load(thres_file)

    if not on_the_fly:
        if type(Cluster) in [KMeans, Pipeline]:
            triplet_pool = generate_triplet_pool(cls2vecs=cls2vecs, sample_pool_size=args.n_samples_kmeans,
                            cls2thres=cls2thres, normalize_encoding=normalize_encoding, normalize_diff=normalize_diff,
                            record_runtime=record_runtime, print_info=print_info)
        else:
            triplet_pool = generate_triplet_pool(cls2vecs=cls2vecs, symmetric_pairs=symmetric_pairs, 
                            cls2thres=cls2thres, normalize_encoding=normalize_encoding, normalize_diff=normalize_diff,
                            record_runtime=record_runtime, print_info=print_info)
        delta_array = list(triplet_pool.values())
        if print_info:
            print("Number of triplets:", len(delta_array))
        if type(Cluster)==clusteringEMA_torch:
            Cluster.initialize_centroids(delta_array)
            Cluster()
        elif type(Cluster) in [MiniBatchKMeans, KMeans, Pipeline]:
            Cluster.fit(np.stack(delta_array, 0))

        pickle_dump(Cluster, outfile)

    else: # Gernerate triplets on the fly
        if type(Cluster)==clusteringEMA_torch:
            with torch.no_grad():
                for i in tqdm(range(Cluster.num_step)):
                    if i == 0:
                        batch_vectors = sample_batch_on_the_fly(
                                cls2vecs, 
                                Cluster.batch_size,
                                cls2thres=cls2thres, 
                                normalize_encoding=normalize_encoding, 
                                normalize_diff=normalize_diff
                                )
                        Cluster.initialize_centroids(batch_vectors)
                        batch_vectors = torch.from_numpy(
                            np.stack(batch_vectors, 0)
                        )
                    else:
                        batch_vectors = sample_batch_on_the_fly(
                                cls2vecs, 
                                Cluster.batch_size, 
                                cls2thres=cls2thres, 
                                normalize_encoding=normalize_encoding, 
                                normalize_diff=normalize_diff, 
                                return_tensor=True
                                )
                    last_c = Cluster.weight # [vec_size, num_cluster]
                    Cluster.minibatch_update(
                        batch_vectors.to(Cluster.device)
                    )
                    if Cluster.thres is not None and Cluster._check_stopping_criterion(last_c):
                        pickle_dump(Cluster, outfile)
                        break

                    if i%5==0 or i==Cluster.num_step-1:
                        pickle_dump(Cluster, outfile)

        elif type(Cluster) == clusteringEMA_v2_torch:
            Cluster(cls2vecs, outfile, centroid_change_logfile)

        else: # MiniBatchKMeans
            preprocessor = None
            if type(Cluster) == Pipeline:
                preprocessor = Cluster['preprocessor']
                Cluster = Cluster['Cluster']

            n_possible_pairs = get_number_possible_pairs(cls2vecs, False)
            n_steps = (Cluster.max_iter * n_possible_pairs * 2) // Cluster.batch_size
            start_time = time.time()
            for i in tqdm(range(n_steps)):
                batch_vectors = sample_batch_on_the_fly(
                                cls2vecs, 
                                Cluster.batch_size,
                                cls2thres=cls2thres,
                                normalize_encoding=normalize_encoding, 
                                normalize_diff=normalize_diff
                                )
                batch_vectors = np.stack(batch_vectors, 0)

                if preprocessor is not None:
                    batch_vectors = preprocessor.fit_transform(batch_vectors)

                last_centroid = Cluster.cluster_centers_.copy() if hasattr(Cluster, "cluster_centers_") else None
                Cluster.partial_fit(batch_vectors)

                if record_centroid_change and last_centroid is not None:
                    delta_centroid = Cluster.cluster_centers_ - last_centroid # [25, 512]
                    change = ((delta_centroid**2).sum(1)**(1/2)).sum().item()

                    with open(centroid_change_logfile, "a") as f:
                        st = "update step {}: sum of centroid changes: {}\n".format(i, round(change, 4))
                        f.write(st)

                if i%5==0 or i==n_steps-1:
                    if preprocessor is not None:
                        pickle_dump(Pipeline(
                            [
                                ('preprocessor', PCA(n_components=64, random_state=42)),
                                ('Cluster', Cluster)
                            ]
                        ), outfile)
                    else:
                        pickle_dump(Cluster, outfile)

                if i == 99:
                    time_pass = time.time() - start_time
                    print(f"{time_pass} seconds has passed. Avg. Time per iter: {time_pass/100}s")
    return Cluster


def rescale_cluster_centroids(cluster_dir:str=None, img_pairs_size:int=131072):

    if cluster_dir is None:
        cluster_dir = "./learned_clusters/{}/cluster_run_{}".format(args.cluster_type, args.run)

    Cluster = pickle_load(os.path.join(cluster_dir, "Cluster.pkl"))
    if type(Cluster) in [clusteringEMA_torch, clusteringEMA_v2_torch]:
        n_clusters = Cluster.num_cluster
    else:
        n_clusters = Cluster.n_clusters

    _, _, _, _, train_cls2vecs, _ = process_raw_dataset(
        dataset=args.dataset.lower(),
        dataset_path=os.path.join(args.dataset_loc, args.dataset.lower()),
        split="train"
    )

    cls2thres = None
    if args.prefilter_image_pairs:
        thres_file = os.path.join(stored_path, "cls2thres.pkl")
        if os.path.exists(thres_file):
            cls2thres = pickle_load(thres_file)
        else:
            cls2thres = generate_class_threshold(train_cls2vecs, outfile=thres_file)

    sample_pool = generate_triplet_pool(
        train_cls2vecs, sample_pool_size=img_pairs_size, cls2thres=cls2thres
    )
    delta_data = np.stack(list(sample_pool.values()), 0) # [n_samples, 512]
    
    if type(Cluster) in [KMeans, MiniBatchKMeans]:
        labels = Cluster.predict(delta_data) # [n_samples]
        rescaled_weight = Cluster.cluster_centers_.copy()

    elif type(Cluster) in [clusteringEMA_torch, clusteringEMA_v2_torch]:
        delta_data = torch.from_numpy(delta_data).float()
        weight = Cluster.weight.float().cpu().clone() # [512, n_clusters]
        dist = (((delta_data.unsqueeze(-1) - weight.unsqueeze(0))**2).sum(1))**(0.5) # [n_samples, n_clusters]
        labels = dist.argmin(-1) # [n_samples]
        rescaled_weight = weight.t() # [n_clusters, 512]

    for cluster_idx in range(n_clusters):
        weight_ = rescaled_weight[cluster_idx] # [512]
        try:
            sample_indices = np.where(labels==cluster_idx)[0].tolist() # [n]
        except:
            sample_indices = torch.where(labels==cluster_idx)[0].tolist() # [n]

        if len(sample_indices) == 0:
            warning = f"Cluster {cluster_idx} does not have member vectors. Keep original norm."
            warnings.warn(warning)
            continue
        
        delta_ = delta_data[sample_indices] # [n, 512]
        avg_norm = (((delta_**2).sum(1))**(1/2)).mean()
        weight_re = weight_ / ((weight_**2).sum())**(0.5) * avg_norm
        rescaled_weight[cluster_idx] = weight_re
    
    rescaled_weight_path = os.path.join(cluster_dir, "rescaled_weight.pkl")
    pickle_dump(rescaled_weight, rescaled_weight_path)

    return rescaled_weight

def take_cluster_weight(cluster_dir:str, init_scale:bool, return_tensor:bool=True):
    """
    cluster_dir: (str) the path of the directory where the target cluster is located
    init_scale: (bool) rescale the norm of centroids to the average norm of member vectors
    return_tensor: (bool) return cluster weights as torch.Tensor
    """
    assert os.path.exists(cluster_dir)
    Cluster = pickle_load(os.path.join(cluster_dir, "Cluster.pkl"))
    
    if init_scale:
        # scale each cluster center to the average norm of its members
        rescale_path = os.path.join(cluster_dir, "rescaled_weight.pkl")
        if os.path.exists(rescale_path):
            print("Load existing rescaled weight.")
            weight = pickle_load(rescale_path)
        else:
            weight = rescale_cluster_centroids(cluster_dir)
    else:
        # use the cluster centres as they are as without rescaling
        weight = Cluster.cluster_centers_.copy()
    
    if return_tensor and type(weight)==np.ndarray:
        weight = torch.from_numpy(weight).float()

    return weight

def get_cluster_entropy(Cluster, entropy_log:str=None, verbose:bool=True):

    if type(Cluster) in [clusteringEMA_v2_torch]:
        n_clusters = Cluster.num_cluster
        cluster_size = Cluster.cluster_size.cpu().data # [n_cluster]

        hist = Cluster.hist.cpu().data # [n_cluster, n_class]
        hist_normalized = hist / cluster_size.unsqueeze(-1) # [n_cluster, n_class]
        hist_normalized_nonzero = hist_normalized.clone()
        hist_normalized_nonzero[hist_normalized_nonzero==0.0] = 1
        entropy = (-1 * hist_normalized * hist_normalized_nonzero.log()).sum(1) # [n_cluster]

        if verbose:
            for idx, ent in enumerate(entropy.tolist()):
                print(f"Cluster {idx} entropy: {ent}")

        if entropy_log is not None:
            text = "Entropy Report:\n\n"
            for idx, (ent, size) in enumerate(zip(entropy.tolist(), cluster_size.tolist())):
                text += f"Cluster {idx} entropy: {round(ent, 4)}, cardinality: {round(size, 4)}\n"
            write_text(text, entropy_log)
    return entropy


def process_clusters(Cluster, entropy_min_thres:float=0.0, entropy_max_thres:float=10.0, retain_lower:bool=True):
    from copy import deepcopy
    Cluster_new = deepcopy(Cluster)

    if isinstance(Cluster, clusteringEMA_v2_torch):
        entropy = get_cluster_entropy(Cluster, verbose=False) # [n_clusters]
        retain_cluster_idx = torch.where(
            torch.logical_and(entropy >= entropy_min_thres, entropy <= entropy_max_thres)
        )[0].to(Cluster.weight.device)

        new_weight = Cluster.weight.index_select(dim=1, index=retain_cluster_idx) # [512, n_clusters_]
        new_hist = Cluster.hist.index_select(dim=0, index=retain_cluster_idx) # [n_clusters_, n_cls]
        new_cluster_size = Cluster.cluster_size.index_select(dim=0, index=retain_cluster_idx)
        new_embed_avg = Cluster.embed_avg.index_select(dim=0, index=retain_cluster_idx)

        Cluster_new.weight = new_weight
        Cluster_new.hist = new_hist
        Cluster_new.cluster_size = new_cluster_size 
        Cluster_new.embed_avg = new_embed_avg
        Cluster_new.num_cluster = new_weight.size(1)

        flag = 'lower' if retain_lower else 'higher'
        print(f"{Cluster_new.num_cluster} clusters with Entropy {flag} bwtween {entropy_min_thres} and {entropy_max_thres} retained.")

        return Cluster_new

def get_cluster_number(Cluster):
    if isinstance(Cluster, (MiniBatchKMeans, KMeans)):
        return Cluster.n_clusters
    else:
        return Cluster.num_cluster

def get_class_counter(Cluster, image_pairs_pool:dict, write_file:str=None):
    n_clusters = get_cluster_number(Cluster)

    image_pairs_class = [image_pair_key[0] for image_pair_key in image_pairs_pool.keys()]
    image_pairs_diff = np.stack(list(image_pairs_pool.values()), 0) # [pool_size, 512]
    if isinstance(Cluster, MiniBatchKMeans):
        image_pairs_labels = Cluster.predict(image_pairs_diff) # [pool_size]
    
    from collections import Counter
    show_n_class = 50
    for cluster in tqdm(range(n_clusters)):
        indexes = np.where(image_pairs_labels==cluster)[0] # [n]
        class_labels = [image_pairs_class[ii] for ii in list(indexes)]
        class_counter = Counter(class_labels)

        if write_file is not None:
            appear_text = f"{len(class_counter)} classes found in cluster [{cluster}] (Top {show_n_class} displayed):\n"
            
            for idx, (class_name, counts) in enumerate(class_counter.most_common(show_n_class)):
                appear_text += f"\t- {class_name:<35} {counts:>12}\n"
            appear_text += "-"*55
            appear_text += "\n"

            n_assign = sum(class_counter.values())
            ratio_assign = n_assign / len(image_pairs_pool)

            appear_text += f"Total: {n_assign:>47} ({(ratio_assign*100):.2f}%)\n" 

            appear_text += "\n"
            write_text(appear_text, write_file, 'a')

    return

if __name__ == '__main__':
    pass
