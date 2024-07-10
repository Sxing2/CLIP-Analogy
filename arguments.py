import argparse

def define_arguments():
    parser = argparse.ArgumentParser()

    # Model setting
    parser.add_argument('--device', type=str, choices=['cuda', 'cuda:0', 'cuda:1', 'cpu'], default='cuda',
                       help='Device to run the process [cuda]')
    parser.add_argument('--clip_vision_model', type=str, 
                        choices=['ViT-L/14', 'ViT-B/16', 'ViT-B/32', 'RN50'], default="ViT-B/32")
    parser.add_argument('--seed', type=int, default=0)

    # path setting
    parser.add_argument('--dataset_loc', type=str, default='./datasets')
    parser.add_argument('--dataset', type=str, default="ImageNet-100", 
                       help='Dataset to use [ImageNet-100]')

    # Hyperparameters for clustering
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--train_cluster', action="store_true",
                       help='whether to train a cluster or use a learned object w/o training.')
    parser.add_argument('--cluster_from_checkpoint', action="store_true",
                       help="Continue training clusters from checkpoints.")
    parser.add_argument('--cluster_type', type=str, choices=['EMA', 'EMA_v2', 'minibatch_kmeans', 'kmeans'], default='minibatch_kmeans',
                       help='Clustering algorithm to cluster difference vectors [EMA]')
    parser.add_argument('--num_cluster', type=int, default=25,
                       help='Number of clusters [25]')
    parser.add_argument('--prefilter_image_pairs', type=int, choices=[0,1], default=0,
                       help='whether to use L2 norm to prefilter image pairs before clustering [0]')
    parser.add_argument('--on_the_fly', type=int, default=1,
                       help='whether to generate triplet vectors on-the-fly when doing clustering [0].')
    parser.add_argument('--symmetric_pairs', type=int, default=0,
                       help='whether to generate both z_(i,j) and z_(j,i) as triplets.')
    parser.add_argument('--n_samples_kmeans', type=int, default=None, 
                       help='Number of triplet samples for K-Means clustering [None]')
    parser.add_argument('--reduced_dim', type=int, default=None, 
                       help='Use PCA to reduce dimension before clustering [None]')
    parser.add_argument('--normalize_encoding', type=int, choices=[0,1], default=0, 
                       help='Whether to normalize image encodings [0]')
    parser.add_argument('--normalize_diff', type=int, choices=[0,1], default=0,
                       help='Whether to normalize image difference vector [0]')
    parser.add_argument('--reassignment_ratio', type=float, default=None,
                       help='Reassignment ratio for preventing sparse clusters (for minibatch kmeans only) [None]')
    parser.add_argument('--max_iter', type=int, default=None,
                       help='Max iterations over triplets (for Minibatch-KMeans) [2]')
    parser.add_argument('--minibatch_size', type=int, default=None,
                       help='Minibatch size for EMA or minibatch K-Means clustering [4096]')
    parser.add_argument('--max_step', type=int, default=None,
                       help='Number of maximum steps for clustering [300000]')
    parser.add_argument('--gamma', type=float, default=None,
                       help='Decay weight for EMA clustering [0.99]')
    parser.add_argument('--thres', type=float, default=None,
                       help='The minimal change of centroids at one minibatch step [None]')

    # configuration for visualising learned clusters
    parser.add_argument('--show_images', action='store_true',
                       help='Whether to show image pairs within clusters [0]')
    parser.add_argument('--sample_top', type=int, choices=[0,1], default=0,
                       help='Whether to sample nearest points for each cluster [0]')
    parser.add_argument('--sample_bottom', type=int, choices=[0,1], default=0,
                       help='Whether to sample farthest points for each cluster [0]')  
    parser.add_argument('--show_diff_cls', type=int, choices=[0,1], default=0,
                       help='Whether to show different classes [0]')
    parser.add_argument('--num_clusters_shown', type=int, default=-1,
                       help='number of clusters to visualize [-1] (all)')
    parser.add_argument('--num_images_shown', type=int, default=5,
                       help='Number of images to show [5]')

    args = parser.parse_args()

    return parser, args
