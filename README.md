Pytorch implementation for our ICPR-24 paper:

**“From One to Many Lorikeets: Discovering Image Analogies in the CLIP Space”**.
Songlong Xing, Elia Peruzzo, Enver Sangineto, Nicu Sebe.

The main file implements the whole pipeline, including preprocessing of the target dataset, image encoding, analogy discovery by clustering, and visulization.

### Setup
- Install CLIP:
  ```shell script
  conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
  pip install ftfy regex tqdm gdown
  pip install git+https://github.com/openai/CLIP.git
  ```
- Create conda env
  ```shell script
  cd ./CLIP-Analogy
  conda env create -f environment.yml
  ```
- Set up dataset path
  ```shell script
  mkdir datasets
  ```
  Download the raw dataset of ImageNet and put it under ./datasets.

### Implement
```shell script
python main.py --clip_vision_model 'ViT-B/16' --dataset 'ImageNet' --train_cluster --cluster_type 'minibatch_kmeans' --num_cluster 256 --on_the_fly 1 --reassignment_ratio 0.1 --normalize_diff 1 --minibatch_size 1048576
```
You can change the arguments in your own implementation. If you would like to run our pipeline on other datasets, download your dataset in ```./datasets``` and specify the dataset in ```--dataset``` when running ```main.py```.

After running this process, the learned clusters will be saved in ```./learned_clusters/ViT-B16/minibatch_kmeans/cluster_run_0```. Feel free to modify the target path in the main file.
