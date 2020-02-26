# Proposal-Free Temporal Moment Localization

Code accompanying the paper [Proposal-free Temporal Moment Localization of a Natural-Language Query in Video using Guided Attention](https://arxiv.org/abs/1908.07236). 

This repository includes:

* Code for training and testing our model for temporal moment localization 
in the Charades-STA and Activity-Net datasets.
* Links to the I3D features we extracted for the Charades-STA and Activity-Net which were used for the experiments in our paper.
* Links to pre-trained models on the Charades-STA and Activity-Net datasets. 

# Installation

1. Clone this repo                                                                                                              
   ```bash
   git clone https://github.com/crodriguezo/TMLGA.git
   cd TMLGA
   ```

2. Create a conda environment based on our dependencies and activate it

   ```bash
   conda create -n <name> --file packageslist.txt
   conda activate <name>
   ```

   Where you can replace `<name>` by whatever you want.

2. Download everything
   ```bash
   sh ./download.sh
   ```
   This script will download the following things in the folder `~/data/TMLGA`: 
   * The `glove.840B.300d.txt` pretrained word embeddings.
   * The I3D features for Charades-STA and Activity-Net we extracted and used in our experiments.

   If you would like to change the default output folder for these downloads, please run `sh ./download.sh <download_path>`.

   This script will also install the `en_core_web_md` pre-trained spacy model, and download weights of our model pre-trained on the Charades-STA and Activity-Net datasets on the folders `./checkpoint/chares_sta` and `./checkpoint/anet` respectively.
   
   Downloading everything can take a while depending on your internet connection, please be patient. 

## Configuration
 If you have modified the download path from the defaults in the script above please modify the contents of the file `./config/settings.py` accordingly.
  
# Training

To train our model in the Charades-STA dataset, please run:
```bash
python main.py --config-file=experiments/charades-sta_train.yaml
```
We use tensorboardX to visualize progress of our model during training. Please run the followig command to see launch tensorborad:  
```bash
tensorboard --logdir=experiments/visualization/charades_sta_train/
```
# Testing

To load our pre-trained model and test it, first make sure the weigths have been downloaded and are in the `./checkpoints/charades_sta` foldel. Then simply run:

```bash
python main.py --config-file=experiments/charades-sta.yaml
```

# Download Links

If you are interested in downloading some specific resource only, we provide the links below.

**I3D Features**
* [Charades-STA](https://zenodo.org/record/3590426/files/i3d_charades_sta.zip)
* Activity-Net (coming soon)
* TACoS (coming soon)
* YouCookII (coming soon)
  
**GLoVe**
* [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip)

**Pretrained weights**

* [Charades-STA](https://zenodo.org/record/3590426/files/model_charades_sta)
* TACoS (coming soon)
* Activity-Net (coming soon)


# Citation

If you use our code or data please consider citing our work.

```bibtex
@article{opazo2019proposal,
 author = {Rodríguez-Opazo, Cristian and Marrese-Taylor, Edison and Saleh, Fatemeh Sadat and Li, Hongdong and Gould, Stephen},
 journal = {Winter Conference on Applications of Computer Vision},
 title = {Proposal-free Temporal Moment Localization of a Natural-Language Query in Video using Guided Attention},
 year = {2020}
}
```


    

