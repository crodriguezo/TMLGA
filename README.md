# Proposal-Free Temporal Moment Localization

The repository includes:
* I3D features for Charades-STA.
* I3D features for ANET (soon).
* Training and testing code
* Pre-trained weights for temporal moment localization in charades-sta.
* Evaluation

# Installation
## Requirements
We share a list of package used in our virtual enviroment in packagelist.txt, which can be used to create an identical environment with the following command

```bash
conda create --name myenv --file packageslist.txt
```

**spacy**
`python -m spacy download en_core_web_md`

## Download

**I3D Features**
* [Charades-STA](https://drive.google.com/open?id=16CNli3XE8B_Bsr3EzcRHu-VSI_juvv8t)
* ANET (comming soon)
* YouCookII (comming soon)
  
**GLoVe**
* [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip)


## Configuration
To run these code we first need to change the path directory to features, glove embeddings and annotations in `config/path_catalog.py`.

```python
"charades_sta_train" : {
    "feature_path": "<feature_path>",
    "ann_file_path" : "<path_to_project_code>/preprocessing/charades-sta/charades_sta_train_tokens.json",
    "embeddings_path" : "<path_to_glove>",
    },

"charades_sta_test" : {
    "feature_path": "<feature_path>",
    "ann_file_path" : "<path_to_project_code>/preprocessing/charades-sta/charades_sta_test_tokens.json",
    "embeddings_path" : "<path_to_glove>",
    },
```

  
# Training

```bash
python main.py --config-file=experiments/charades-sta_train.yaml
```

We use tensorboardx to visualize the training and testing stage of our network. 

```bash
tensorboard --logdir=experiments/visualization/charades_sta_train/
```
# Testing

**Pretrained weights**
* [charades-sta](https://drive.google.com/open?id=1SwvR-CeB3xL-UdqiHSPMoWRT2RFmP9Jh)

To run the pretrained weights we need to change the `experiments/charades_sta.yaml`  file and change the TEST.MODEL configuration and pointing to the file path of downloaded model.

```yaml
ENGINE_STAGE: "TESTER"
TEST:
    MODEL: "<path_to_project>/checkpoints/charades_sta/model_charades_sta"
```
## Citation
```bibtex
@article{opazo2019proposal,
 author = {Rodríguez-Opazo, Cristian and Marrese-Taylor, Edison and Saleh, Fatemeh Sadat and Li, Hongdong and Gould, Stephen},
 journal = {Winter Conference on Applications of Computer Vision},
 title = {Proposal-free Temporal Moment Localization of a Natural-Language Query in Video using Guided Attention},
 year = {2020}
}
```


    

