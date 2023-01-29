# Positive-Augmented Contrastive Learning for Image and Video Captioning Evaluation
This repository contains the reference code for the paper "Positive-Augmented Contrastive Learning for Image and Video Captioning Evaluation".

<p align="center">
  <img src="images/model.png" alt="PACS" width="820" />
</p> 

## Environment Setup
Clone the repository and create the ```pacs``` conda environment using the ```environment.yml``` file:


```
conda env create -f environment.yml
conda activate pacs
```

## Loading CLIP Models and Data Preparation
Checkpoints of different backbones are available at [this link](https://drive.google.com/drive/folders/15Da_nh7CYv8xfryIdETG6dPFSqcBiqpd?usp=sharing).

Once you have downloaded the checkpoints, place them under the ```checkpoints/``` folder.

| **Backbone**       | **Checkpoint**         |
| -------------- | -------------      |
| **CLIP ViT-B-32**  | clip_ViT-B-32.pth  |
| **OpenCLIP ViT-L-14**  |  openClip_ViT-L-14.pth |

An example set of inputs, including a candidate json, image directory, and references json is provided in this repository under ```example/```. The input files are formatted as follows.

The candidates json should be a dictionary that maps from {"image_identifier": "candidate_captions"}:
```
{"image1": "A white dog is laying on the ground with its head on its paws .",
  ...}
```
The image directory should be a directory containing the images that act as the keys in the candidates json:
```
images/
├── image1.jpg
└── image2.jpg
```
The references json should be a dictionary that maps from {"image_identifier": ["list", "of", "references"]}:
```
{"image1":
    [
        "A closeup of a white dog that is laying its head on its paws .",
        "a large white dog lying on the floor .", 
        "A white dog has its head on the ground .",
        "A white dog is resting its head on a tiled floor with its eyes open .",
        "A white dog rests its head on the patio bricks ."
    ]}
```
## Quick Start: Compute PAC-S

Run ```python -u compute_metrics.py``` to obtain standard captioning metrics (_e.g._ BLEU, METEOR, etc.) and PAC-S.

To compute RefPAC-S run ```python -u compute_metrics.py --compute_refpac```.

The default backbone used is the CLIP ViT-B-32 model. To use a different backcbone (_e.g._ OpenCLIP ViT-L/14 backbone) specify in the command input ```--clip_model open_clip_ViT-L/14```. 

```
BLEU-1: 0.6400
BLEU-4: 0.3503
METEOR: 0.3057
ROUGE: 0.5012
CIDER: 1.4918
PAC-S: 0.8264
RefPAC-S: 0.8393
```
Worse captions should get lower scores:

``` 
python -u compute_metrics.py --candidates_json example/bad_captions.json --compute_refpac  

BLEU-1: 0.4500
BLEU-4: 0.0000
METEOR: 0.0995
ROUGE: 0.3268
CIDER: 0.4259
PAC-S: 0.5772
RefPAC-S: 0.6357

```
## Human Correlation Scores

#### Flickr8k

The Flickr8k dataset can be downloaded at [this link](https://drive.google.com/drive/folders/1oQY8zVCmf0ZGUfsJQ_OnqP2_kw1jGIXp?usp=sharing).

Once you have downloaded the dataset, place them under the ```datasets/flickr8k``` folder.


#### Run Code and Expected Output

Run ```python -u compute_correlations.py``` to compute correlation scores on **Flickr8k-Expert** and **Flickr8k-CF** datasets.


``` 
Computing correlation scores on dataset: flickr8k_expert
BLEU-1   Kendall Tau-b: 32.175    Kendall Tau-c: 32.324
BLEU-4   Kendall Tau-b: 30.599    Kendall Tau-c: 30.776
METEOR   Kendall Tau-b: 41.538    Kendall Tau-c: 41.822
ROUGE    Kendall Tau-b: 32.139    Kendall Tau-c: 32.314
CIDER    Kendall Tau-b: 43.602    Kendall Tau-c: 43.891
PAC-S    Kendall Tau-b: 53.919    Kendall Tau-c: 54.292

Computing correlation scores on dataset: flickr8k_cf
BLEU-1   Kendall Tau-b: 17.946    Kendall Tau-c: 9.256
BLEU-4   Kendall Tau-b: 16.863    Kendall Tau-c: 8.710
METEOR   Kendall Tau-b: 22.269    Kendall Tau-c: 11.510
ROUGE    Kendall Tau-b: 19.903    Kendall Tau-c: 10.274
CIDER    Kendall Tau-b: 24.619    Kendall Tau-c: 12.724
PAC-S    Kendall Tau-b: 36.037    Kendall Tau-c: 18.628
```

For the reference based version of the PACScore, add ```--compute_refpac```.
