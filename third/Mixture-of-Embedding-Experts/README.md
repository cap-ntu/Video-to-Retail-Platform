# Mixture-of-Embeddings-Experts

This github repo provides a Pytorch implementation of the Mixture-of-Embeddings-Experts model (MEE) [1].

## Usage example

Creating an MEE block:

```python
from model import MEE

'''
Initializig an MEE module
Input:
- video_modality_dim: dictionary of all video modality with input dimension and output embedding dimension.
In this example: You have face modality (input dimension 128, output embedding dimension 128), 
audio, visual and motion modalities as an example.
- text_dim: dimensionality of sentence representation (e.g 1000)

'''

video_modality_dim = {'face': (128,128), 'audio': (128*16,128),
'visual': (2048,2048), 'motion': (1024,1024)}

text_dim = 1000

mee_block = MEE(video_modality_dim, text_dim)

```

MEE forward pass:

```python
'''
Inputs:
- captions: an Nx1000 input (N sentences, 1000 is the dimension of the sentences)
- videos: a dictionary with the modalities input, for instance face_data is of size Nx128 or
visual_data is of size Nx2048.
- ind: ind provides binary list for each modality. 1 means the data modality is provided and 0 means the data is not provided.
For instance, if the visual modality is provided for all N inputs then visual_ind = np.ones((N)).
If the first half only are provided with the visual modality, then visual_ind = np.concatenate((np.ones((N/2)),np.zeros((N/2)), axis=0).
'''

videos = {'face': face_data, 'audio': audio_data, 'visual': visual_data, 'motion': motion_data}
ind = {'face': face_ind, 'audio': audio_ind, 'visual': visual_ind, 'motion': motion_ind}

# Gives matrix scores
matrix_result  = mee_block(captions, videos, ind, conf=True)

# Gives pairwise scores
pairwise_result = mee_block(captions, videos, ind, conf=False)
```


## Reproducing results on MPII dataset and MSR-VTT dataset

Downloading the data:

```bash
wget https://www.rocq.inria.fr/cluster-willow/amiech/ECCV18/data.zip
unzip data.zip
```


Training on MSR-VTT:

```bash
python train.py --epochs=100 --batch_size=64 --lr=0.0004  --coco_sampling_rate=0.5 --MSRVTT=True --coco=True
```

Training on MPII:

```bash
python train.py --epochs=50 --batch_size=512 --lr=0.0001  --coco=True
```

## Web demo
We implemented a small demo using our MEE model to perform Text-to-Video retrieval.
You can try to search for any videos from the MPII (Test/Val) or MSRVTT dataset with your 
own query. The model was trained on the MPII dataset.

The demo is available at: http://willow-demo.inria.fr

## References

If you use this code, please cite the following paper:

[1] Antoine Miech and Ivan Laptev and Josef Sivic, Learning a Text-Video Embedding from Incomplete and Heterogeneous Data, arXiv link: https://arxiv.org/abs/1804.02516
```
@article{miech18learning,
  title={Learning a {T}ext-{V}ideo {E}mbedding from {I}ncomplete and {H}eterogeneous {D}ata},
  author={Miech, Antoine and Laptev, Ivan and Sivic, Josef},
  journal={arXiv:1804.02516},
  year={2018},
}
```



Antoine Miech
