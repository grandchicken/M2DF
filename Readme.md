# M2DF

[Coming soon]The implementation of EMNLP2023 paper "M2DF: Multi-grained Multi-curriculum Denoising Framework for Multimodal Aspect-based Sentiment Analysis"

## Overview

<img src="m2df.png" style="width:200px height:300px" />

## Setup

### Dependencies

```
+ python 3.7.13
+ tensorflow 1.12.0+cu113
+ numpy 1.21.6
```


### Download and preprocess the datasets

Because the image features after processing is very large, you can download them via the link [Google Drive](https://drive.google.com/drive/folders/1dXtmvkLi1N19Uv6zWClNnK2P3WYkexKQ?usp=share_link). It should be noted that the path of the data is consistent with the file tree.

```
├── /src/
│  ├── /data/
│  │  │  ├── /jsons/	       
│  │  │  │  ├── twitter15_info.json	        
│  │  │  │  ├── twitter17_info.json
│  │  │  │  ├── amended_similarity_by_region2015.json
│  │  │  │  ├── amended_similarity_by_region2017.json
│  │  │  │  ├── amended_similarity_by_whole2015.json
│  │  │  │  ├── amended_similarity_by_whole2017.json
│  ├── /twitter2015/
│  ├── /twitter2017/
│  ├── /twitter2015_box_att_NER/
│  ├── /twitter2017_box_att_NER/
```

### Usage

Train on twitter2015

sh 15_pretrain_full.sh

Train on twitter2017

sh 15_pretrain_full.sh