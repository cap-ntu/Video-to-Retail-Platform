<h1 align="center">
Hysia Video to Online Platform
</h1>

<sub>* This project is supported by 
[Cloud Application and Platform Lab](https://wiki.withcap.org) 
led by [Prof. Yonggang Wen](https://www.ntu.edu.sg/home/ygwen/)</sub>  

<p align="center">
    <a href="https://www.python.org/downloads/release/python-369/" title="python version"><img src="https://img.shields.io/badge/Python-3.6%2B-blue.svg"></a>
    <a href="https://travis-ci.com/cap-ntu/Video-to-Online-Platform" title="Build Status"><img src="https://travis-ci.com/cap-ntu/Video-to-Online-Platform.svg?branch=master"></a>
    <a href="https://app.fossa.com/projects/git%2Bgithub.com%2Fcap-ntu%2FVideo-to-Online-Platform?ref=badge_shield" alt="FOSSA Status"><img src="https://app.fossa.com/api/projects/git%2Bgithub.com%2Fcap-ntu%2FVideo-to-Online-Platform.svg?type=shield"/></a>
    <a href="https://www.codacy.com/gh/cap-ntu/Video-to-Online-Platform?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cap-ntu/Video-to-Online-Platform&amp;utm_campaign=Badge_Grade" title="Codacy Badge"><img src="https://app.codacy.com/project/badge/Grade/aeb994fbdb36493e8b5a3a62edcfd24f"></a>
    <a href="https://codebeat.co/projects/github-com-cap-ntu-video-to-online-platform-master"><img alt="codebeat badge" src="https://codebeat.co/badges/a29fe416-0b03-4c2a-b416-287337e96c63" /></a>    <a href="https://github.com/cap-ntu/Video-to-Online-Platform/graphs/commit-activity" title="Maintenance"><img src="https://img.shields.io/badge/Maintained%3F-YES-yellow.svg"></a>
    <a href="https://gitter.im/Video-to-Online-Platform/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge" title="Gitter"><img src="https://badges.gitter.im/Video-to-Online-Platform/community.svg"></a>
    <a href="https://hub.docker.com/repository/docker/hysia/hysia"><img src="https://img.shields.io/docker/image-size/hysia/hysia/v2o"></a>
</p>


An intelligent multimodal-learning based system for video, product and ads analysis. You can build various downstream 
applications with the system, such as product recommendation, video retrieval. Several examples are provided.

**V2** is under active development currently. You are welcome to create a issue, pull request here. We will credit them
into V2.

![hysia-block-diagram](docs/img/hysia-block-diagram.png)

<p align="center">
  <a href="#showcase">Showcase</a> •
  <a href="#features">Features</a> •
  <a href="#setup-environment">Setup Environment</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#demo">Demo</a> •
  <a href="#credits">Credits</a> •
  <a href="#about-us">About Us</a>
</p>

## Showcase

:point_right: Full list of [showcase](docs/Showcase.md).

<ol>
<li> Upload video and process it by selecting different models  

![select-models](docs/img/select-models.gif)

</li>

<li> Display video processing result  
    
![display-analytic-result](docs/img/display-analytic-result.gif)

</li>

<li> Search scene by image and text
    
![search-result](docs/img/search-result.gif)

</li>

<li> Insert product advertisement and display insertion result
    
![view-ads](docs/img/view-ads.gif)

</li>

</ol>

## Features

- Multimodal learning-based video analysis:
    - Scene / Object / Face detection and recognition
    - Multimodality data preprocessing
    - Results align and store
- Downstream applications:
    - Intelligent ads insertion
    - Content-product match
- Visualized testbed
    - Visualize multimodality results
    - Can be installed seperatelly

## Setup Environment

### 1. Download Data

:point_right: For [:x: no Google Drive access](CONTRIBUTING.md#1-download-data).

```shell script
# Make sure this script is run from project root
bash scripts/download-data.sh
```

### 2. Installation

:point_right: Install with Docker :whale:

```shell script
docker pull hysia/hysia:v2o
```

## Configuration

Change decoder and model server running devices at [device_placement.yml](server/config/device_placement.yml):  
```yaml
decoder: CPU
visual_model_server: CUDA:1
audio_model_server: CUDA:2
feature_model_server: CUDA:3
product_search_server: CUDA:2
scene_search_server: CUDA:3
```

Device value format: `cpu`, `cuda` or `cuda:<int>`.

## Demo

Run with docker :whale:
```shell script
docker run --rm \
  --gpus all -d -p 8000:8000 \
  --mount source=server/config/device_placement.yml,target=/content/server/config/device_placement.yml \
  hysia/hysia:v2o
```

Then you can go to http://localhost:8000. Use username: admin and password: admin to login.

## Some Useful Tools

- Large dataset preprocessing
- Video/audio decoding
- Model profiling
- Multimodality data testbed

## Credits

Here is a list of models that we used in Hysia-V2O. 

| Task                  | Model Name                  | License    | GitHub Repo |
| --------------------- |:---------------------------:|:----------:|:-----------:|
| MMDetection           |                             |            |             |
| Object detection      |                             | Apache-2.0 | [TensorFlow detection model zoo] |
|                       | [SSD MobileNet v1 COCO]     |            |             |
|                       | [SSD Inception v2 COCO]     |            |             |
|                       | [FasterRCNN ResNet101 COCO] |            |             |
| Scene Recognition | | |
| Audio Recognition | | |
| Image Retrieval | | |
| Face Detection | | |
| Face Recognition | | |
| Text Detection| | |
| Text Recognition| | |

## Contributing

You are welcome to contribute to Hysia! Please refer to [here](CONTRIBUTING.md) to get start.

## Paper Citation

Coming soon!


## About Us

### Maintainers
- Huaizheng Zhang [[GitHub]](https://github.com/HuaizhengZhang)
- Yuanming Li [[GitHub]](https://github.com/YuanmingLeee)
- Qiming Ai [[GitHub]](https://github.com/QimingAi)
- Shengsheng Zhou [[GitHub]](https://github.com/ZhouShengsheng)

### Previous Contributors
- Wenbo Jiang (Now, Shopee) [[GitHub]](https://github.com/Lancerchiang)
- Ziyuan Liu (Now, Tencent) [[GitHub]](https://github.com/ProgrammerYuan)
- Yongjie Wang (Now, NTU PhD) [[GitHub]](https://github.com/iversonicter)


[Tensorflow detection model zoo]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
[SSD MobileNet v1 COCO]: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
[SSD Inception v2 COCO]: http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
[FasterRCNN ResNet101 COCO]: http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz

