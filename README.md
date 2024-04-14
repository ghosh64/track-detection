# track-detection
Codebase for Weighted Branch Aggregation based Deep Learning Model for Track Detection in Autonomous Racing@ ICLR 2024, Tiny Papers Track. 

Our code is divided into two branches-FOV and webacnn. FOV contains all code related to generating the the upper and lower bound for the Field of View including annotated data files. Webacnn contains code related to generating the lane masks used for detecting the lane.

## Dataset

The dataset used to train the webacnn model can be found under data/ in this repository. This is a custom dataset that was created by sampling online racing videos. The sampled frames are all compiled into a single dataset that is divided into train-test splits. The videos used can be found at the following links:
[Video 1](https://youtu.be/2f1PtJV0vIs?si=9vsb7QVW6_21kysS)
[Video 2](https://youtu.be/S_jdcUVtaTU?si=YDfYDO5cto1HnBfG)
[Video 3](https://youtu.be/U7JcOEKw-r4?si=OA-p5JHZWvS55FSU)
[Video 4](https://youtu.be/cxxeRzfJ1_c?si=MwWrgL1rrcJfNcXy)

## Environment

To reproduce our environment:

```bash
conda env create -f environment.yml
```

## Run:

To train and test on our dataset on webacnn:

```bash
cd webacnn
python main.py
```
Run FoV model

The annotated train, test and validation data splits for FoV are available at track-detection/FOV/data. The images are annotated manually using roboflow in a yolo trainable format. Each label contains (class, x_center, y_center, width, height) of the annotated rectange. The upper and lower bounds for the FoV are (y_center-height/2, y_center+height/2) and this is used to train the DNN. 

To train and test on FoV Model:

```bash
cd FOV
python main.py
```
