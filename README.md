# track-detection
Codebase for Weighted Branch Aggregation based Deep Learning Model for Track Detection in Autonomous Racing@ ICLR 2024, Tiny Papers Track. 


## Folders
Since our contribution includes the introduction of the lane detection model, webacnn, and the model that predicts the field of view of a frame, FOV, we separate the files and dataset into "webacnn" and "FOV" folders.  

## Dataset

The dataset used to train the webacnn model can be found under data/ in this repository. This is a custom dataset that was created by sampling online racing videos. The sampled frames are all compiled into a single dataset that is divided into train-test splits. The videos used can be found at the following links:
[Video 1](https://youtu.be/2f1PtJV0vIs?si=9vsb7QVW6_21kysS)
[Video 2](https://youtu.be/S_jdcUVtaTU?si=YDfYDO5cto1HnBfG)
[Video 3](https://youtu.be/U7JcOEKw-r4?si=OA-p5JHZWvS55FSU)
[Video 4](https://youtu.be/cxxeRzfJ1_c?si=MwWrgL1rrcJfNcXy)

## Environment

To reproduce our environment, load the required packages as:

```bash
conda env create -f environment.yml
```

## Run:

To train and test on our dataset on webacnn:

```bash
cd webacnn
python main.py
```

To train and test on our dataset on webacnn:

```bash
cd FOV
python main.py
```
