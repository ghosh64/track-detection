#track-detection
Codebase for Weighted Branch Aggregation based Deep Learning Model for Track Detection in Autonomous Racing@ ICLR 2024, Tiny Papers Track. 

## Dataset

The dataset used to train the model can be found under data/ in this repository. This is a custom dataset that was created by sampling online racing videos. The sampled frames are all compiled into a single dataset that is divided into train-test splits. The videos used can be found at the following links:
[Video 1](https://youtu.be/2f1PtJV0vIs?si=9vsb7QVW6_21kysS)
[Video 2](https://youtu.be/S_jdcUVtaTU?si=YDfYDO5cto1HnBfG)
[Video 3](https://youtu.be/U7JcOEKw-r4?si=OA-p5JHZWvS55FSU)
[Video 4](https://youtu.be/cxxeRzfJ1_c?si=MwWrgL1rrcJfNcXy)

## Environment

To reproduce our environment, load the required packages as:

'conda env create -f environment.yml'

## Run:

To train and test on our dataset:

python main.py

FoV model: FoV.py, use training functions in train_model.py to train and test.
