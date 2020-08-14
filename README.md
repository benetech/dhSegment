# TextBook Segmentation

## Overview

A semantic image segmentation implementation of the dhSegment network, designed to segment textbook pages into sections which can be processed by either an OCR or classifier. 


## Software Stack
dhSegment has a list of requirements that it will automatically download in dhSegment/setup.py.

You must use Tensorflow 1.14, as they use tf.contrib, which was removed in later versions of Tensorflow.



## Project Layout
dhSegment - folder containing the dhSegment network including the train and visualize scripts.

demo - Folder left over from forked dhSegment. Unused.

doc - Folder left over from forked dhSegment. Unused.

config.json - file containing the parameters for training 

data/ - contains train/, val/, and test/, each of which contains an images/ directory and a labels/ directory

images/ - images of the original textbook page. These files must have the same names as the files in labels/

labels/ - segmented images of the textbook page. These files must have the same names as the files in images/

data/class.txt - txt file containing rgb values representing each class for the network to segment.

train.py - Used to train the model.

visualize.py - Used to visualize the model's results.


## Data & Model Storage

The dataset used to train the model and the fully trained model are stored in S3 using DVC. 

To bring in all DVC managed project componenets, run the following command. Note: You will need to be authenticated to the Git repo and AWS

```
dvc pull
```
Note that you may already have these data in TextbookSegmentationDatasetGenerator, so you could save some disk space by just pathing to that directory in the config file.


## Training
Before training, you must first download pretrained weights via dhSegment/pretrained_models/download_resnet_pretrained_model.py.
```
cd dhSegment/pretrained_models
python3 download_resnet_pretrained_model.py
```
There is an option to use a pretrained VGG model. We did not experiment with this.


```
python3 dhSegment/train.py with general_config.json
```

If you are training on your own dataset, you will need to adjust data/classes.txt to reflect your dataset. We do this for our dataset of novels with our novel_config.json.

## Inference/Visualization

visualize.py will take in a model, an image directory, and an output directory. It will segment each page, then draw the bounding boxes onto the image and export them. It will also export a txt file containing each bounding box's pixel coordinates and an xml file for each image containing the coordinates of each box.

```
python3 dhSegment/visualize.py <model_directory> <image_directory> <output_directory> 
```

## Changes from vanilla dhSegment
We added visualize.py, described above.

We moved bb_detection.py from somewhere else in the code to its current location. 

We also edited general_config.json to suit the needs of our textbook dataset and we created novel_config.json for our novel dataset.
