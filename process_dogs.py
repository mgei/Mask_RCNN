import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN-master/samples/coco/"))  # To find local version
import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Import dog classifier engine
sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN-master/dog-classifier/"))
import dog_classify


classifier = dog_classify.Dog_Classifier()
classifier.train_model()

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


dog_names = classifier.dog_names
class_names = class_names + dog_names

# Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN-master/dog-classifier/"))
image = skimage.io.imread(os.path.join(IMAGE_DIR, "MANY_DOGS.jpg"))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])


def classify_by_dog_breed(image):
    new_class_names = []
    breed_labels = []
    score_labels = []
    
    def extract_dogs(image):
        dog_subimages = []
        for i in range(0, len(r['class_ids'])):
            if r['class_ids'][i] == 17:   # class ID for DOG in Coco 
                y1, y2, x1, x2 = r['rois'][i][0], r['rois'][i][2], r['rois'][i][1], r['rois'][i][3]
                dog_subimages.append(image[y1:y2, x1:x2])
        return dog_subimages
    
    def get_dog_breeds(dog_subimages):
        results = []
        for dog in dog_subimages:
            result = classifier.predict_breed(dog)
            results.append(result)
        return results
    
    def change_labels(results):

        for i in results:
            breed_labels.append(i[0][0])
            score_labels.append(i[1][0])
        
        #global class_names
        #class_names = class_names + breed_labels
    
        counter = 0
        for i in range(0, len(r['class_ids'])):
            if r['class_ids'][i] == 17:
                r['class_ids'][i] = class_names.index(breed_labels[counter])
                r['scores'][i] = score_labels[counter]
                counter += 1 
    
    dogs = extract_dogs(image)
    breeds = get_dog_breeds(dogs)
    change_labels(breeds)    
    

import cv2
import numpy as np
from visualize_cv2 import model, display_instances, class_names


capture = cv2.VideoCapture('FRANK.mp4')
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('frank.avi', codec, 30.0, size)

while(capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        # add mask to frame
        results = model.detect([frame], verbose=0)
        r = results[0]
        classify_by_dog_breed(frame)
        #class_names = class_names + breed_labels
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        output.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
output.release()
cv2.destroyAllWindows()
