# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 01:15:27 2022

@author: Ibrahim
"""

import numpy as np
import cv2
import os.path
from knn import KNearestNeighbor


Dataset = 'D:\MS courses\Third Semester\Deep learning\Project\pets'                            # Provide the path of the directory containing all the training images.
Prediction = 'D:\MS courses\Third Semester\Deep learning\Project\dogs'           # Provide the path of the directory containing all the testing images 
labels = ["Cat","Dog" ]

def pre_processing(directory_path):
    
    img = cv2.imread(directory_path)
    resize = cv2.resize(img,(200, 200),  interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    pred_image = np.asarray(gray)
    pred_image =pred_image / 255
    
    return pred_image

def load(directory):

    train_img = []
    for image in os.listdir(directory):
        if (os.path.isfile(directory + "/" + image)):
            pred_image = pre_processing(directory + "/" + image)
            train_img.append(pred_image)
    X = np.array(train_img)
    return X


X = []
X = load(Dataset)

y0 = np.zeros(10)
y1 = np.ones(10)
y = []
y = np.concatenate((y0,y1), axis=0)

from builtins import range


num_training = X.shape[0]
mask = list(range(num_training))
X_train = X[mask]
y_train = y[mask]

num_test = X.shape[0]
mask = list(range(num_test))
X_test = X[mask]
y_test = y[mask]

print("X_train: "+str(X_train.shape))
print("X_test: "+str(X_test.shape))
print("y_train: "+str(y_train.shape))
print("y_test: "+str(y_test.shape))


# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


print("X_train: "+str(X_train.shape))
print("X_test: "+str(X_test.shape))
print("y_train: "+str(y_train.shape))
print("y_test: "+str(y_test.shape))

def prediction_function(image_name):
    
    predictions = pre_processing(Prediction +'/'+ image_name)
    predictions = np.reshape(predictions, (1, predictions.shape[0]*predictions.shape[1]))
    classification = KNearestNeighbor()
    classification.train(X_train, y_train)
    distance_L2 = classification.compute_distances(predictions)
    y_test_pred = classification.predict_labels(distance_L2, k=1)                                    # k=3 is giving best performance on this limited data
    print('Predicted '+image_name + ' as a ' + labels[int(y_test_pred)])


print("Predicting custom images")
for names in os.listdir(Prediction):    
    prediction_function(names)