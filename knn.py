# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 01:17:07 2022

@author: Ibrahim
"""

import numpy as np

class KNearestNeighbor(object):

    def __init__(self):
        pass

    def predict_label(self, distance, k=1):
        test = distance.shape[0]
        predict_y = np.zeros(test)
        for i in range(test):
            closest_y = []
            closest_y = self.y_train[np.argsort(distance[i])][0:k]
            predict_y[i] = np.bincount(closest_y).argmax()
        return predict_y

    def train(self, X, y):
       
        self.X_train = X
        self.y_train = y

    def predict(self, X, k):
       
        distance = self.compute_distances(X)

        return self.predict_labels(distance, k=k)

    def compute_distances(self, X):
        
        test = X.shape[0]
        train = self.X_train.shape[0]
        distance = np.zeros((test, train))
        #########################################################################
        ########################## Calculating Distances ########################
        #########################################################################
        
        distance = np.sqrt((X ** 2).sum(axis=1, keepdims=1) + (self.X_train ** 2).sum(axis=1) - 2 * X.dot(self.X_train.T))

        return distance

    def predict_labels(self, distance, k=1):
       
        num_test = distance.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            nearest_y = []
            nearest_y = self.y_train[np.argsort(distance[i])][0:k]
            nearest_y = nearest_y.astype(int)
            y_pred[i] = np.bincount(nearest_y).argmax()
        return y_pred