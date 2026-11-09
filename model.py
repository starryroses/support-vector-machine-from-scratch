"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
import numpy as np

def standardize_features(x):
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    std[std==0] = 1.0
    x = (x-mean)/std
    return x

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    w = np.zeros(n_features)
    b = 0.0
    return {
        "w":w,
        "b":b
    }

# Step 3 - compute_scores
import numpy as np

def compute_scores(x, params):
    return x@params["w"] + params["b"]

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    scores[scores<0] = -1
    scores[scores>=0] = 1
    return scores

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    return np.maximum(0,1-(y*score))

# Step 6 - svm_objective
import numpy as np
def svm_objective(x, y, params, reg_lambda):
    scores = compute_scores(x,params)
    mean_hinge = np.mean(hinge_loss_example(scores,y))
    reg = (reg_lambda)*(np.dot(params["w"],params["w"]))
    return mean_hinge + reg

# Step 7 - compute_gradients
import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    scores = compute_scores(x,params)
    margins = 1-y*scores
    mask = margins > 0
    dw = -np.sum(y[mask, None] * x[mask], axis=0) / len(y) + 2*reg_lambda*params["w"]
    db = -np.sum(y[mask]) / len(y)
    return {
        "dw":dw,
        "db":float(db)
    }

# Step 8 - apply_update
def apply_update(params, grads, learning_rate):
    w = params["w"] - learning_rate*grads["dw"]
    b = params["b"] - learning_rate*grads["db"]
    return {
        "w":w,
        "b":b
    }

# Step 9 - train_svm
def train_svm(x, y, learning_rate, reg_lambda, n_epochs):
    params = initialize_parameters(x.shape[1])
    for i in range(n_epochs):
        grads = compute_gradients(x,y,params,reg_lambda)
        params = apply_update(params,grads,learning_rate)
    return params

# Step 10 - predict_labels
import numpy as np

def predict_labels(x, params):
    scores = compute_scores(x,params)
    preds = predict_from_scores(scores)
    return preds

# Step 11 - accuracy_score (not yet solved)
# TODO: implement

