import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr, wasserstein_distance, ks_2samp, entropy
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

#-----------------------------------------------------
#---------------------- METRICS ----------------------
#-----------------------------------------------------

class AverageMeter(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models
    (from the Deep Learning tutorials of DSSC)
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
