# Author: Matthew Wicker

import math
import numpy as np
import tensorflow as tf
from tqdm import trange

def layer_wise_relevance_propagation(model, input):
    """
	LRP algorithm (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
	param model - a deepbayes posterior or optimizer object
	param input - an input to pass through the model

	returns np.array of the same shape as input with a 
		'relevance' attribution for each feature dimension
    """
    return None


def deeplift(model, input_ref, input):
    """
        DeepLift algorithm (http://proceedings.mlr.press/v70/shrikumar17a.html)
        param model - a deepbayes posterior or optimizer object
	param input_ref - a refence input to check against the passed input
        param input - an input to pass through the model

        returns np.array of the same shape as input with a
                'relevance' attribution for each feature dimension
    """
    return None

def shapely_values(model, input, samples=100):
    """
        Shapley value based on sampling algorithm (https://www.sciencedirect.com/science/article/pii/S0305054808000804)
        param model - a deepbayes posterior or optimizer object
        param input - an input to pass through the model
        param samples - a number of samples to use to compute the value

        returns np.array of the same shape as input with a 
                'relevance' attribution for each feature dimension
    """
    return None


def occlusion_attr(model, input):
    """
        Occlusion attribution algorithm (https://arxiv.org/abs/1311.2901)
        param model - a deepbayes posterior or optimizer object
        param input - an input to pass through the model

        returns np.array of the same shape as input with a 
                'relevance' attribution for each feature dimension
    """
    return None
