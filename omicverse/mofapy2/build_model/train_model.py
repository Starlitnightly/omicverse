"""
Module to train a bioFAM model
"""

import scipy as s
import pandas as pd
import numpy as np

from ..core.BayesNet import BayesNet


def train_model(model):

    # Sanity check on the Bayesian Network
    assert isinstance(model, BayesNet), "'model' has to be a BayesNet class"


    ####################
    ## Start training ##
    ####################

    print ("\n")
    print ("#"*38)
    print ("## Training the model with seed %d ##" % (model.options['seed']))
    print ("#"*38)
    print ("\n")

    model.iterate()

    print("\n")
    print("#"*23)
    print("## Training finished ##")
    print("#"*23)
    print("\n")
