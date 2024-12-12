import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

from pgmpy.models import BayesianNetwork

from preprocessing import preprocessing
from data_cleaning.cleaning_disc import data_clean_discrete

import os
import yaml

import pdb



def create_pscount_dict_from_model(model_bn, card_dict, prior_weight, size_prior_dataset):
    model_infer = VariableElimination(model_bn)
    dict_ps = {}
    for variable in list(model_infer.variables):
        dict_ps[variable] = (model_infer.model.get_cpds(variable).values.reshape(model_bn.get_cardinality(variable), card_dict[variable]) * size_prior_dataset*50).tolist()
    
    return dict_ps


def prior_update_iteration(model_bn, card_dict, pscount_dict, size_prior_dataset, config_file, logger):
    dir = os.getcwd()
    with open(f'{dir}\configs\{config_file}', 'r') as file:
        cfg = yaml.safe_load(file)

    years = [2012,2013,2014,2015]
    counts_per_year = dict.fromkeys(years, None)
    prior_weight_list = [1, 50, 100, 500]
    for i in range(len(years)):
        df_aux = pd.read_csv("data/af_clean.csv", index_col = None)
        df_aux = data_clean_discrete(df_aux, selected_year = years[i], cancer_type = cfg["cancer_type"], cancer_renamed = cfg["cancer_renamed"], logger=logger)
        df_aux = preprocessing(df_aux, cancer_type = cfg["cancer_renamed"])
        
        counts_tables = model_bn.fit(df_aux, estimator=BayesianEstimator,weighted = False, prior_type = 'dirichlet', pseudo_counts = pscount_dict, n_jobs = -1)

        pscount_dict = create_pscount_dict_from_model(model_bn, card_dict, prior_weight_list[i], size_prior_dataset)

        logger.info(f"Year {years[i]} update completed")
        counts_per_year[years[i]] = counts_tables

    model_infer = VariableElimination(model_bn)
    return model_infer, counts_per_year