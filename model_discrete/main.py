import os
import pandas as pd
import numpy as np

from pgmpy.models import BayesianNetwork

from pgmpy.estimators import HillClimbSearch, BDsScore
from pgmpy.factors.discrete import State
from data_cleaning.cleaning_disc import data_clean_discrete

import pyAgrum as gum
import pyAgrum.lib.image as gumimage
import matplotlib.pyplot as plt

from pgmpy.models import BayesianNetwork

from table_statistics import from_counts_to_mean_and_variance, csv_quantiles

from risk_mapping import heatmap_plot_and_save
from predictive_interval import predictive_interval

from influential_variables import influential_variables

from evaluation_classification import evaluation_classification

import configs.config_CRC as lists_CRC


from preprocessing import preprocessing

import yaml

dir = os.getcwd()
with open(f'{dir}\configs\config_CRC.yaml', 'r') as file:
    cfg = yaml.safe_load(file)


import pdb

# ---- Read CSV and short preprocessing ---------------------------------

file_path = os.path.join(dir, "data/af_clean.csv")

df = pd.read_csv(file_path, index_col = None)
df = data_clean_discrete(df, selected_year = 2012, cancer_type = cfg["cancer_type"], cancer_renamed = cfg["cancer_renamed"])
df = preprocessing(df)

print("Successful data read")
# -----------------------------------------------------------------------


# ---- Structure Learning -----------------------------------------------
target = cfg["inputs"]["target"]
blck_lst = [tuple(item) for item in cfg["black_list"]]  
fxd_edges = [tuple(item) for item in cfg["fixed_edges"]] 

est = HillClimbSearch(data = df)
model = est.estimate(scoring_method=BDsScore(df, equivalent_sample_size = 5), fixed_edges=fxd_edges, black_list=blck_lst)
print("Successful structure learning")
# -----------------------------------------------------------------------


# ----- Save learned model ----------------------------------------------

if not os.path.exists(f"images/{target}"):
        os.mkdir(f"images/{target}")

if not os.path.exists(f"riskmap_datasets/{target}"):
        os.mkdir(f"riskmap_datasets/{target}")

# PRIOR NET
bn_gum = gum.BayesNet()
bn_gum.addVariables(list(df.columns))
bn_gum.addArcs(list(fxd_edges))

path = f"images/{target}/"
file_name = str('cancer_colorrectal_prior') + '.png'
file_path = os.path.join(path,file_name)

gumimage.export(bn_gum, file_path, size = "20!",
                nodeColor = cfg["node_color"],
                            )

# POSTERIOR NET
bn_gum_2 = gum.BayesNet()
bn_gum_2.addVariables(list(df.columns))
bn_gum_2.addArcs(list(model.edges))

arcColor_mine = dict.fromkeys(bn_gum_2.arcs(), 0.3)
for elem in list(bn_gum.arcs()):
    arcColor_mine[elem] = 1

path = f"images/{target}/"
file_name = str('cancer_colorrectal_learned_bds') + '.png'
file_path = os.path.join(path,file_name)

gumimage.export(bn_gum_2, file_path, size = "20!",
                 nodeColor = cfg["node_color"],
              
                cmapArc =  plt.get_cmap("hot"),
                arcColor= arcColor_mine )

print("Successful graphic models save")
# -----------------------------------------------------------------------


# ---- Parameter estimation ---------------------------------------------
model_bn = BayesianNetwork(model)


card_dict = dict.fromkeys(model.nodes, 1)
for node in card_dict.keys():
    for parent in model.get_parents(node): 
        card_dict[node] *= len(set(df[parent]))


# Prior parameters
size_prior_dataset = len(df) / 10000

pscount_dict = {
    "Sex": [[df["Sex"].value_counts(normalize=True).sort_index().iloc[i] * size_prior_dataset] for i in range(len(np.unique(df["Sex"])))],
    "Age": [[df["Age"].value_counts(normalize=True).sort_index().iloc[i] * size_prior_dataset] for i in range(len(np.unique(df["Age"])))],
    'BMI': [[df["BMI"].value_counts(normalize=True).sort_index().iloc[i] * size_prior_dataset]*card_dict["BMI"] for i in range(len(np.unique(df["BMI"])))],
    'Alcohol': [[df["Alcohol"].value_counts(normalize=True).sort_index().iloc[i] * size_prior_dataset]*card_dict["Alcohol"] for i in range(len(np.unique(df["Alcohol"])))],
    'Smoking': [[df["Smoking"].value_counts(normalize=True).sort_index().iloc[i] * size_prior_dataset]*card_dict["Smoking"] for i in range(len(np.unique(df["Smoking"])))],
    'PA': [[df["PA"].value_counts(normalize=True).sort_index().iloc[i] * size_prior_dataset]*card_dict["PA"] for i in range(len(np.unique(df["PA"])))],
    'SD': [[df["SD"].value_counts(normalize=True).sort_index().iloc[i] * size_prior_dataset]*card_dict["SD"] for i in range(len(np.unique(df["SD"])))],
    'SES': [[df["SES"].value_counts(normalize=True).sort_index().iloc[i] * size_prior_dataset]*card_dict["SES"] for i in range(len(np.unique(df["SES"])))],
    'Depression': [[df["Depression"].value_counts(normalize=True).sort_index().iloc[i] * size_prior_dataset]*card_dict["Depression"] for i in range(len(np.unique(df["Depression"])))],
    'Anxiety': [[df["Anxiety"].value_counts(normalize=True).sort_index().iloc[i] * size_prior_dataset]*card_dict["Anxiety"] for i in range(len(np.unique(df["Anxiety"])))],
    'Diabetes': [[df["Diabetes"].value_counts(normalize=True).sort_index().iloc[i] * size_prior_dataset]*card_dict["Diabetes"] for i in range(len(np.unique(df["Diabetes"])))],
    'Hypertension': [[df["Hypertension"].value_counts(normalize=True).sort_index().iloc[i] * size_prior_dataset]*card_dict["Hypertension"] for i in range(len(np.unique(df["Hypertension"])))],
    'Hyperchol': [[df["Hyperchol"].value_counts(normalize=True).sort_index().iloc[i] * size_prior_dataset]*card_dict["Hyperchol"] for i in range(len(np.unique(df["Hyperchol"])))],
    'CRC': [[df["CRC"].value_counts(normalize=True).sort_index().iloc[i] * size_prior_dataset]*card_dict["CRC"] for i in range(len(np.unique(df["CRC"])))],
}


# Parameter estimation
model_bn = BayesianNetwork(model)

from parameter_estimation import prior_update_iteration

model_infer, counts_per_year = prior_update_iteration(model_bn, card_dict, pscount_dict = pscount_dict, size_prior_dataset=size_prior_dataset)

print("Successful parameter estimation")
# ----------------------------------------------------------------------


# ---- Save model statistics of interest (90% posterior predictive interval) -----

if not os.path.exists("bounds"):
        os.mkdir("bounds")

mean, var = from_counts_to_mean_and_variance( counts_per_year[2012][0] )

csv_quantiles(model_bn, counts_per_year=counts_per_year)

print("Successful statistics save")
# -----------------------------------------------------------------------


# ---- Risk mapping -----------------------------------------------------

col_var = cfg["pointwise_risk_mapping"]["col_var"]
row_var = cfg["pointwise_risk_mapping"]["row_var"]

heatmap_plot_and_save(df, model_bn, target, col_var, row_var)


# If calculate interval = True, an approximation of the posterior predictive intervals will be 
# by sampling. However, it is a task that requires relatively large computation and time 
# resources, so we encourage to use the example case available.

calculate_interval = cfg["inputs"]["calculate_interval"]
if calculate_interval:
    predictive_interval(model_bn, col_var, target, row_var, path_to_data = "interval_df/")

col_var = cfg["interval_risk_mapping"]["col_var"]
row_var = cfg["interval_risk_mapping"]["row_var"]

heatmap_plot_and_save(df, model_bn, target, col_var, row_var, interval = True)

print("Successful risk mapping")

# -----------------------------------------------------------------------



# ---- Influential variables --------------------------------------------

df_pos = df[df[target] == True].copy()

# Increase the n_random_trials to get meaningful results.
heatmap_data = influential_variables(data=df_pos, target=target, model_bn = model_bn, n_random_trials = cfg["inputs"]["n_random_trials"])

print("Successful influential variables")
# -----------------------------------------------------------------------



# ---- Evaluation of the model ------------------------------------------

file_path = os.path.join(dir, "data/af_clean.csv")
df_test = pd.read_csv(file_path, index_col = None)
df_test = data_clean_discrete(df_test, selected_year = 2016, cancer_type = cfg["cancer_type"], cancer_renamed = cfg["cancer_renamed"])
df_test = preprocessing(df_test)

evaluation_classification(df_test, model_bn)

print("Successful evaluation of the model")
# -----------------------------------------------------------------------