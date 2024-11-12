import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.inference import VariableElimination
import random
from query2df import query2df
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pdb
import yaml
import os
import gc

dir = os.getcwd()
with open(f'{dir}\configs\config_CRC.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

def influential_variables(data, target, model_bn, n_random_trials = 50):

    data = data.reset_index(drop=True)
    ordered_variables = ["Sex","Age", "SD", "SES", "PA", "Depression", "Smoking", "BMI","Alcohol","Anxiety", "Diabetes", "Hyperchol", "Hypertension"]

    dict_impact_patient = dict.fromkeys(list(range(len(data))))
    
    with ProcessPoolExecutor(max_workers=cfg["inputs"]['max_workers']) as executor:
        futures = [executor.submit(run_iteration, i, model_bn, ordered_variables, target, data, dict_impact_patient) for i in range(n_random_trials)]
        all_results = []
        for future in tqdm(as_completed(futures), total=n_random_trials, desc="Processing iterations inf vars"):
            # all_results.append(future.result())
            result = future.result()
            for key, df in result.items():
                dict_impact_patient[key] = pd.concat([dict_impact_patient.get(key, pd.DataFrame()), df], axis=0, ignore_index=True)
            
            del result, df
            gc.collect()


    for i in range(data.shape[0]):
        if i==0:
            grouped_data = pd.concat([data.iloc[i].rename(index = 'Evidence'), dict_impact_patient[i].replace(0,float('nan')).median(axis = 0).rename('Influence')], axis = 1)
        else:
            grouped_data_aux = pd.concat([data.iloc[i].rename(index = 'Evidence'), dict_impact_patient[i].replace(0,float('nan')).median(axis = 0).rename('Influence')], axis = 1)
            grouped_data = pd.concat([grouped_data, grouped_data_aux], axis = 0)
                
    def combine_categories(row):
                return f"{row.name} = {row['Evidence']}"

    grouped_data['Influential Variable and Reason'] = grouped_data.apply(combine_categories, axis=1)      
        

    heatmap_data = grouped_data[["Influential Variable and Reason", "Influence"]].sort_values(by = ["Influence"], ascending = False).copy().set_index(["Influential Variable and Reason"])

    mean_of_medians = heatmap_data.groupby("Influential Variable and Reason")["Influence"].mean()
    std_of_medians = heatmap_data.groupby("Influential Variable and Reason")["Influence"].std()

    # Create a DataFrame for the heatmap with both mean and std
    heatmap_data = pd.DataFrame({
        "Mean Influence": mean_of_medians,
        "Std of Medians": std_of_medians
    }).sort_values(by=["Mean Influence"], ascending=False)

    heatmap_data.dropna(subset=["Mean Influence"], inplace=True)

    annotations = (heatmap_data["Mean Influence"].round(1).astype(str) + " ± " + heatmap_data["Std of Medians"].round(1).astype(str)).values

    plt.figure(figsize=(2,8))
    ax = sns.heatmap(heatmap_data[['Mean Influence']], cmap='RdBu_r', annot=annotations.reshape(-1,1), fmt='s', linewidths=1, center = 0)
    
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    
    plt.savefig(f'images/{target}/{target}_influential_variables.png', bbox_inches='tight')
    plt.close()

    return heatmap_data




def run_iteration(n, model_bn, ordered_variables, target, data, dict_impact_patient):

    local_dict_impact_patient = {}

    model_infer = VariableElimination(model_bn)
    random.shuffle(ordered_variables)

    dropped = list(model_bn.nodes())
    for elem in list(model_bn.get_ancestral_graph(target).nodes):
        if elem == target:
            continue
        else:
            dropped.remove(elem)
    
    target_sample_aux = np.zeros((data.shape[0], len(data.iloc[0].drop(labels = dropped).dropna())))
    for i in range(data.shape[0]):

        sample = data.iloc[i].drop(labels = dropped).dropna()

        j = 0
        list_elem = []
        def_variables = [x for x in ordered_variables if x not in dropped]

        for elem in [x for x in ordered_variables if x not in dropped]:
            list_elem.append(elem)
            sample_aux = sample[list_elem].copy()
            sample_aux_dict = sample_aux.to_dict()
            q_sample_aux = model_infer.query(variables=[target], evidence = sample_aux_dict)

            target_sample_aux[i,j] =  np.log(1 - query2df(q_sample_aux, verbose = 0)["p"][0].copy())

            j += 1

        impact_aux = pd.DataFrame(columns=def_variables)
        aux = np.zeros(len(sample))
        
        for j in range(len(target_sample_aux[i])):
            if j == 0:

                sample_target = model_infer.query(variables=[target])
                aux[j] = (target_sample_aux[i,j] - np.log(1 - query2df(sample_target, verbose = 0)["p"][0].copy()))   / np.abs(np.log(1 - query2df(sample_target, verbose = 0)["p"][0].copy())) * 100
            
                continue

            else:
                aux[j] = (target_sample_aux[i,j] - target_sample_aux[i,j-1])  /  np.abs( target_sample_aux[i,j-1]) * 100
                                    
                        
        impact_aux = pd.DataFrame([aux], columns = def_variables)

        local_dict_impact_patient[i] = pd.concat([local_dict_impact_patient.get(i, pd.DataFrame()), impact_aux], axis=0, ignore_index=True)


    return local_dict_impact_patient