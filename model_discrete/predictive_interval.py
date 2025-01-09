import pandas as pd
import numpy as np
import os
# os.environ["NUMEXPR_MAX_THREADS"] = "64"


from pgmpy.inference import ApproxInference, VariableElimination
from pgmpy.factors.discrete import State
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import yaml
import gc

from query2df import query2df


def predictive_interval(model_bn, col_var, row_var, n_samples = 30000 , q_length = 100, target_variable = "CRC", path_to_data = "interval_df", cfg = None, logger = None):
    if not os.path.exists(path_to_data):
        os.mkdir(path_to_data)
    
    model_infer = VariableElimination(model_bn)
    model_approx_infer = ApproxInference(model_bn)

    mod_columns = model_bn.states[col_var]
    mod_rows = model_bn.states[row_var]

    df_hom_inf = pd.DataFrame(columns = mod_columns, index = mod_rows )
    df_hom_sup = pd.DataFrame(columns = mod_columns, index = mod_rows )
    df_hom= pd.DataFrame(columns = mod_columns, index = mod_rows )

    df_hom_str = pd.DataFrame(columns = mod_columns, index = mod_rows )

    df_muj_inf = pd.DataFrame(columns = mod_columns, index = mod_rows )
    df_muj_sup = pd.DataFrame(columns = mod_columns, index = mod_rows )
    df_muj = pd.DataFrame(columns = mod_columns, index = mod_rows )

    df_muj_str = pd.DataFrame(columns = mod_columns, index = mod_rows )

    A_hom = model_infer.query(variables = [target_variable], evidence={"Sex": "M"}, show_progress = False)
    A_muj = model_infer.query(variables = [target_variable], evidence={"Sex": "W"}, show_progress = False)


    for i in range(len(model_bn.states["Sex"])):
        Sex = model_bn.states["Sex"][i]

        for j in range(len(mod_rows)):
            row = mod_rows[j]

            for k in range(len(mod_columns)):
                column = mod_columns[k]

                ev = [State("Sex", Sex), State(col_var,column), State(row_var,row)]

                q_point = np.log(1 - query2df(  model_infer.query(variables=[target_variable], 
                                                                            evidence={"Sex": Sex, col_var: column, row_var: row},
                                                                            show_progress = False)  ,   verbose = 0)["p"][0])
                

                df_partial_samples = pd.DataFrame(columns = ["Sex" , col_var, row_var])
                for n in range(n_samples): df_partial_samples = pd.concat([df_partial_samples, pd.DataFrame(data = {"Sex":[i], col_var:[k], row_var: [j]})])
                df_partial_samples.reset_index(drop=True, inplace = True)
                df_partial_samples = df_partial_samples.astype("int32")

                import time

                q = None
                chunk_size = 10
                n_chunks = int(q_length / chunk_size)
                
                
                with ProcessPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(run_iteration, i, chunk_size, model_approx_infer, target_variable, Sex, col_var, column, row_var, row, n_samples ) for i in range(int(n_chunks))]
                    all_results = []
                    for future in tqdm(as_completed(futures), total=n_chunks, desc="Processing iterations predictive interval"):
                        # all_results.append(future.result())
                        result = future.result()
                        
                        all_results.append(result)

                        del result
                        gc.collect()
                    
                    q = np.concatenate(all_results)

                a = np.sort(q)

                if Sex == "M":
                    df_hom_inf.loc[row,column] = round( a[round(q_length*5 / 100)] - np.log( 1 - query2df(A_hom, verbose = 0)["p"][0]) , 3 )
                    df_hom_sup.loc[row,column] = round( a[round(q_length*95 / 100)] - np.log( 1 - query2df(A_hom, verbose = 0)["p"][0]) , 3 )

                    df_hom_str.loc[row,column] = f"[ {df_hom_inf.loc[row,column]}, {df_hom_sup.loc[row,column]}]"

                    logger.info(f'Risk interval for men with {col_var} = {column} and {row_var} = {row} is: {df_hom_str.loc[row,column]} ({n_samples} samples and interval of size {q_length})')

                    df_hom.loc[row,column] = round( q_point - np.log( 1 - query2df(A_hom, verbose = 0)["p"][0]) , 3 )

                    logger.info(f'Pointwise estimation of the risk: {df_hom.loc[row,column]}')

                    df_hom.to_csv(f"{path_to_data}/df_hom_{col_var}_{row_var}_{q_length}_{n_samples}.csv")
                    df_muj.to_csv(f"{path_to_data}/df_muj_{col_var}_{row_var}_{q_length}_{n_samples}.csv")
                    
                else:
                    df_muj_inf.loc[row,column] = round( a[round(q_length*5/ 100)] - np.log( 1 - query2df(A_muj, verbose = 0)["p"][0]) , 3 )
                    df_muj_sup.loc[row,column] = round( a[round(q_length*95 / 100)] - np.log( 1 - query2df(A_muj, verbose = 0)["p"][0]) , 3 )

                    df_muj_str.loc[row,column] = f"[ {df_muj_inf.loc[row,column]}, {df_muj_sup.loc[row,column]}]"

                    logger.info(f'Risk interval for women with {col_var} = {column} and {row_var} = {row} is: {df_muj_str.loc[row,column]} ({n_samples} samples and interval of size {q_length})')

                    df_muj.loc[row,column] = round( q_point - np.log( 1 - query2df(A_muj, verbose = 0)["p"][0]) , 3 )

                    logger.info(f'Pointwise estimation of the risk: {df_muj.loc[row,column]}')

                    df_hom_str.to_csv(f"{path_to_data}/df_hom_{col_var}_{row_var}_{q_length}_{n_samples}_interval.csv")
                    df_muj_str.to_csv(f"{path_to_data}/df_muj_{col_var}_{row_var}_{q_length}_{n_samples}_interval.csv")

                df_hom.to_csv(f"{path_to_data}/df_hom_{col_var}_{row_var}_{q_length}_{n_samples}.csv")
                df_muj.to_csv(f"{path_to_data}/df_muj_{col_var}_{row_var}_{q_length}_{n_samples}.csv")

                df_hom_str.to_csv(f"{path_to_data}/df_hom_{col_var}_{row_var}_{q_length}_{n_samples}_interval.csv")
                df_muj_str.to_csv(f"{path_to_data}/df_muj_{col_var}_{row_var}_{q_length}_{n_samples}_interval.csv")

    return df_hom, df_hom_str, df_muj, df_muj_str



def run_iteration(i, chunk_size, model_approx_infer, target_variable, Sex, col_var, column, row_var, row, n_samples):
    q_iter = np.zeros(chunk_size)
    for k in range(chunk_size):
        q_iter[k] = np.log(1 - query2df(model_approx_infer.query(variables=[target_variable], evidence= {"Sex": Sex, col_var: column, row_var: row},n_samples = n_samples, show_progress = False), verbose = 0)["p"][0])

    return q_iter
    