import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

from missforest import MissForest

import time
import pdb


def redefine_imc(df):
    cond = [(df["imc"] < 18.5), ((df["imc"] >= 18.5) & (df["imc"] < 25)), ((df["imc"] >= 25) & (df["imc"] < 30)), (df["imc"] >= 30)]
    value = ["bmi_1_underweight", "bmi_2_normal", "bmi_3_overweight", "bmi_4_obese"]
    df["imc4"] = np.select(condlist = cond, choicelist=value, default = "err")
    return df

def redefine_age(df):
    cond_edad = [(df['edad'] < 18), ((df['edad'] >= 18) &  (df['edad'] <= 24)) , ((df['edad'] > 24) & (df['edad'] <= 34)) , ((df['edad'] > 34) & (df['edad'] <= 44)), ((df['edad'] > 44) & (df['edad'] <=  54)), ((df['edad'] > 54) & (df['edad'] <= 65)), (df['edad'] > 65)]
    val_edad = ["[0,18)","[18,24]", "(24,34]" , "(34,44]", "(44,54]", "(54,65]", ">65"]
    df["edad5"] = np.select(cond_edad, val_edad, default = "err")

    
    df.loc[df['edad5'] == '[18,24]', ["edad5"]] = "age_1_very_young"
    df.loc[df['edad5'] == '(24,34]',["edad5"]] = "age_2_young"
    df.loc[df['edad5'] == '(34,44]', ["edad5"]] = "age_3_young_adult"
    df.loc[df['edad5'] == '(44,54]',["edad5"]] = "age_4_adult"
    df.loc[df['edad5'] == '(54,65]', ["edad5"]] = "age_5_old_adult"
    df.loc[df['edad5'] == ">65", ["edad5"]] = "age_6_elderly"
    return df

def redefine_medical_cond(df):
    df['diabetes1'] = np.where(df['glucosa'] >= 125, True, df["diabetes1"] )
    df['hipertension1'] = np.where((df['tas'] >= 139) | (df['tad'] >= 90), True, df["hipertension1"] )
    df['hipercolesterolemia1'] = np.where((df['colesterol_ldl'] >= 130) | (df['colesterol_hdl'] <= 40) | (df['colesterol'] >= 200), True, df["hipercolesterolemia1"] )
    return df



def impute_missing_values_iter_imp(df):

    df_preimp = df[["año_reco", "peso", "talla", "imc", "af", "tad", "tas", "colesterol",
                         "glucosa", "diabetes","fumador", "consumo_alcohol", "condicion_socioeconomica_media"]].copy()

    # Manually label encode the columns of interest

    df_preimp["consumo_alcohol"] = df_preimp["consumo_alcohol"].map({"no":0, "esporadico":1, "fin de semana":2, "exconsumidor":3, "habitual":4, "dependiente":5})
    df_preimp["fumador"] = df_preimp["fumador"].map({"no fumador": 0, "ex-fumador":1, "fumador":2})
    df_preimp["af"] = df_preimp["af"].map({ 1:1, 2:2, 3:3,4:4,4.1:4, 4.2:4, 4.3:4, 5:5})

    imputer = IterativeImputer(max_iter = 50, random_state = 42, sample_posterior=False) # Uses Bayesian Ridge Regression as estimator
    df_imputed = pd.DataFrame(imputer.fit_transform(df_preimp), columns=df_preimp.columns)

    df_imputed["consumo_alcohol"] = df_imputed["consumo_alcohol"].round()
    df_imputed["fumador"] = df_imputed["fumador"].round()
    df_imputed["af"] = df_imputed["af"].round()
    
    
    # Now the inverse mapping
    df_imputed["consumo_alcohol"] = df_imputed["consumo_alcohol"].map({0:"no", 1:"esporadico", 2:"fin de semana", 3:"exconsumidor", 4:"habitual", 5:"dependiente"})
    df_imputed["fumador"] = df_imputed["fumador"].map({0:"no fumador", 1:"ex-fumador", 2:"fumador"})
    # df_imputed["af"] = df_imputed["af"].map({1:1, 2:2, 3:3, 4:4, 5:4.1, 6:4.2, 7:4.3, 8:5})

    df[["año_reco", "peso", "talla", "imc", "af", "tad", "tas", "colesterol",
        "glucosa", "diabetes","fumador", "consumo_alcohol", "condicion_socioeconomica_media"]] = df_imputed[["año_reco", "peso", "talla", "imc", "af", "tad", "tas", "colesterol",
                                                                                                            "glucosa", "diabetes","fumador", "consumo_alcohol", "condicion_socioeconomica_media"]].copy()
    
    return df

def impute_missing_values_missForest(df):

    selected_variables_to_impute = ["año_reco", "peso", "talla", "imc", "af", "tad", "tas", "colesterol", "duracion_sueño",
                         "glucosa", "diabetes","fumador", "consumo_alcohol", "condicion_socioeconomica_media"]

    df_preimp = df[selected_variables_to_impute].copy()

    # Manually label encode the columns of interest

    df_preimp["consumo_alcohol"] = df_preimp["consumo_alcohol"].map({"no":0, "esporadico":1, "fin de semana":2, "exconsumidor":3, "habitual":4, "dependiente":5})
    df_preimp["fumador"] = df_preimp["fumador"].map({"no fumador": 0, "ex-fumador":1, "fumador":2})
    df_preimp["af"] = df_preimp["af"].map({ 1:1, 2:2, 3:3,4:4,4.1:4, 4.2:4, 4.3:4, 5:5})

    imputer = MissForest(max_iter = 10)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_preimp, categorical = ['año_reco', 'af', 'duracion_sueño',
                                        'diabetes', 'fumador', 'consumo_alcohol'] ))

    # Now the inverse mapping
    df_imputed["consumo_alcohol"] = df_imputed["consumo_alcohol"].map({0:"no", 1:"esporadico", 2:"fin de semana", 3:"exconsumidor", 4:"habitual", 5:"dependiente"})
    df_imputed["fumador"] = df_imputed["fumador"].map({0:"no fumador", 1:"ex-fumador", 2:"fumador"})
    # df_imputed["af"] = df_imputed["af"].map({1:1, 2:2, 3:3, 4:4, 5:4.1, 6:4.2, 7:4.3, 8:5})

    df[selected_variables_to_impute] = df_imputed[selected_variables_to_impute].copy()
    
    return df


def impute_missing_values_knn(df):
    t1 = time.time()
    # NOT WORKING YET

    df_preimp = df[["año_reco", "peso", "talla", "imc", "af", "tad", "tas", "colesterol",
                         "glucosa", "diabetes","fumador", "consumo_alcohol", "condicion_socioeconomica_media"]].copy()

    # Manually label encode the columns of interest

    df_preimp["consumo_alcohol"] = df_preimp["consumo_alcohol"].map({"no":0, "esporadico":1, "fin de semana":2, "exconsumidor":3, "habitual":4, "dependiente":5})
    df_preimp["fumador"] = df_preimp["fumador"].map({"no fumador": 0, "ex-fumador":1, "fumador":2})
    df_preimp["af"] = df_preimp["af"].map({ 1:1, 2:2, 3:3,4:4,4.1:4, 4.2:4, 4.3:4, 5:5})

    imputer = KNNImputer(n_neighbors = 5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_preimp), columns=df_preimp.columns)
    
    # Now the inverse mapping
    df_imputed["consumo_alcohol"] = df_imputed["consumo_alcohol"].map({0:"no", 1:"esporadico", 2:"fin de semana", 3:"exconsumidor", 4:"habitual", 5:"dependiente"})
    df_imputed["fumador"] = df_imputed["fumador"].map({0:"no fumador", 1:"ex-fumador", 2:"fumador"})
    # df_imputed["af"] = df_imputed["af"].map({1:1, 2:2, 3:3, 4:4, 5:4.1, 6:4.2, 7:4.3, 8:5})

    df[["año_reco", "peso", "talla", "imc", "af", "tad", "tas", "colesterol",
        "glucosa", "diabetes","fumador", "consumo_alcohol", "condicion_socioeconomica_media"]] = df_imputed[["año_reco", "peso", "talla", "imc", "af", "tad", "tas", "colesterol",
                                                                                                            "glucosa", "diabetes","fumador", "consumo_alcohol", "condicion_socioeconomica_media"]].copy()
    t2 = time.time()
    print("Time elapsed: ", t2-t1, " seconds")
    return df

def unique_patient(df, cancer_type):

    # Still under revision

    dict_ids = dict.fromkeys(set(df["fpi"]), [])
    i = 0
    for id in df["fpi"]:
        fecha = df["fecha_reco"][i]
        cancer = df[cancer_type][i]
        lst = list(df.loc[i][1:])
        dict_ids[id] = dict_ids[id] + [[fecha, cancer, lst]]
        # dict_ids[id] = dict_ids[id] + [[fecha,str(cancer)]]   #Activar esta linea en caso de querer guardar en formato JSON (no reconoce booleanos)
        i += 1

    ult_fecha_pos = dict.fromkeys(set(df["fpi"]), None)
    j = 0
    for id in dict_ids.keys():
        # Ordenamos cada lista de fecha para que aparezcan en orden cronológico
        dict_ids[id] = sorted(dict_ids[id])
        for i in range(len(dict_ids[id])):
            if dict_ids[id][i][1] == True:
                # Escogemos para cada paciente con cancer los datos de su primer positivo detectado
                ult_fecha_pos[id] = dict_ids[id][i][2]
                break
        # En caso de no haber positivo, escogemos los datos médicos más recientes
        if ult_fecha_pos[id] == None:
            ult_fecha_pos[id] =  dict_ids[id][i][2]

    df_select = pd.DataFrame.from_dict(ult_fecha_pos, orient = 'index')
    df_select.reset_index(names = 'fpi', inplace = True)
    df_select.columns = df.columns

    return df_select

def redefine_ses(df):
    # Discretize continous variable. The approach is very naive. We should use a more sophisticated method in the future.

    ses_mean = df["condicion_socioeconomica_media"].mean() 
    ses_std = df["condicion_socioeconomica_media"].std()
    # 0.9, 1.1
    df["condicion_socioeconomica_media"] = pd.cut(df["condicion_socioeconomica_media"], [df["condicion_socioeconomica_media"].min(), 0.9, 1.1 ,df["condicion_socioeconomica_media"].max()], labels=False, include_lowest=True)

    df.loc[df['condicion_socioeconomica_media'] == 0, ["condicion_socioeconomica_media_3"]] = "ses_0"
    df.loc[df['condicion_socioeconomica_media'] == 1, ["condicion_socioeconomica_media_3"]] = "ses_1"
    df.loc[df['condicion_socioeconomica_media'] == 2, ["condicion_socioeconomica_media_3"]] = "ses_2"

    return df

def redefine_pa(df):

    # Meething WHO guidelines for PA
    '''arr = np.where(df['af'] < 4, 1, 2)

    df.loc[arr == 1, ["af_2"]] = "PA_1"
    df.loc[arr == 2, ["af_2"]] = "PA_2"'''

    # Inactive, insufficiently active, active
    df.loc[df['af'] == 1.0, "PA"] = "PA_1"
    df.loc[(df['af'] == 2.0) | (df['af'] == 3.0), "PA"] = "PA_2"
    df.loc[(df['af'] >= 4.0), "PA"] = "PA_3"


    # Original database categories
    '''df.loc[df['af'] == 1.0, "PA"] = "PA_1"
    df.loc[df['af'] == 2.0, "PA"] = "PA_2"
    df.loc[df['af'] == 3.0, "PA"] = "PA_3"
    df.loc[(df['af'] >= 4.0) & (df['af'] < 5), "PA"] = "PA_4"
    df.loc[(df['af'] == 5.0), "PA"] = "PA_5"'''


    return df

def redefine_alcohol(df):
    # Redefine categories of alcohol consumption. 
    # Option 1: low, high
    # Option 2: no, low, high

    df.loc[df['consumo_alcohol'] == 'no', ["consumo_alcohol"]] = "no"
    df.loc[df['consumo_alcohol'] == 'esporadico',["consumo_alcohol"]] = "low"
    df.loc[df['consumo_alcohol'] == 'fin de semana', ["consumo_alcohol"]] = "low"
    df.loc[df['consumo_alcohol'] == 'exconsumidor', ["consumo_alcohol"]] = "high" # Some ambiguity here, could be low. If there is time, we should try both options.
    df.loc[df['consumo_alcohol'] == 'habitual',["consumo_alcohol"]] = "high"
    df.loc[df['consumo_alcohol'] == 'dependiente', ["consumo_alcohol"]] = "high"

    return df

def remove_patients_other_cancers(df, cancer_type):
    # Remove
    cancer_list = [
             'cancer_cabeza_y_cuello', 'cancer_digestivo',
       'cancer_colorrectal', 'cancer_higado', 'cancer_panc/vb', 'cancer_otros',
       'cancer_hematologico', 'cancer_pulmon', 'cancer_hueso/pb',
       'cancer_piel', 'cancer_mama', 'cancer_ovario', 'cancer_prostata',
       'cancer_vejiga/vu', 'cancer_rinon', 'cancer_cerebral',
       'cancer_endocrino'
    ]

    cancer_list.remove(cancer_type)

    for cancer in cancer_list:
        df = df[df[cancer] == False].copy()
    
    df.reset_index(inplace=True, drop = True)
    return df

def rename_vars(df,cancer_type, cancer_renamed):

    df.loc[df['sexo'] == 'hombre', ["sexo"]] = "M"
    df.loc[df['sexo'] == 'mujer',["sexo"]] = "W"

    df.loc[df['duracion_sueño'] == '<6h', ["duracion_sueño"]] = "SD_1_short"
    df.loc[df['duracion_sueño'] == '6-9h',["duracion_sueño"]] = "SD_2_normal"
    df.loc[df['duracion_sueño'] == '>9h', ["duracion_sueño"]] = "SD_3_excessive"

    df.loc[df['fumador'] == 'no fumador', ["fumador"]] = "sm_1_not_smoker"
    df.loc[df['fumador'] == 'ex-fumador', ["fumador"]] = "sm_3_ex_smoker"
    df.loc[df['fumador'] == 'fumador', ["fumador"]] = "sm_2_smoker"


    df.rename(columns = 
                       {"edad5" : "Age",
                        "sexo" : "Sex", 
                        "imc4" : "BMI",
                        "consumo_alcohol": "Alcohol",
                        "fumador": "Smoking",
                        "af_2": "PA",
                        "depresion": "Depression",
                        "ansiedad": "Anxiety",
                        "diabetes1": "Diabetes",
                        "hipertension1": "Hypertension",
                        "hipercolesterolemia1": "Hyperchol", 
                        "medicacion": "Medication",
                        "duracion_sueño": "SD",
                        "condicion_socioeconomica_media_3": "SES",
                        cancer_type: cancer_renamed
                           
                       }, inplace = True)
    
    df = df[(df["Age"] != "age_1_very_young") & (df["Age"] != "age_6_elderly")].copy()

    df.dropna(subset=["Sex"], inplace = True)
    df.reset_index(inplace = True, drop = True)

    return df

def data_clean_discrete(df, cancer_type = "cancer_colorrectal", cancer_renamed = "CRC", selected_year = 2012, logger = None):

    logger.info("Dropping duplicates")
    df = df.drop_duplicates(subset = ["fpi", "fecha_reco"], ignore_index = True).copy()
    
    logger.info("Dropping outliers")
    df = df[df["outlier"] == False].reset_index(drop = True).copy()
    
    logger.info(f"Selecting year: {selected_year}")
    df = df[df["año_reco"] == selected_year].reset_index(drop = True).copy()
    
    # Impute missing values.
    logger.info("Imputing missing values")
    df = impute_missing_values_missForest(df)

    # Redefine variables. Discretize continuous variables.
    logger.info("Redefining variables")
    df = redefine_imc(df)
    df = redefine_age(df)
    df = redefine_medical_cond(df)
    

    # For now we will drop the missing values.
    '''df.dropna(subset=["sexo", "af","fumador", "consumo_alcohol",
                    "duracion_sueño", "condicion_socioeconomica_media"],
                    inplace = True)
    df.reset_index(inplace=True, drop = True)'''

    # Should we remove patients with other cancers?
    logger.info("Removing patients with other cancers")
    logger.info(f"Before: {len(df)}")
    df = remove_patients_other_cancers(df, cancer_type)
    logger.info(f"After:  {len(df)}")

    # Filter by unique patient
    logger.info("Filtering by unique patient")
    logger.info(f"Before: {len(df)}")
    df = unique_patient(df, cancer_type)
    logger.info(f"After:  {len(df)}")

    # Redefine variables
    logger.info("Redefining variables")
    df = redefine_ses(df)
    df = redefine_alcohol(df)
    df = redefine_pa(df)
    df = rename_vars(df,cancer_type, cancer_renamed)

    
    logger.info("Data cleaning finished")
    logger.info(f"Final shape: {df.shape}")
    return df









