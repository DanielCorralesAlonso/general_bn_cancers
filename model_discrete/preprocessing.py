


def preprocessing(df, cancer_type="CRC"):
    try:
        df.drop(columns = ["Medication", "año_reco", "fpi", "Unnamed: 0"], inplace = True)
    except:
        df = df
    
    try:
        df = df[(df["Age"] != "1_very_young") & (df["Age"] != "6_elderly")].copy()
    except:
        df = df

    try:
        df = df[['Age', 'Sex', 'BMI', 'Alcohol',
       'Smoking', 'PA', 'Depression', 'Anxiety', 'Diabetes', 'Hypertension',
       'Hyperchol', 'SD', 'SES', cancer_type]].copy()
    except:
        df = df

    return df