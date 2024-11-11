data_cleaning = {
    "tipo_de_cancer": "cancer_piel",
}

inputs = {
    "target": "cancer_piel",
    "calculate_interval": False,
    "n_random_trials": 50,

}


structure = {

    "black_list" : [
                    ('Age', 'Sex'),
                    ('BMI', 'Sex'),
                    ('PA', 'Sex'),
                    ('Alcohol', 'Sex'),
                    ('Smoking', 'Sex'),
                    ('SD', 'Sex'),
                    ('Diabetes', 'Sex'),
                    ('Hypertension', 'Sex'),
                    ('Hyperchol.', 'Sex'),
                    ('Depression', 'Sex'),
                    ('Anxiety', 'Sex'),
                    ('CRC', 'Sex'),
                    ('SES', 'Sex'),            

                    ('Sex', 'Age'),
                    ('BMI', 'Age'),
                    ('PA', 'Age'),
                    ('Alcohol', 'Age'),
                    ('Smoking', 'Age'),
                    ('SD', 'Age'),
                    ('Diabetes', 'Age'),
                    ('Hypertension', 'Age'),
                    ('Hyperchol.', 'Age'),
                    ('Depression', 'Age'),
                    ('Anxiety', 'Age'),
                    ('CRC', 'Age'),
                    ('SES', 'Age'),

                    ('BMI', 'SES'),
                    ('PA', 'SES'),
                    ('Alcohol', 'SES'),
                    ('Smoking', 'SES'),
                    ('SD', 'SES'),
                    ('Diabetes', 'SES'),
                    ('Hypertension', 'SES'),
                    ('Hyperchol.', 'SES'),
                    ('Depression', 'SES'),
                    ('Anxiety', 'SES'),
                    (inputs["target"], 'SES'),
        ], 
    "fixed_edges": [
                    ('Sex', 'Anxiety'),
                    ('Sex', 'Depression'),

                    ('Age', 'Diabetes'),
                    ('Age', 'SD'), 
                    ('Age', 'Smoking'), 
                    ('Age', 'Hypertension'), 
                    ('Age', 'BMI'), 
                    
                    ('BMI', 'Diabetes'), 
                    ('BMI', 'Hyperchol.'), 
                    ('BMI', 'Hypertension'), 

                    ('Alcohol', 'Hypertension'),
                    ('Alcohol', 'Hyperchol.'),

                    ('Smoking', 'Hyperchol.'),
                    ('Smoking', 'Hypertension'), 

                    ('PA', 'Diabetes'), 
                    ('PA', 'Hyperchol.'), 
                    ('PA', 'Hypertension'), 
                    ('PA', 'BMI'),

                    ('Diabetes', 'Hypertension'),

                    ('SD', 'PA'),
                    ('SD', 'Anxiety'),
                    ('Anxiety', 'Hypertension'),

                    ('SES', 'PA'),
        ]
}


node_color = {'Age': 0.3,
                'Sex': 0.3,
                'BMI': 0.1,
                'Alcohol': 0.1,
                'Smoking': 0.1,
                'PA': 0.1,
                'Depression': 0.1,
                'Anxiety': 0.1,
                'Diabetes': 0.2,
                'Hypertension': 0.2,
                'Hyperchol.': 0.2,
                'SD': 0.1,
                'SES': 0.3,
                inputs["target"]: 0.4}


pointwise_risk_mapping = {
    "col_var": "Age",
    "row_var": "BMI"
}

interval_risk_mapping = {
    "col_var": "Age",
    "row_var": "BMI"
}

interval_path = {'path': "prueba22nov/"}