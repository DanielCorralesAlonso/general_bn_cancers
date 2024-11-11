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
                    ('CRC', 'SES'),
        ], 

    "fixed_edges" : [
                    ('Sex', 'Anxiety'),
                    ('Sex', 'Depression'),
                    ('Sex', 'CRC'),

                    ('Age', 'CRC'),
                    ('Age', 'Diabetes'),
                    ('Age', 'SD'), 
                    ('Age', 'Smoking'), 
                    ('Age', 'Hypertension'), 
                    ('Age', 'BMI'), 
                    
                    ('BMI', 'Diabetes'), 
                    ('BMI', 'Hyperchol.'), 
                    ('BMI', 'Hypertension'), 

                    ('Alcohol', 'CRC'),
                    ('Alcohol', 'Hypertension'),
                    ('Alcohol', 'Hyperchol.'),

                    ('Smoking', 'CRC'), 
                    ('Smoking', 'Hyperchol.'),
                    ('Smoking', 'Hypertension'), 

                    ('PA', 'Diabetes'), 
                    ('PA', 'Hyperchol.'), 
                    ('PA', 'Hypertension'), 
                    ('PA', 'BMI'),

                    ('Diabetes', 'CRC'), 
                    ('Diabetes', 'Hypertension'),

                    ('Hypertension', 'CRC'), 

                    ('Hyperchol.', 'CRC'),

                    ('SD', 'PA'),
                    ('SD', 'Anxiety'),
                    ('Anxiety', 'Hypertension'),

                    ('SES', 'PA'),
        ]
}