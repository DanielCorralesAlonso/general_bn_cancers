import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, PrecisionRecallDisplay, fbeta_score, brier_score_loss
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean

from sklearn.calibration import calibration_curve, CalibrationDisplay

from pgmpy.inference import VariableElimination

from query2df import query2df

import os
import os.path as mkdir

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pdb
import yaml
import os

dir = os.getcwd()
with open(f'{dir}\configs\config_CRC.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

def evaluation_classification(df_test, model_bn, test_var = "CRC"):
    

    model_infer = VariableElimination(model_bn)

    g_means_iter = []
    sensitivity_iter = []
    specificity_iter = []
    brier_score = []

    y_pred = []
    y_prob_pred = []


    '''with ProcessPoolExecutor(max_workers=cfg["inputs"]['max_workers']//2) as executor:
        futures = [executor.submit(run_iteration_y_pred, i, df_test.iloc[i] , model_infer, test_var) for i in range(df_test.shape[0])]
        all_results = []
'''
    for i in tqdm(range(df_test.shape[0]), desc="Processing samples"):
        sample = df_test.iloc[i]
        sample = sample.drop(labels = [test_var])
        sample_dict = sample.to_dict()
        q_sample = model_infer.query(variables=[test_var], evidence = sample_dict)

        y_prob_pred.append(query2df(q_sample, verbose = 0)["p"][1])

    fpr, tpr, thresholds = roc_curve(list(df_test[test_var]*1), y_prob_pred)
    # calculate the g-mean for each threshold
    gmeans = gmean( [tpr, 1-fpr], axis = 0, weights=[[1],[1]])
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    g_means_iter.append(gmeans[ix])


    y_pred = np.where(y_prob_pred > thresholds[ix], 1, 0)

    beta = 2
    fbeta = fbeta_score(list(df_test[test_var]*1), y_pred, beta=beta)
    print(f"F_{beta} score =", fbeta)


    conf_mat = confusion_matrix(list(df_test[test_var]), y_pred)
    disp = ConfusionMatrixDisplay(conf_mat, display_labels=np.array(sorted(set(df_test[test_var]*1))) )
    disp.plot()
    plt.savefig(f"images/{test_var}/{test_var}_confusion_matrix.png")
    plt.close()

    brier_score.append(brier_score_loss(list(df_test[test_var]*1), y_prob_pred))

    print("Brier loss:", brier_score_loss(list(df_test[test_var]*1), y_prob_pred))


    prob_true, prob_pred = calibration_curve(list(df_test[test_var]*1), y_prob_pred, n_bins = 10, strategy="quantile")
    disp = CalibrationDisplay(prob_true, prob_pred, y_prob_pred)
    disp.plot(name = test_var)
    plt.savefig(f"images/{test_var}/{test_var}_calibration_plot.png")    

    plt.xlim([0, max(max(prob_pred), max(prob_true))])  #CRC: 0.005, Diabetes: 0.25
    plt.ylim([0, max(max(prob_pred), max(prob_true))])

    dashed =  len(df_test[df_test[test_var] == True]) / len(df_test)
    plt.axvline(x=dashed, linestyle = '--', color='gray') 

    # Show the modified plot
    plt.title(f"Calibration plot for {test_var}")
    plt.savefig(f"images/{test_var}/{test_var}_calibration_plot_zoomed.png")
    plt.close()

    report = classification_report(list(df_test[test_var]*1), y_pred, output_dict=True)

    print(classification_report(list(df_test[test_var]*1), y_pred))

    sensitivity_iter.append(report["1"]["recall"])
    specificity_iter.append(report["0"]["recall"])
        

    print("\nAverage G-Mean: ", np.mean(g_means_iter), '+/- ', np.std(g_means_iter))
    print("\nAverage Sensitivity: ", np.mean(sensitivity_iter), '+/- ', np.std(sensitivity_iter))
    print("\nAverage Specificity: ", np.mean(specificity_iter), '+/- ', np.std(specificity_iter))

    return y_prob_pred



def run_iteration_y_pred(i, sample, model_infer, test_var):
    sample = sample.drop(labels = [test_var])
    sample_dict = sample.to_dict() 
    q_sample = model_infer.query(variables=[test_var], evidence = sample_dict)

    result = (query2df(q_sample, verbose = 0)["p"][1]).round(7)

    print(result)

    return result 