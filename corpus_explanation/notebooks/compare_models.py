import pandas as pd
import os
import re
import argparse
from datetime import datetime
DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'
def load_explanations(path):
    print(f"Loading from {path}")
    df = pd.read_csv(path, header=0, names=["id","review","explanation","contribution","frequency","confidence_score","prediction","label","raw_pred", "vanilla_prediction", "c"], encoding="utf-8")
    #df["contribution"] = df["contribution"].apply(lambda c: float(str(c).split(":")[0]))
    df["contribution"] = df["contribution"].astype('float64')
    df["frequency"] = df["explanation"].apply(lambda f: list(re.findall(r'(\d+)', str(f)))).apply(lambda x: x[0] if x else None)
    df["confidence_score"] = df["confidence_score"].astype('float64')
    df["prediction"] = df["prediction"].astype('float64')
    df["label"] = df["label"].astype('float64')
    df["raw_pred"] = df["raw_pred"].astype('float64')
    df["explanation"] = df["explanation"].apply(lambda s: s.split("'")[1])
    return df

def print_model_metrics(e1,e2):
    print("M1(large)")
    print("Highest contribution explanations")
    e1.sort_values(["contribution"], ascending=[False], inplace=True)
    # print("Most frequent explanations")
    # print(e1[e1["contribution"]>0]["explanation"].value_counts()[:20])
    # print(e2[e2["contribution_M2"]>0]["explanation_M2"].value_counts()[:20])

    # print(e1[(e1["label"]==e1["prediction"]) & (e1["label"]==0)].head(100)["explanation"].value_counts()[:20])
    # print(e2[(e2["label_M2"]==e2["prediction_M2"]) & (e2["label_M2"]==0)].head(100)["explanation_M2"].value_counts()[:20])
    
    # print(e1[(e1["label"]==e1["prediction"]) & (e1["label"]==1)].head(100)["explanation"].value_counts()[:20])
    # print(e2[(e2["label_M2"]==e2["prediction_M2"]) & (e2["label_M2"]==1)].head(100)["explanation_M2"].value_counts()[:20])
        
    print(e1[(e1["label"]!=e1["prediction"]) & (e1["label"]==0)].head(100)["explanation"].value_counts()[:20])
    print(e2[(e2["label_M2"]!=e2["prediction_M2"]) & (e2["label_M2"]==0)].head(100)["explanation_M2"].value_counts()[:20])
    
    return
    # print(e1[e1["label"]==e1["prediction"]].head(100)["explanation"].value_counts())
    # print("E1 incorrect")
    # print(e1[e1["label"]!=e1["prediction"]].head(100)["explanation"].value_counts())
    top = e1.head(50)
    print("Out of top 50, correctly classified: ")
    print(top[top["label"]==top["prediction"]].count())
    print(top[top["label"]==top["prediction"]]["explanation"].value_counts())

# explanations_m1 = "experiments/soa-dicts/bilstm_mlp_improve_30-30_l20.1_dr0.7_lr0.001_soa_vlstm2-256-0.5_pretrained_rake-4-600-e20-dnn30-1-30-decay0.0-L2-dr0.7-eval1-rake-inst-4-600-improveloss_mean-alpha0.7-c-e20-2020-06-23_15-15-42/explanations/descending_contribution.txt"
#explanations_m1 = "experiments/soa-dicts/bilstm_mlp_improve_30-30_l20.1_dr0.7_lr0.001_soa_vlstm2-256-0.5_rake-inst-distr100-4-300-dnn30-1-30-decay0.0-L2-dr0.7-eval1-rake-inst-4-600-improveloss_mean-alpha0.7-c-e10-2020-07-10_08-52-54/explanations/new/descending_contribution.txt"
#explanations_m2 = "experiments/soa-dicts/bilstm_mlp_improve_30-30_l20.01_dr0.8_lr0.01_soa_vlstm2-64-0.5_pretrained_rake-4-600-dnn30-1-30-decay0.0-L2-dr0.8-eval1-rake-inst-4-600-improveloss_mean-alpha0.7-c-e10-2020-06-24_15-27-01/explanations/descending_contribution.txt" # small
#explanations_m2 = "experiments/soa-dicts/bilstm_mlp_improve_30-30_l20.01_dr0.8_lr0.01_soa_vlstm2-64-0.5_pretrained_rake-4-600-dnn30-1-30-decay0.0-L2-dr0.8-eval1-rake-inst-4-600-improveloss_mean-alpha0.7-c-e10-2020-06-24_15-27-01/explanations/new/descending_contribution.txt"
# explanations_m1 = "experiments/soa-dicts/bilstm_mlp_improve_30-30_l20.1_dr0.7_lr0.001_soa_vlstm2-256-0.5_rake-inst-distr100-4-300-dnn30-1-30-decay0.0-L2-dr0.7-eval1-rake-inst-4-600-improveloss_mean-alpha0.7-c-e10-2020-07-10_08-52-54/explanations/new/descending_contribution.txt"
# explanations_m2 = "experiments/soa-dicts/bilstm_mlp_improve_30-30_l20.01_dr0.8_lr0.01_soa_vlstm2-64-0.5_pretrained_rake-4-600-dnn30-1-30-decay0.0-L2-dr0.8-eval1-rake-inst-4-600-improveloss_mean-alpha0.7-c-e10-2020-06-24_15-27-01/explanations/descending_contribution.txt" # small

explanations_m1 = "experiments/soa-dicts/bilstm_mlp_improve_30-30_l20.01_dr0.7_lr0.001_soa_vlstm2-256-0.5_rake-inst-distr100-4-600-dnn30-1-30-decay0.0-L2-dr0.7-eval1-rake-inst-4-600-improveloss_mean-alpha0.7-c-e10-2020-09-02_12-07-23/explanations/new/descending_contribution.txt"
explanations_m2 = "experiments/soa-dicts/bilstm_mlp_improve_30-30_l20.001_dr0.7_lr0.01_soa_vlstm2-64-0.3_rake-inst-distr100-4-600-dnn30-1-30-decay0.0-L2-dr0.7-eval1-rake-inst-4-600-improveloss_mean-alpha0.7-c-e10-2020-09-03_21-28-16/explanations/new-raw/descending_contribution.txt"




start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

# python compare_models.py -p1 P:\uottawa\Thesis\text_nn-master\corpus_explanation\notebooks\out\large\descending_contribution.txt -p2 P:\uottawa\Thesis\text_nn-master\corpus_explanation\notebooks\out\small\descending_contribution.txt -o P:\uottawa\Thesis\text_nn-master\corpus_explanation\notebooks\out\sample_with_polarity.csv
parser = argparse.ArgumentParser(description='Config params.')

# experiments\independent\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\explanations\e_test-7_2020-05-24_03-16-15.txt
#  experiments\independent\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\explanations\e_test-7_2020-05-24_03-16-15_fix.txt
parser.add_argument('-p1', metavar='path', type=str, default=explanations_m1,
                    help='expl path')
parser.add_argument('-p2', metavar='path', type=str, default=explanations_m2,
                    help='expl path')
parser.add_argument('-o', metavar='output path', type=str,
                    help='expl file')
args = parser.parse_args()


e1 = load_explanations(args.p1)#.sort_values(by="id") #large (VLSTM+)
e2 = load_explanations(args.p2)#.sort_values(by="id") #small
e2=e2.drop(["review"], axis=1)
e2.columns = [col +"_M2" if col !="id" else col for col in e2.columns]

print_model_metrics(e1,e2)
#exit()

e1_correct = e1[(e1["prediction"]==e1["label"]) & (e1["contribution"]>0)].sort_values(["contribution"], ascending=[False])
e1_incorrect = e1[(e1["prediction"]!=e1["label"]) & (e1["contribution"]>0)]

print("E1: correct/incorrect")
print(e1_correct.shape)
print(e1_incorrect.shape)


e2_correct = e2[(e2["prediction_M2"]==e2["label_M2"]) & (e2["contribution_M2"]>0)].sort_values(["contribution_M2"], ascending=[False])
e2_incorrect = e2[(e2["prediction_M2"]!=e2["label_M2"]) & (e2["contribution_M2"]>0)]
print("E2: correct/incorrect")
print(e2_correct.shape)
print(e2_incorrect.shape)

#import ipdb
#ipdb.set_trace(context=10)

res1 = pd.merge(left=e1_correct, right=e2_incorrect, left_on="id", right_on="id")
res2 = pd.merge(left=e1_incorrect, right=e2_correct, left_on="id", right_on="id")
print(f"E1_C/E2_INC: {res1.shape}")
print(f"E1_INC/E2_C: {res2.shape}")

#ies2 = pd.concat(columns=e1_correct.columns+e2_correct.columns, join="outer")
res = pd.concat([res1,res2]) 
print(f"Together: {res.shape}")

#contribution was not enough
print("Inccorect predictions where the contribution was not enough to change the prediction")
print(e1_incorrect[e1_incorrect.apply(lambda entry: round(float(entry["contribution"]) + float(entry["raw_pred"])) == round(float(entry["raw_pred"])), axis=1)].shape)
print(e2_incorrect[e2_incorrect.apply(lambda entry: round(float(entry["contribution_M2"]) + float(entry["raw_pred_M2"])) == round(float(entry["raw_pred_M2"])), axis=1)].shape)
print("Correct predictions where the contribution changed an incorrect prediction")
print(e1_correct[e1_correct.apply(lambda entry: round(float(entry["contribution"]) + float(entry["raw_pred"])) != round(float(entry["raw_pred"])), axis=1)].shape)
print(e2_correct[e2_correct.apply(lambda entry: round(float(entry["contribution_M2"]) + float(entry["raw_pred_M2"])) != round(float(entry["raw_pred_M2"])), axis=1)].shape)

print("Other")
print(res1[res1.apply(lambda entry: round(float(entry["contribution"]) + float(entry["raw_pred"])) != round(float(entry["raw_pred"])), axis=1)].shape)
print(res2[res2.apply(lambda entry: round(float(entry["contribution_M2"]) + float(entry["raw_pred_M2"])) != round(float(entry["raw_pred_M2"])), axis=1)].shape)


print("Top 100 best contributions from M1 and M2")
result = pd.merge(left=e1_correct, right=e2_correct, left_on="id", right_on="id")
result = result[(result["contribution_M2"]>0)&(result["contribution"]>0)]
print("merged correct and positive contribution for both")
print(result.shape)
print(result[result["contribution"]>0].count()["contribution"])
result = pd.concat([result.sort_values(["contribution"], ascending=[False]).head(50), result.sort_values(["contribution_M2"], ascending=[False]).head(50)])
print(result.shape)
print(result)
print(result[result["contribution"]>0].count()["contribution"])
result = result.sample(frac=1).reset_index(drop=True)
result.to_csv(args.o)
# res.sort_values(["contribution", "contribution_M2"], ascending=[False, False], inplace=True)
# print(res.head())
# print(res.shape)


def sample_c1_2(res1,res2):
    subset_r1_C1_gt_C2 = res1[res1["contribution"]>res1["contribution_M2"]].sort_values(["contribution_M2"], ascending=[False])
    subset_r1_C2_gt_C1 = res1[res1["contribution"]<res1["contribution_M2"]].sort_values(["contribution"], ascending=[False])

    subset_r2_C1_gt_C2 = res2[res2["contribution"]>res2["contribution_M2"]].sort_values(["contribution_M2"], ascending=[False])
    subset_r2_C2_gt_C1 = res2[res2["contribution"]<res2["contribution_M2"]].sort_values(["contribution"], ascending=[False])
    result = pd.concat([subset_r1_C1_gt_C2.head(25), subset_r1_C2_gt_C1.head(25),subset_r2_C1_gt_C2.head(25), subset_r2_C2_gt_C1.head(25)])
    return result.sample(frac=1).reset_index(drop=True)

res = sample_c1_2(res1,res2)
print(res.shape)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def polarity(phrase):
    if analyzer.polarity_scores(phrase)['neg']>0.5:
        return 1
    if analyzer.polarity_scores(phrase)['pos']>0.5:
        return 0
    return None

res["E1_polarity"] = res["explanation"].apply(polarity)

res["E2_polarity"] = res["explanation_M2"].apply(polarity)

print(f"Polarity coherence with label M1: {res[res['E1_polarity']==res['label']].count()}")
print(f"Polarity coherence with label M2: {res[res['E2_polarity']==res['label']].count()}")

print(f"Polarity coherence with prediction M1: {res[res['E1_polarity']==res['prediction']].count()}")
print(f"Polarity coherence with prediction M2: {res[res['E2_polarity']==res['prediction_M2']].count()}")

res.to_csv(args.o)
