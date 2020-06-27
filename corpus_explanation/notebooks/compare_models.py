import pandas as pd
import os
import re
import argparse
from datetime import datetime
DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'
def load_explanations(path):
    print(f"Loading from {path}")
    df = pd.read_csv(path, header=0, names=["id","review","explanation","contribution","frequency","confidence_score","prediction","label","raw_pred"], encoding="utf-8")
    #df["contribution"] = df["contribution"].apply(lambda c: float(str(c).split(":")[0]))
    df["contribution"] = df["contribution"].astype('float64')
    df["frequency"] = df["explanation"].apply(lambda f: list(re.findall(r'(\d+)', str(f)))).apply(lambda x: x[0] if x else None)
    df["confidence_score"] = df["confidence_score"].astype('float64')
    df["prediction"] = df["prediction"].astype('float64')
    df["label"] = df["label"].astype('float64')
    df["raw_pred"] = df["raw_pred"].astype('float64')
    df["explanation"] = df["explanation"].apply(lambda s: s.split("'")[1])
    return df


explanations_m1 = "experiments/soa-dicts/bilstm_mlp_improve_30-30_l20.01_dr0.8_lr0.01_soa_vlstm2-64-0.5_pretrained_rake-4-600-dnn30-1-30-decay0.0-L2-dr0.8-eval1-rake-inst-4-600-improveloss_mean-alpha0.7-c-e10-2020-06-24_15-27-01/explanations/descending_contribution.txt" # small
explanations_m2 = "experiments/soa-dicts/bilstm_mlp_improve_30-30_l20.1_dr0.7_lr0.001_soa_vlstm2-256-0.5_pretrained_rake-4-600-e20-dnn30-1-30-decay0.0-L2-dr0.7-eval1-rake-inst-4-600-improveloss_mean-alpha0.7-c-e20-2020-06-23_15-15-42/explanations/descending_contribution.txt"

start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

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


e1 = load_explanations(args.p1)#.sort_values(by="id")
e2 = load_explanations(args.p2)#.sort_values(by="id")
e2=e2.drop(["review"], axis=1)
e2.columns = [col +"_M2" if col !="id" else col for col in e2.columns]

e1_correct = e1[e1["prediction"]==e1["label"]]
e1_incorrect = e1[e1["prediction"]!=e1["label"]]
print("E1: correct/incorrect")
print(e1_correct.shape)
print(e1_incorrect.shape)


e2_correct = e2[e2["prediction_M2"]==e2["label_M2"]]
e2_incorrect = e2[e2["prediction_M2"]!=e2["label_M2"]]
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
res.sort_values(["contribution", "contribution_M2"], ascending=[False, False], inplace=True)
print(res.head())
print(res.shape)
res.to_csv(args.o)
