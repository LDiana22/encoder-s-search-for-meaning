import pandas as pd
import os
import re
import argparse
from datetime import datetime
DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'
def load_explanations(path):
    print(f"Loading from {path}")
    df = pd.read_csv(path, sep="~", header=0, names=["review", "explanation", "contribution"])
    #df["contribution"] = df["contribution"].apply(lambda c: float(str(c).split(":")[0]))
    df["contribution"] = df["contribution"].apply(lambda c: float(c))
    df["frequency"] = df["explanation"].apply(lambda f: list(re.findall(r'(\d+)', str(f)))).apply(lambda x: x[0] if x else None)
    df["confidence_score"] = df["explanation"].apply(lambda f: re.findall(r'\d+\.\d+', str(f))).apply(lambda x: float(x[0]) if x else None)
    df["prediction"] = df["explanation"].apply(lambda f: re.findall(r'\d+\.\d+',str(f))).apply(lambda x: float(x[1]) if len(x)>1 else None)
    df["label"] = df["explanation"].apply(lambda x: re.findall(r'\d+\.\d+', str(x))).apply(lambda x: float(x[2]) if len(x)>2 else None)
    df["raw_pred"] = df["explanation"].apply(lambda x: re.findall(r'\d+\.\d+', str(x))).apply(lambda x: float(x[3]) if len(x)>3 else None)
    return df


explanations_m1 = "experiments/soa-dicts/bilstm_mlp_improve_30-30_l20.01_dr0.8_lr0.01_soa_vlstm2-64-0.5_pretrained_rake-4-600-dnn30-1-30-decay0.0-L2-dr0.8-eval1-rake-inst-4-600-improveloss_mean-alpha0.7-c-e10-2020-06-24_15-27-01/explanations/all_instances_metrics.txt" # small
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


e1 = load_explanations(args.p1)
e2 = load_explanations(args.p2)

print(e1.head())
print(e2.head())
print(e1.head().cat(e2.head(), join="merge"))
print(e1.head().cat(e2.head(), join="merge"))
print(e1.head().cat(e2.head(), join="left"))
print(e1.head().cat(e2.head(), join="right"))