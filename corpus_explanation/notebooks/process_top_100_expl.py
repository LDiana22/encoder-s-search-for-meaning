import pandas as pd
import os
import re
import argparse
from datetime import datetime
DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'


def load_explanations(path, increasing, no_examples):
	print(f"Loading from {path}")
	if increasing:
		df = pd.read_csv(path, sep=",",header=0, names=["id","review","explanation","contribution","frequency","confidence_score","prediction","label"])
		df = df.tail(no_examples)
	else:
		df = pd.read_csv(path, sep=",",header=0, names=["id","review","explanation","contribution","frequency","confidence_score","prediction","label"], nrows=no_examples)

	df["explanation"] = df["explanation"].apply(lambda x: re.findall(r"('.*')", x)[0])
	# print(df.head())
	# print(len(pd.unique(df["explanation"])))
	# print(df["explanation"].value_counts())
	# print(df.shape)
	return df


start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

parser = argparse.ArgumentParser(description='Config params.')

# experiments\independent\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\explanations\e_test-7_2020-05-24_03-16-15.txt
#  experiments\independent\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\explanations\e_test-7_2020-05-24_03-16-15_fix.txt
parser.add_argument('-f', metavar='file name', type=str,
                    help='expl file')
parser.add_argument('--fix', metavar='fix the explanation file format', type=bool, default=False)

parser.add_argument('-p', metavar='path', type=str,
                    help='expl file')
parser.add_argument('-f1', metavar='file name', type=str,
                    help='expl file')
parser.add_argument('-f2', metavar='file name', type=str,
                    help='expl file')
args = parser.parse_args()

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def polarity(phrase):
	if analyzer.polarity_scores(phrase)['neg']>0.5:
		return 1
	if analyzer.polarity_scores(phrase)['pos']>0.5:
		return 0
	return None

def load_new_explanations(path):
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
  df["e_polarity"] = df["explanation"].apply(polarity)
  return df


def merge_and_shuffle(path1, path2, output):
	#path = "experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improveloss_mean-alpha0.7-c-e40-2020-05-26_01-06-16\\explanations\\descending_contribution_correct.txt"
	#path = "experiments\\soa-dicts\\rc_bilstm_mlp_improve_30-30_l20.01_dr0.8_lr0.01_soa_vlstm2-64-0.3_pretrained_rake-polarity-4-60-dnn30-1-30-decay0.0-L2-dr0.8-eval1-rake-polarity-4-600-improveloss_mean-alpha0.7-c-e10-2020-06-29_01-40-17\\explanations\\descending_contribution_correct.txt"
  df1=load_new_explanations(path1)
	#path = "experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improveloss_mean-alpha0.7-c-e40-2020-05-26_01-06-16\\explanations\\descending_contribution_incorrect.txt"
	#path = "experiments\\soa-dicts\\rc_bilstm_mlp_improve_30-30_l20.01_dr0.8_lr0.01_soa_vlstm2-64-0.3_pretrained_rake-polarity-4-60-dnn30-1-30-decay0.0-L2-dr0.8-eval1-rake-polarity-4-600-improveloss_mean-alpha0.7-c-e10-2020-06-29_01-40-17\\explanations\\descending_contribution_incorrect.txt"
  df2=load_new_explanations(path2)
	df = pd.concat([df1.head(50),df2.head(50)])
	df = df.sample(frac=1).reset_index(drop=True)

	# df.to_csv("experiments\\soa-dicts\\rc_bilstm_mlp_improve_30-30_l20.01_dr0.8_lr0.01_soa_vlstm2-64-0.3_pretrained_rake-polarity-4-60-dnn30-1-30-decay0.0-L2-dr0.8-eval1-rake-polarity-4-600-improveloss_mean-alpha0.7-c-e10-2020-06-29_01-40-17\\explanations\\100_posneg_shuffled_contributions.csv")
	df.to_csv(output)
	return df

path1 = os.path.join(args.p, args.f1)
path2 = os.path.join(args.p, args.f2)
result = os.path.join(args.p, "merged_100c_i.csv")
merged = merge_and_shuffle(path1, path2, result)

print(merged.shape)
print("Count polarity=label")
print(merged[merged["e_polarity"]==merged["label"]].count()["id"])
print("Count polarity=prediction")
print(merged[merged["e_polarity"]==merged["prediction"]].count()["id"])

# path= "experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improveloss_mean-alpha0.7-c-e40-2020-05-26_01-06-16\\explanations\\100_shuffled_contributions.csv"
# print(f"Loading from {path}")
# df = pd.read_csv(path, sep=",",header=0, names=["id","review","explanation","contribution","frequency","confidence_score","prediction","label"])
# df["Explanation_Sentiment"] = df["explanation"].apply(polarity)
# path = "experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improveloss_mean-alpha0.7-c-e40-2020-05-26_01-06-16\\explanations"
# print(os.path.exists(path))
# df.to_csv("experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improveloss_mean-alpha0.7-c-e40-2020-05-26_01-06-16\\explanations\\result.csv")

