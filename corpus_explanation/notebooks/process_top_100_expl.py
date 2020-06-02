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

args = parser.parse_args()

def merge_and_shuffle():
	path = "experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improveloss_mean-alpha0.7-c-e40-2020-05-26_01-06-16\\explanations\\descending_contribution_correct.txt"
	df1=load_explanations(path)
	path = "experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improveloss_mean-alpha0.7-c-e40-2020-05-26_01-06-16\\explanations\\descending_contribution_incorrect.txt"
	df2=load_explanations(path)
	df = pd.concat([df1,df2])
	df = df.sample(frac=1).reset_index(drop=True)

	df.to_csv("experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improveloss_mean-alpha0.7-c-e40-2020-05-26_01-06-16\\explanations\\100_posneg_influences_shuffled_contributions.csv")

merge_and_shuffle()

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def polarity(phrase):
	if analyzer.polarity_scores(phrase)['neg']>0.5:
		return 1
	if analyzer.polarity_scores(phrase)['pos']>0.5:
		return 0
	return None

path= "experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improveloss_mean-alpha0.7-c-e40-2020-05-26_01-06-16\\explanations\\100_shuffled_contributions.csv"
print(f"Loading from {path}")
df = pd.read_csv(path, sep=",",header=0, names=["id","review","explanation","contribution","frequency","confidence_score","prediction","label"])
print(df.shape)
df["Explanation_Sentiment"] = df["explanation"].apply(polarity)
path = "experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improveloss_mean-alpha0.7-c-e40-2020-05-26_01-06-16\\explanations"
# print(os.path.exists(path))
df.to_csv("experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improveloss_mean-alpha0.7-c-e40-2020-05-26_01-06-16\\explanations\\result.csv")
