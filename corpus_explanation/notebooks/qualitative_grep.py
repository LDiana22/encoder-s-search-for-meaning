import pandas as pd
import re

def fix_file(path):
	new_path = path.split(".")[0]+"_fix"+path.split(".")[1]
	print(f"New path {new_path}")
	with open(new_path, "w") as f:
		with open(path, "r") as g:
			line = g.readline()
			count = 0
			while line:
				count += 1
				if len(line.split("~")) > 3:
					sublines = re.split('C: (\+|-)?\d\.\d+', line)
					f.write(sublines[0] + re.findall(r'(C: (\+|-)?\d\.\d+)', line)[0][0])
					f.write("\n")
					f.write(sublines[2] + re.findall(r'(C: (\+|-)?\d\.\d+)', line)[1][0])
				else:
					f.write(line)
				f.write("\n")
				line = g.readline()	
	return new_path

def load_explanations(path):
	print(f"Loading from {path}")
	df = pd.read_csv(path, sep="~", header=0, names=["review", "explanation", "contribution"])
	df["contribution"] = df["contribution"].apply(lambda c: float(c.split(":")[1]))
	df["frequency"] = df["explanation"].apply(lambda f: re.findall(r'\d+',f)[0][0])
	df["confidence_score"] = df["explanation"].apply(lambda f: float(re.findall(r'\d+\.\d+',f)[0][0]))
	df["prediction"] = df["explanation"].apply(lambda f: re.findall(r'\d+\.\d+',f)[1][0])
	df["label"] = df["explanation"].apply(lambda x: re.findall(r'\d+\.\d+',x)[2][0])
	return df



import argparse
from datetime import datetime
DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'

start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

parser = argparse.ArgumentParser(description='Config params.')

# experiments\independent\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\explanations\e_test-7_2020-05-24_03-16-15.txt

parser.add_argument('-p', metavar='path', type=str,
                    help='expl path')
args = parser.parse_args()

path = fix_file(args.p)
df = load_explanations(path)

# print(df.head())

print(df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100])
print(df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]["contribution"].mean())
print(df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]["contribution"].max())
print(df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]["contribution"].min())
print(df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]["contribution"].std())

contributions_top = df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]["contribution"]

import matplotlib.pyplot as plt
plt.hist(contributions_top, label="contribution")
plt.title('First 100 contributions plot')
plt.legend()

plt.savefig("contributions_hist.png") 

