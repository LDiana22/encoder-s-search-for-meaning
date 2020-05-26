import pandas as pd

def load_explanations(path):
	df = pd.DataFrame()
	df.read_csv(path, sep="~")
	return df
	# with open(path, "r") as f:
	# 	line = f.readline()
	# 	while line:
	# 		line_data = line.split(" ~ ")

	# 		expl_text = line_data[1].split("'")[1]
	# 		expl_freq = line_data[1].split("(")
			
	# 		return 
	# 		f.readline()


import argparse
from datetime import datetime

start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

parser = argparse.ArgumentParser(description='Config params.')

parser.add_argument('-p', metavar='path', type=str,
                    help='expl path')
args = parser.parse_args()

df = load_explanations(args.p)
print(df.head())