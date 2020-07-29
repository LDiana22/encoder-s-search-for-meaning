import pandas as pd
import os
import re
import argparse
from datetime import datetime
DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'
##################################################################
def fix_file(path):
    new_path = path[:path.rindex(".")]+"_fix.txt"
    print(f"New path {new_path}")
    with open(new_path, "w") as f:
        with open(path, "r") as g:
            line = g.readline()
            count = 0
            while line:
                count += 1
                if len(line.split("~")) > 4:
                    sublines = re.split('C: ((\+|-)?\d\.\d+(e-)?\d*)', line)
                    count=0
                    f.write(sublines[0] + sublines[1])
                    f.write("\n")
                    f.write(sublines[4] + sublines[5])
                else:
                    f.write("".join(line.split("C: ")))
                f.write("\n")
                line = g.readline() 
    return new_path

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
##################################################################


def plot_hist(contributions, title, path):
    import matplotlib.pyplot as plt
    plt.hist(contributions, label="contribution")
    # plt.hist(df[(df["prediction"]==df["label"])]["contribution"], label="contribution")
    plt.title(title)
    plt.legend()
    plt_path = "experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\\explanations\\" 
    plt.savefig(plt_path + "contributions_hist_100pn.png") 
'Contributions distribution of the first 100 correctly classified instances'

##################################################################

def print_metrics(df, full_path, field="contribution"):
    mean, min_c, max_c, std = df[field].mean(), df[field].min(), df[field].max(), df[field].std()
    with open(full_path, "w") as f:
        f.write(f"Mean: {mean}\n")
        f.write(f"Min: {min_c}\n")
        f.write(f"Max: {max_c}\n")
        f.write(f"StdDev: {std}\n")
        f.write(f"Positive %: {df[df[field]>0][field].count()*100/df[field].count()}\n\n")
        f.write(f"Positive contributions count {df[df[field]>0][field].count()}\n")
        f.write(f"All contributions count {df[field].count()}\n")

def print_percentages(df, full_path):
    cp = df[round(df["contribution"]+ df["raw_pred"])!=round(df["raw_pred"])].count()
    ccp = df[(round(df["contribution"]+ df["raw_pred"])!=round(df["raw_pred"])) & (df["label"]==df["prediction"])].count()
    icp = df[(round(df["contribution"]+ df["raw_pred"])!=round(df["raw_pred"])) & (df["label"]!=df["prediction"])].count()

    dp = df[round(df["raw_pred"])!=df["prediction"]].count()

    with open(full_path, "w") as f:
        f.write(f"Changed prediction: {cp}")
        f.write(f"Different predictions (should be equal to changed pred): {dp}")

        f.write(f"Correctly changed predictions: {ccp}")
        f.write(f"Correctly changed predictions (out of changed predictions): {ccp*100.0/cp}")

        f.write(f"Incorrectly changed predictions: {icp}")
        f.write(f"Incorrectly changed predictions (out of changed predictions): {icp*100.0/cp}")

        if df[round(df["contribution"]+ df["raw_pred"])!=round(df["prediction"])].count()!=0:
            f.write(f"Something's wrong: contribution+raw pred != prediction")

##################
# RUN COMMAND
#python qualitative_grep.py -p experiments\independent\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\explanations -f e_test-7_2020-05-24_03-16-15.txt
##################
start = datetime.now()
formated_date = start.strftime(DATE_FORMAT)

parser = argparse.ArgumentParser(description='Config params.')

# experiments\independent\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\explanations\e_test-7_2020-05-24_03-16-15.txt
#  experiments\independent\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\explanations\e_test-7_2020-05-24_03-16-15_fix.txt
parser.add_argument('-p', metavar='path', type=str,
                    help='expl path')
parser.add_argument('-f', metavar='file name', type=str,
                    help='expl file')
parser.add_argument('--fix', help='fix the explanation file format', action='store_true')

args = parser.parse_args()
##################################################################

e_path = os.path.join(args.p, args.f)
if args.fix:
    print("Fixing...")
    path = fix_file(e_path)
    print(f"Fixed {path}")
else:
    path = e_path # 'experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\\explanations\\e_test-7_2020-05-24_03-16-15_fix.txt'

print("Loading explanations...")
df = load_explanations(path)
print("Loaded")
##################################################################

def sample_metrics(df, path, sample_count=10):
    for i in range(10):
        sample_path = os.path.join(path,f"metrics_sample-{i}.txt")
        print(sample_path)
        sample = pd.DataFrame({"contribution":df["contribution"].sample(100)})
        print_metrics(sample, sample_path)
        sample.to_pickle(os.path.join(path,f"samples-dump-{i}.h5")) 

        hist_path = os.path.join(path, f"hist_sample-{i}.png")
        hist_title = "Sample contribution distribution for correctly classified test instances"
        plot_hist(sample['contribution'], hist_title, hist_path)

##################################################################

#sample_metrics(df[df["prediction"]==df["label"]], args.p)

print("all instances...")
all_metrics_path = os.path.join(args.p, "all_instances_metrics.txt")
print_metrics(df, all_metrics_path)
print_percentages(df, os.path.join(args.p, "percentages-changed-preds.txt"))

all_hist_path = os.path.join(args.p, "all_instances_hist.png")
plot_hist(df["contribution"], "Histogram all contributions", all_hist_path)

print("all correct instances...")
all_metrics_path = os.path.join(args.p, "all_correct_metrics.txt")
print_metrics(df[df["prediction"]==df["label"]], all_metrics_path)
pos_hist_path = os.path.join(args.p, "all_correct_hist.png")
plot_hist(df["contribution"], "Histogram positives' contributions", pos_hist_path)

print("all incorrect instances...")
all_metrics_path = os.path.join(args.p, "all_incorrect_metrics.txt")
print_metrics(df[df["prediction"]!=df["label"]], all_metrics_path)
neg_hist_path = os.path.join(args.p, "all_incorrect_hist.png")
plot_hist(df["contribution"], "Histogram negatives' contributions", neg_hist_path)

path = os.path.join(args.p, "descending_contribution.txt")
df.sort_values(by=["contribution"], ascending=False).to_csv(path)

path = os.path.join(args.p, "descending_contribution_correct.txt")
df[df["label"]==df["prediction"]].sort_values(by=["contribution"], ascending=False).to_csv(path)

path = os.path.join(args.p, "descending_contribution_incorrect.txt")
df[df["label"]!=df["prediction"]].sort_values(by=["contribution"], ascending=False).to_csv(path)

# print(df.head())
# print(df[(df["prediction"]==df["label"])].shape)
# print(df[(df["prediction"]==df["label"])]["contribution"].mean())
# print(df[(df["prediction"]==df["label"])]["contribution"].min())
# print(df[(df["prediction"]==df["label"])]["contribution"].max())
# print(df[(df["prediction"]==df["label"])]["contribution"].std())
# print(df[(df["prediction"]==df["label"])& (df["contribution"]>0)]["contribution"].count())
# print(100*df[(df["prediction"]==df["label"])& (df["contribution"]>0)]["contribution"].count()/df[(df["prediction"]==df["label"])]["contribution"].count())

# # print(df[df["contribution"]>0]["contribution"])
# top_100_positives = df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]
# # top_50pos = df[(df["prediction"]==df["label"]) & (df["prediction"]==1.0)]["contribution"][:50]
# # top_50neg = df[(df["prediction"]==df["label"]) & (df["prediction"]==0.0)]["contribution"][:50]
# # top_100 = top_50pos.append(top_50neg)

# top_100 = df["contribution"].sample(100)

# print(top_100.count())
# print(top_100[top_100>0].count())
# print(df[(df["prediction"]==df["label"])][:100][df["contribution"]>0].count())
# print(df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100])
# print(df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]["contribution"].mean())
# print(df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]["contribution"].max())
# print(df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]["contribution"].min())
# print(df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]["contribution"].std())

# contributions_top = df[(df["prediction"]==df["label"]) & (df["contribution"]>0)][:100]["contribution"]



# file_name = "experiments\\independent\\bilstm_mlp_improve-dnn15-1-25-decay0.0-L2-dr0.3-eval1-rake-polarity-improve100loss-alpha0.7-c-tr10\\explanations\\sample100.h5"
# top_100.to_pickle(file_name)  # where to save it, usually as a .pkl
# # Then you can load it back using:

# # df = pd.read_pickle(file_name)
