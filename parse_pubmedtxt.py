import argparse
import sys
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--path", help="path of textfile downloaded from pubmed search")
#

def first_only(collection):
        first = [coll.split(".")[0] for coll in collection]
        return [f for f in first if f is not (None or "")]

def main(dir):
    with open("pm_sample.txt") as f:
    #print(f.read())
        text = f.read()

    #where the abstracts start
    abstracts = [t for t in text.split("\n\n") if "BACKGROUND:" in t]

    background_statements = [(".").join(a.split(":")[1].split(".")[:-1]).replace("\n", "") for a in abstracts ]
    objective_statements = [(".").join(a.split(":")[2].split(".")[:-1]).replace("\n", "") for a in abstracts ]
    discussion_statements = [(".").join ( (get(a.split(":"), 3) or "").split(".")[:-1]).replace("\n", "") for a in abstracts ]

    #only first sentences
    background_statements = first_only(background_statements)
    objective_statements = first_only(objective_statements)
    discussion_statements = first_only(discussion_statements)  


    df = pd.DataFrame([[s,1] for s in background_statements], columns=['text', 'label']).append(pd.DataFrame([[s,0] for s in objective_statements], columns=['text', 'label']), ignore_index=True).append(pd.DataFrame([[s,0] for s in discussion_statements], columns=['text', 'label']), ignore_index=True)
    df.to_csv("pm_sample.csv")
    print("parsing done -- csv ready")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.dir)