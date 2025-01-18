"""
Args: 
    input.csv 
    output_1.csv
    output_2.csv
    percent: int, percentage of rows to select from input.csv
    seed: int, random seed (optional, default = 6)
Choose percent of rows from input.csv and save them to output_1.csv
the rest of the rows are saved to output_2.csv if provided
    - preserves headers in both output files
"""

import pandas as pd
import argparse
import warnings

warnings.filterwarnings('ignore')

# get args from command line
parser = argparse.ArgumentParser()
parser.add_argument('input', help = 'input csv file')
parser.add_argument('output_1', help = 'output csv file')
parser.add_argument('output_2', help = 'output csv file')
parser.add_argument('percent', help = 'percentage of rows to select from input.csv', type = int)
parser.add_argument('--seed', help = 'random seed', type = int, default = 6)
args = parser.parse_args()

# read input csv file
df = pd.read_csv(args.input)
# split dataframe randomly
df_1 = df.sample(frac = args.percent / 100, random_state = args.seed)
df_2 = df.drop(df_1.index)
# save to output csv files
df_1.to_csv(args.output_1, index = False)
df_2.to_csv(args.output_2, index = False)

# total number of rows in input file
print(f'number of rows in {args.input}: {len(df)}')

# print number of rows in each output file
print(f'number of rows in {args.output_1}: {len(df_1)}')
print(f'number of rows in {args.output_2}: {len(df_2)}')
