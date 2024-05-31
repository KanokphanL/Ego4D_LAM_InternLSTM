
import pandas as pd
import glob

csv_file1 = 'validate/gazeintern_multrainset/result/pred.csv'
csv_file2 = 'validate/gazeintern_multrainset_fliptest/result/pred.csv'

file_paths = [csv_file1, csv_file2]

dataframes = [pd.read_csv(file, header=None) for file in file_paths]

merged_df = pd.concat(dataframes)

unique_columns = [0, 1, 2, 3]
grouped_df = merged_df.groupby(unique_columns)

averaged_df = grouped_df.agg({merged_df.columns[-1]: 'mean'}).reset_index()

averaged_df.to_csv('new.csv', index=False, header=False)
