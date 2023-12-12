import pandas as pd 
import numpy as np


# load data
train = pd.read_csv("Train.csv")
print(f"The old length of the Dataset: {len(train)}")

# divide the 19 outliers by 10
a = 6000
b = 500

def line(x):
    return a * x + b

over_line_df = train[(train['Yield'] >= train['Acre'].apply(line)) & (train['Yield'] > 2500)]
print(f"Number of outliers: {len(over_line_df)}")
mult_by_10_index = over_line_df.index
train.loc[mult_by_10_index, 'Yield'] = over_line_df.Yield / 10

# sample random ids and create holdout set
np.random.seed(123)
rand_idx = np.random.choice(train.index, size = int(len(train) * 0.15), replace=False)

train_holdout = train.loc[rand_idx]
train_new = train.drop(rand_idx)
print(f"length of the new train set: {len(train_new)}"), print(f"length of the holdout set: {len(train_holdout)}")

# create the new csv files

train_new.to_csv('train_new.csv', index=False)
train_holdout.to_csv('train_holdout.csv', index=False)
