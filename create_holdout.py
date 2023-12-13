import pandas as pd 
import numpy as np


# load data
train = pd.read_csv("Train.csv")
print(f"The old length of the Dataset: {len(train)}")

# divide the 21 outliers by 10

outliers = train[((train.Yield / train.Acre) >= 6500)]
print(f"Number of outliers: {len(outliers)}")
mult_by_10_index = outliers.index
train.loc[mult_by_10_index, 'Yield'] = outliers.Yield / 10

# sample random ids and create holdout set
np.random.seed(123)
rand_idx = np.random.choice(train.index, size = int(len(train) * 0.15), replace=False)

train_holdout = train.loc[rand_idx]
train_new = train.drop(rand_idx)
print(f"length of the new train set: {len(train_new)}"), print(f"length of the holdout set: {len(train_holdout)}")

print("Distribution of District in old train set:") 
print(f"{train['District'].value_counts()/len(train)}")
print("Distribution of District in new train set:") 
print(f"{train_new['District'].value_counts()/len(train_new)}")
print("Distribution of District in new test set:") 
print(f"{train_holdout['District'].value_counts()/len(train_holdout)}")

# create the new csv files
train_new.to_csv('train_new.csv', index=False)
train_holdout.to_csv('train_holdout.csv', index=False)



