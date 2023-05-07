# https://github.com/Parasgr7/Movie-Recommendation-System
# AutoEncoders

import numpy as np
import pandas as pd
import random# UserID::Gender::Age::Occupation::Zip-code
# MovieID::Title::Genres
# UserID::MovieID::Rating::Timestamp (5-star scale)

# Importing the dataset
movies = pd.read_csv('movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
# ratings = pd.read_csv('train.csv', sep = ',', header = None, engine = 'python', encoding = 'latin-1')
# ratings[3] = ratings1[3][:len(ratings)]
num_users = len(ratings[0].unique())
num_movies= len(ratings[1].unique())
sorted_item_id = sorted(ratings[1].unique())

# sorted_item_id[:31]
# # making item id continuous
# for i in range(len(sorted_item_id)):
#     ratings[1] = ratings[1].replace(sorted_item_id[i], i+1)

train_lst = []
val_lst   = []
test_lst  = []

for uid in range(num_users):
    watches = ratings.loc[ratings[0] == uid]
    
    train_lst.append(watches.iloc[:int(len(watches)*0.7)])
    val_lst.append(watches.iloc[int(len(watches)*0.7):int(len(watches)*0.8)])
    test_lst.append(watches.iloc[int(len(watches)*0.8):])

train = pd.concat(train_lst)
val   = pd.concat(val_lst)
test  = pd.concat(test_lst)
train.to_pickle('./data/ml/train.pkl')
val.to_pickle('./data/ml/val.pkl')
test.to_pickle('./data/ml/test.pkl')

num_users  = int(max(max(train.values[:,0]), max(val.values[:,0]), max(test.values[:,0]))) + 1
num_movies = int(max(max(train.values[:,1]), max(val.values[:,1]), max(test.values[:,1]))) + 1
num_users, num_movies

train_lst = []
val_lst   = []
test_lst  = []
neg_lst  = []

for uid in range(1, num_users+1):
    watches = ratings.loc[ratings[0] == uid]
    
    watched = watches[1].values.tolist()
    unwatch = set(range(1, num_movies+1)) - set(watched)
    
    ns_list = random.sample(unwatch, 100)
    # print(watches.iloc[:-2])
    # print()
    # print(watches.iloc[-2])
    # print()
    # print(watches.iloc[-1])
    train_lst.append(watches.iloc[:-2])
    val_lst.append(watches.iloc[-2])
    test_lst.append(watches.iloc[-1])
    neg_lst.append(list(ns_list))

train = pd.concat(train_lst)
val   = pd.concat(val_lst, 1).T
test  = pd.concat(test_lst, 1).T

train.to_pickle('./data/ml/train_score.pkl')
val.to_pickle('./data/ml/val_score.pkl')
test.to_pickle('./data/ml/test_score.pkl')
np.save('./data/ml/neg_score.npy', neg_lst)