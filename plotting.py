from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import os
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']



feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day"]



train_df = pd.read_json("train.json")

def violinBedroomsBathrooms(train_df):
    train_df['bathrooms'].ix[train_df['bathrooms'] <= 0] = 1
    train_df['bedrooms'].ix[train_df['bedrooms'] <= 0] = 1

    train_df['bathrooms'].ix[train_df['bathrooms'] > 3] = 3
    train_df['bedrooms'].ix[train_df['bedrooms'] > 3] = 3

    train_df['bedrooms_bathrooms'] = train_df['bedrooms'] * 10.0 + train_df['bathrooms']
    print(train_df['bedrooms_bathrooms'])
    # plt.figure(figsize=(8,4))

    sns.violinplot(x='interest_level', y='bedrooms_bathrooms', data=train_df)
    plt.show()
    
def barBedroomsBathrooms(df):
    df['bathrooms'].ix[df['bathrooms'] <= 0] = 1
    df['bedrooms'].ix[df['bedrooms'] <= 0] = 1

    df['bathrooms'].ix[df['bathrooms'] > 3] = 3
    df['bedrooms'].ix[df['bedrooms'] > 3] = 3

    df['bedrooms_bathrooms'] = df['bedrooms'] * 10.0 + df['bathrooms']

    sns.countplot(x='bedrooms_bathrooms', hue='interest_level',data=df)
    plt.show()

def barBedroomsBathroomsPrice(df):
    df['bathrooms'].ix[df['bathrooms'] <= 0] = 1
    df['bedrooms'].ix[df['bedrooms'] <= 0] = 1

    df['bathrooms'].ix[df['bathrooms'] > 3] = 3
    df['bedrooms'].ix[df['bedrooms'] > 3] = 3

    df['bedrooms_bathrooms'] = df['bedrooms'] * 10.0 + df['bathrooms']

    df = df[df['price'] < 10000]

    red_df = df[df['interest_level'] == 'low']
    blue_df = df[df['interest_level'] == 'medium']
    green_df = df[df['interest_level'] == 'high']

    trans = {
        'low':'red',
        'medium': 'green',
        'high':'blue'
    }
    colors = [trans[x.interest_level] for i,x in df.iterrows()]
    plt.scatter(red_df.price, red_df.bedrooms_bathrooms, c=colors)

    #plt.yticks(np.arange(0,40,1))
    plt.yticks([10,11,12,13,14,15,20,21,22,23,24,30,31,32,33,34])
    plt.show()

barBedroomsBathroomsPrice(train_df)



