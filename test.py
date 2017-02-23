from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import os
import xgboost as xgb
import numpy as np
import pandas as pd

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']



feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day","bedrooms_bathrooms"]

def clean_data(df):
    df = df.dropna()
    df = df[df.bedrooms.apply(lambda x: type(x) == int or type(x) == float or x.isdecimal())]
    # test_df = test_df[test_df.bathrooms.apply(lambda x: type(x)==int or type(x)==float or x.isdecimal())]
    return df

def featureEngineering(df):
    #df = df[df['price'] < 20000]
    df['bedrooms'].ix[df['bedrooms'] > 4] = 5
    df['bathrooms'].ix[df['bathrooms'] > 4] = 5
    df['bathrooms'].ix[df['bedrooms'] < df['bathrooms']] = df['bedrooms']
    df["num_photos"] = df["photos"].apply(len)
    df["num_features"] = df["features"].apply(len)
    df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
    df["created"] = pd.to_datetime(df["created"])
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day

    df['bedrooms_bathrooms'] = df['bedrooms'] * 10.0 + df['bathrooms']


train_df = pd.read_json("train.json")

featureEngineering(train_df)
data = train_df[feats]
label = train_df['interest_level']


#model = xgb.XGBClassifier(n_estimators=20)
model = RandomForestClassifier(n_estimators=500)


x_train, x_test, y_train, y_test = train_test_split(data,label,test_size=0.3)
model.fit(x_train,y_train)
y_predicted = model.predict_proba(x_test)

print(log_loss(y_test,y_predicted))
exit()
print("Ready to test")

test_df = pd.read_json("test.json")

featureEngineering(test_df)

listing_id = test_df.listing_id
X = test_df[feats]

y = model.predict_proba(X)

labels2idx = {label: i for i, label in enumerate(model.classes_)}


res = pd.DataFrame()
res["listing_id"] = test_df["listing_id"]
for label in ["high", "medium", "low"]:
    res[label] = y[:, labels2idx[label]]

res.to_csv("./files/res.csv",encoding='utf-8',index=False)



