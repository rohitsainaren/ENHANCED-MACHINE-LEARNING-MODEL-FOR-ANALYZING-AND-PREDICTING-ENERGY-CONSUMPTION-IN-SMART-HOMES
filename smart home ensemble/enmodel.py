import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns  # for data visualizatioon
import pickle
df = pd.read_csv('D:\webproj\HomeC (1).csv')
da = df.copy()
da.info()
da = da[0:-1] # removing the last invalid tuple
# column names are cleand by removing ' [kW]'

da.columns = [col.replace(' [kW]','') for col in da.columns]
# let's keep the aggregated features 'FurnaceSum' and 'KitchenSum' in dataset
# and drop the old features
da = da.drop(['Furnace 1','Furnace 2'],axis=1)
da = da.drop(['Kitchen 12','Kitchen 14','Kitchen 38'],axis=1)
print(da['time'][0])
db = da.copy()
# converting time from UNIX to readable data for entire dataset['time']
db['time'] = pd.date_range('2016-01-01 05:00:00',periods=len(df)-1,freq='min')
db['time'].head()
# set the time column as index for dataset
dbt = db.copy()
db = db.set_index(db['time'])
# now time is set as an index, so let's drop column time

db = db.drop(['time'],axis=1)
# the results looks too similar, so lets drop the 'use' column
db = db.drop(['use'],axis=1)

# 'cloudCover' seems to be numeric type, but there is an invalid value cloudCover
# instead of removing the column here we can replace the value with next valid observation
# bfill method replaces the NULL values with the values from the next row
db['cloudCover'].replace(['cloudCover'], method='bfill', inplace=True)
db['cloudCover'] = db['cloudCover'].astype('float')
db= db.drop(['summary'],axis=1)
# it is observed that there are no null values in the dataset

#copying the dataset
dc=db.copy()
x = dc[['temperature','humidity','visibility','apparentTemperature','pressure','windSpeed','cloudCover','windBearing','precipIntensity','dewPoint','precipProbability']]
house = dc['House overall'] # for House overall
home = dc['Home office']  #  for Home office
# splitting the dataset to 80-20 for House Overall
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
x_train,x_test,house_train,house_test = train_test_split(x,house,test_size=0.2,random_state=2) # splitting the datasets to train test for house overall
len(x_train) # x_train is weather
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

tree_model = DecisionTreeRegressor(random_state=42)
rf_model = RandomForestRegressor(random_state=42)
xgb_model = xgb.XGBRegressor(random_state=42)

# Step 3: Train decision tree and make predictions on test set
tree_model.fit(x_train, house_train)
tree_preds = tree_model.predict(x_test)

# Step 4: Add decision tree predictions as a new feature to train and test sets
x_train['tree_preds'] = tree_model.predict(x_train)
x_test['tree_preds'] = tree_preds

# Step 5: Train random forest and make predictions on test set (using decision tree predictions as input feature)
rf_model.fit(x_train, house_train)
rf_preds = rf_model.predict(x_test)

# Step 6: Add random forest predictions as a new feature to train and test sets
x_train['rf_preds'] = rf_model.predict(x_train)
x_test['rf_preds'] = rf_preds

# Step 7: Train XGBoost and make predictions on test set (using decision tree and random forest predictions as input features)
xgb_model.fit(x_train, house_train)
pickle.dump(xgb_model,open("en2.pkl","wb"))
final_preds = xgb_model.predict(x_test)