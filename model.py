# Step 1: Data cleaning

#importing necessary packages
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#load in the dataset
url = 'https://raw.githubusercontent.com/julian-reed/InnotechMakeathon/main/MentalHealthDataSet.csv'
df = pd.read_csv(url)

#Visualizing the dataset
df.head()

#Percentage of data missing in each column
missing_tot = df.isnull().sum()/len(df) * 100
print(missing_tot)

#Removed due to a significant amount of missnig data
df = df.drop(['state'], axis= 1)
#Removed because when the survey was taken shouldn't correlate with 
#overall mental health
df = df.drop(['Timestamp'], axis= 1)
#Removed because we won't ask for additional comments in practice
df = df.drop(['comments'], axis= 1)
df.head()

# Cleaning 'Gender' column

#Make all elements lower case
gender = df['Gender'].str.lower()

#Print unique elements
print(df['Gender'].unique())

#Group responses into gender groups: male, female and other
male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]       
female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail", "trans-female"]
other_str = ["something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]   

#Remove other special instances from the dataset as they likely are not serious
special = ['A little about you', 'p']
df = df[~df['Gender'].isin(special)]

#Replace values
for (row, col) in df.iterrows():

    if str.lower(col.Gender) in male_str:
        df['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

    if str.lower(col.Gender) in female_str:
        df['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

    if str.lower(col.Gender) in other_str:
        df['Gender'].replace(to_replace=col.Gender, value='other', inplace=True)

#Check that cleaning was successful
print(df['Gender'].unique())

#Replace missing age with median value
df['Age'].fillna(df['Age'].median(), inplace = True)
#Replace other falsified values with median
df['Age'] = df['Age'].replace([-29, 329, 99999999999, -1726], df['Age'].median())
print(df['Age'].unique())

#Since only 1.4% of values are missing, change those to not self-employed, 
#as that is the statistically more likely real value
df['self_employed'] = df['self_employed'].fillna('No')
print(df['self_employed'].unique())

#Fill NaN values with unknown
df['work_interfere'] = df['work_interfere'].fillna('Unknown')
print(df['work_interfere'].unique())

#Restict to only US survey responses due to low sample size in other countries
df_clean = df
df_clean = df_clean[df_clean.get("Country")=="United States"]
df_clean


# Step 2: Encoding

#encoded version for cleaned dataset
labelDict = {}
df_encode = df_clean
for feature in df_encode:
    le = preprocessing.LabelEncoder()
    le.fit(df_encode[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    df_encode[feature] = le.transform(df_encode[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] =labelValue
    
for key, value in labelDict.items():     
    print(key, value)

df_encode

df_encode.columns
# classification report

# Step 3: Model training

#XGBClassifier Model
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# all features except the treatment should be our x axis
X = df_encode[['Age', 'Gender', 'Country', 'self_employed', 'family_history', 'work_interfere',
        'no_employees', 'remote_work', 'tech_company', 'benefits',
       'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave',
       'mental_health_consequence', 'phys_health_consequence', 'coworkers',
       'supervisor', 'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence']]

# treatment is our target feature hence it's on y-axis
y = df_encode['treatment'] 

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = XGBClassifier(n_estimators = 50)
clf.fit(X_train, y_train)

y_prediction = clf.predict(X_test)

print(classification_report(y_test, y_prediction))

import pickle
pickle.dump(clf, open("model.pkl", "wb"))
