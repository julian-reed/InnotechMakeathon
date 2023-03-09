import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn import preprocessing
import json 
from pandas.io.json import json_normalize
import urllib.parse
import json
import requests


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():


    data = request.get_data()

    decoded_data = data.decode('utf-8')


    form_data_decoded = urllib.parse.parse_qs(decoded_data)

    # Convert the dictionary to a Pandas DataFrame
    df = pd.DataFrame.from_dict(form_data_decoded)

    #form_data_encoded = urllib.parse.urlencode(data)
    #form_data_decoded = urllib.parse.parse_qs(form_data_encoded)
    #json_data = json.dumps(form_data_decoded)
    # headers = {'Content-Type': 'application/json'}
    # response = requests.post(url, data=json_data, headers=headers)
    #data = json.loads(data)
    # df = json_normalize(data)

    #encoded version for cleaned dataset
    labelDict = {}
    df_encode = df
    for feature in df_encode:
        le = preprocessing.LabelEncoder()
        le.fit(df_encode[feature])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        df_encode[feature] = le.transform(df_encode[feature])
        # Get labels
        labelKey = 'label_' + feature
        labelValue = [*le_name_mapping]
        labelDict[labelKey] =labelValue

    X = df_encode[['Age', 'Gender', 'Country', 'self_employed', 'family_history', 'work_interfere',
    'no_employees', 'remote_work', 'tech_company', 'benefits',
    'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave',
    'mental_health_consequence', 'phys_health_consequence', 'coworkers',
    'supervisor', 'mental_health_interview', 'phys_health_interview',
    'mental_vs_physical', 'obs_consequence']]

    prediction = model.predict(X)
    

    if prediction == 0:
        output = 'No, you can focus on managing your mental health'
    else:
        output = 'Yes, it would be better to seek professional help'
    

    return render_template('results.html', prediction_text='Do you require an early professional help? {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
