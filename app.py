#!/usr/bin/env python
# coding: utf-8

## Import Liberaries and Modules
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle
import json
import joblib
from scipy.special import boxcox, inv_boxcox
from sklearn.preprocessing import StandardScaler,OneHotEncoder , LabelEncoder ,normalize
import pandas as pd
from datetime import datetime


## Disable Tesorflow / Keras Warning
# import warnings
# warnings.filterwarnings("ignore")
# import tensorflow as tf

# # load and evaluate a saved model keras
# from numpy import loadtxt
# from keras.models import load_model

## Convert string into Datetime format
def convertdate(dstring):
    return datetime.strptime(dstring, '%Y-%m-%dT%H:%M:%S.%fZ')


app = Flask(__name__)

# # load model
# model = load_model('ann_model')
# # summarize model.
# # model.summary()
model = pickle.load(open('decision_tree.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # int_features = [int(x) for x in request.form.values()]
    features = [x for x in request.form.values()]
    print(features)

    # Boxcox transformation lambda values
    fitted_lambda_stake = -0.0466345079709477
    fitted_lambda_betRate = -1.6678009857211833
    fitted_lambda_averagePriceMatched = -1.668065197571059

    ## List of numerical and categorical feature
    numerical_fet = ['stake_boxcox','betRate_boxcox','marketId', 'averagePriceMatched_boxcox']
    categorical_fet = ['type', 'eventType',  'marketName', 'event', 
                   'hour', 'week_of_the_year', 'weekday']


    ## Numerical Features Processing
    stake_boxcox = boxcox(float(features[0]), fitted_lambda_stake)+1
    betRate_boxcox = boxcox(float(features[1]), fitted_lambda_betRate)+1
    marketId = float(features[2])
    averagePriceMatched_boxcox = boxcox(float(features[3]), fitted_lambda_averagePriceMatched)+1

    print([stake_boxcox, betRate_boxcox, marketId, averagePriceMatched_boxcox])
    num_num = pd.DataFrame(columns = numerical_fet)
    # num_num = pd.DataFrame([stake_boxcox, betRate_boxcox, marketId, averagePriceMatched_boxcox],columns=numerical_fet)
    num_num.loc[len(num_num)] = [stake_boxcox, betRate_boxcox, marketId, averagePriceMatched_boxcox]
    print(num_num)
    print(num_num.shape)

    scalar = joblib.load('scaler.joblib')
    scaler_df = pd.DataFrame(data=scalar.transform(num_num), columns=numerical_fet)

    print(scaler_df)

    ## Categorical Features Processing
    type_name = features[4]
    eventType = features[5]
    marketName = features[6]
    event = features[7]
    placedDate = features[8]


    placedDate = convertdate(placedDate)
    print(placedDate)
    hour = placedDate.hour
    week_of_the_year = placedDate.isocalendar()[1]
    weekday = placedDate.weekday()

    # cat_cat = pd.DataFrame([type_name, eventType, marketName, eventevent, placedDate],columns=categorical_fet)
    cat_cat= pd.DataFrame(columns = categorical_fet)
    cat_cat.loc[len(cat_cat)] = [type_name, eventType, marketName, event, hour, week_of_the_year, weekday]
    print(cat_cat)

    onehot_encoder = joblib.load('onehot_encoder.joblib')
    onehot_df = pd.DataFrame(data=onehot_encoder.transform(cat_cat), columns=onehot_encoder.get_feature_names(categorical_fet))

    print(onehot_df)

    final_features = pd.concat([scaler_df, onehot_df], axis=1)
    print(final_features)
    print(final_features.shape)
    # prediction = model.predict(final_features.iloc[0].values.reshape(-1,1))
    prediction = model.predict(final_features)
    if prediction==0:
        return render_template('index.html', prediction_text='Bet is {} (WINNER_DECLARED)'.format(prediction[0]))
    else:
        return render_template('index.html', prediction_text='Bet is {} (INVALID_BET)'.format(prediction[0]))

    # output = round(prediction[0], 2)

    # return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)