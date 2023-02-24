# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 10:30:56 2022

@author: A109021
"""

from flask import Flask, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
@app.route("/")
def hello():
    return render_template('home.html')

#@app.route('/home')
#def home():
#    return render_template('templates/home.html')

#@app.route('/predict', methods=['POST'])
#def predict():
#     json_ = request.json
#     query_df = pd.DataFrame(json_)
#     query = pd.get_dummies(query_df)
#     prediction = clf.predict(query)
#     return jsonify({'prediction': list(prediction)})
#if __name__ == '__main__':
#     clf = joblib.load('/Output/RandomForest.pkl')
#     app.run(port=8080)
