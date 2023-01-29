import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
svmodel=pickle.load(open('svmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=svmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=svmodel.predict(final_input)[0]
    if output==1:
        return render_template("home.html",prediction_text="the person is being detected with parkinson having 83.4% accuracy and result is {}".format(output))
    else:
        return render_template("home.html",prediction_text="the person is notbeing detected with parkinson with 83.4% accuracy and result is {}".format(output))




if __name__=="__main__":
    app.run(debug=True)
   
     
