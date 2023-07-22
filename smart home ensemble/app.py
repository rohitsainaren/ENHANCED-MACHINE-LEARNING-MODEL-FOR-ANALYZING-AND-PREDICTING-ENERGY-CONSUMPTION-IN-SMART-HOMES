import pickle

import numpy as np
from flask import Flask,request,jsonify,render_template

#create flask application

app = Flask(__name__)

#load pickle model

model1 = pickle.load(open("model.pkl","rb"))
ranpk2 = pickle.load(open("ran2.pkl","rb"))
dec1pk = pickle.load(open("dec1.pkl","rb"))
dec2pk = pickle.load(open("dec2.pkl","rb"))
xg1pk = pickle.load(open("xg1.pkl","rb"))
xg2pk = pickle.load(open("xg2.pkl","rb"))
en1pk = pickle.load(open("en1.pkl","rb"))
en2pk = pickle.load(open("en2.pkl","rb"))



@app.route("/")
def Home():
    return render_template("index.html")
@app.route("/an",methods=["POST"])
def an():
    return render_template("table.html")
@app.route("/predict1",methods = ["POST"])
def predict1():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model1.predict(features)

    return render_template("index.html",prediction_text = "The predicted values using random forest for house overall are {}".format(prediction))
@app.route("/ran2",methods = ["POST"])
def ran2():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = ranpk2.predict(features)

    return render_template("index.html",
                           prediction_text1="The predicted values using random forest for home office are {}".format(
                               prediction))
@app.route("/dec1",methods = ["POST"])
def dec1():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = dec1pk.predict(features)

    return render_template("index.html",
                           prediction_text2="The predicted values using decision tree for house overall are {}".format(
                               prediction))
@app.route("/dec2",methods = ["POST"])
def dec2():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = dec2pk.predict(features)

    return render_template("index.html",
                           prediction_text3="The predicted values using decision tree for home office are {}".format(
                               prediction))
@app.route("/xg1",methods = ["POST"])
def xg1():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = xg1pk.predict(features)

    return render_template("index.html",
                           prediction_text4="The predicted values using xg boost for house overall are {}".format(
                               prediction))
@app.route("/xg2",methods = ["POST"])
def xg2():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = xg2pk.predict(features)

    return render_template("index.html",
                           prediction_text5="The predicted values using xgboost for home office are {}".format(
                               prediction))
@app.route("/en1",methods = ["POST"])
def en1():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = en1pk.predict(features)

    return render_template("index.html",
                           prediction_text2="The predicted values using ensemble for house overall are {}".format(
                               prediction))
@app.route("/en2",methods = ["POST"])
def en2():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = en2pk.predict(features)

    return render_template("index.html",
                           prediction_text2="The predicted values using ensemble for house overall are {}".format(
                               prediction))
@app.route("/table")
def table():
    return render_template("table.html")
@app.route("/image")
def image():

    img_urls = ['static/Graphs/DT_BarGraph.png',"static/Graphs/DT_ScatterPLot.png","static/Graphs/Ensemble_BarGraph.png","static/Graphs/Ensemble_ScatterPlot.png"
        ,"static/Graphs/RF_BarGraph.png","static/Graphs/RF_ScatterPlot.png","static/Graphs/XGBoost_BarGraph.png","static/Graphs/XGBoost_ScatterPlot.png"]

    return render_template("image.html",img_urls = img_urls)
if __name__ == "__main__":
    app.run(debug=True)