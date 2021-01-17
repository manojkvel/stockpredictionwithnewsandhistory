from flask import Flask, render_template, request, redirect
import model_lstm_nn as nn

# import selenium.webdriver as wd
# import uuid
# from selenium import webdriver
# from bs4 import BeautifulSoup
# import time
# from bs4.element import Tag
# import pandas as pd

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return 'POST'
    return render_template("index.html")


@app.route("/predict", methods=['GET', 'POST'])
def train_and_predict():
    if request.method == "POST":
        ticker = request.form['ticker']
        # sa.plotPolarities(sa.calculatePolarities(sa.parseFinviz(ticker)))
        nn.train_and_predict(ticker)
    return render_template("index.html")


if __name__ == "__main__":
    app.run()
