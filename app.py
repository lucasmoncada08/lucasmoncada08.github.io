import numpy as np
from flask import Flask, request, render_template
import pickle
from fastai2.vision.all import *
import os

# Saving the working directory and model directory
cwd = os.getcwd()
path = cwd + '/models'

app = Flask(__name__)

standing_model = load_learner(path, 'stage1.pkl')
bottom_model = load_learner(path, 'stage1.pkl')

@app.route('/predict/standing/<sImage>')
def standing_prediction():
    sPred = standing_model.predict(sImage)
    return sPred

@app.route('/predict/bottom/<bImage>')
def bottom_prediction():
    pred = bottom_model.predict(bImage)
    return pred

if __name__ == "__main__":
    app.run(debug=True)