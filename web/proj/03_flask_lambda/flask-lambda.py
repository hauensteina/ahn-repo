#!/usr/bin/env python

# /********************************************************************
# Filename: flask-lambda.py
# Author: AHN
# Creation Date: Feb, 2020
# **********************************************************************/
#
# A hello world flask app deployed on AWS lambda.
#

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    return jsonify({"message": "Hello World!"})

if __name__ == '__main__':
 app.run()
