#!/usr/bin/env python

# /********************************************************************
# Filename: ahn-kv.py
# Author: AHN
# Creation Date: Feb, 2020
# **********************************************************************/
#
# A general purpose REST key-value store, using AWS S3, running on AWS Lambda.
# Keys are strings, values are JSON.

from pdb import set_trace as BP
from flask import Flask, jsonify, abort, make_response, request, url_for
from flask_httpauth import HTTPBasicAuth
import json,os

# AWS S3 api
import boto3

auth = HTTPBasicAuth()
app = Flask(__name__)

AWS_KEY = ''
AWS_SECRET = ''

# Security
#=============

@auth.get_password
#----------------------------
def get_password(username):
    if username == 'anybody':
        return 'Pyiaboar.'
    return None

@auth.error_handler
#--------------------
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), 403)

# Endpoints
#=============

@app.route('/ahn-kv/list_all', methods=['GET'])
@auth.login_required
#-----------------------------------------------------
def ahn_kv_list_all():
    return jsonify( s3_get_keys())

@app.route('/ahn-kv/list_prefix', methods=['POST'])
@auth.login_required
#-----------------------------------------------------
def ahn_kv_list_prefix():
    if not request.json or not 'prefix' in request.json:
        abort(400)
    return jsonify( s3_get_keys( request.json['prefix']))

@app.route('/ahn-kv/get', methods=['POST'])
@auth.login_required
#--------------------------------------------------------
def ahn_kv_get():
    if not request.json or not 'key' in request.json:
        abort(400)
    res = s3_get_value( request.json['key'])
    if not res:
        abort(404)
    res = {
        'key': request.json['key'],
        'value': res
    }
    return jsonify( res)

@app.route('/ahn-kv/put', methods=['POST'])
@auth.login_required
#-----------------------------------------------------
def ahn_kv_put():
    if not request.json or not 'key' in request.json or not 'value' in request.json:
        abort(400)
    kv = {
        'key': request.json['key'],
        'value': request.json.get('value')
    }
    s3_store_kv( kv['key'], kv['value'])
    return jsonify( kv), 200

@app.route('/ahn-kv/delete', methods=['POST'])
@auth.login_required
#---------------------------------------------------------------
def ahn_kv_delete():
    if not request.json or not 'key' in request.json:
        abort(400)
    s3_delete_kv( request.json['key'])
    return jsonify( True)

# Helpers
#==========

@app.errorhandler(404)
#-------------------------
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

# S3 stuff
#===========

AWS_KEY = os.environ['AWS_KEY']
AWS_SECRET = os.environ['AWS_SECRET']

S3 = boto3.client('s3',
                  aws_access_key_id=AWS_KEY,
                  aws_secret_access_key=AWS_SECRET)
S3_BUCKET = 'zappa-ahaux'
S3_FOLDER = 'ahn-kv'

#------------------------------
def s3_store_kv( key, value):
    key = S3_FOLDER + '/' + key
    S3.put_object( Body=json.dumps(value), Bucket=S3_BUCKET, Key=key)

#--------------------------
def s3_delete_kv( key):
    key = S3_FOLDER + '/' + key
    S3.delete_object(  Bucket=S3_BUCKET, Key=key)

#------------------------
def s3_get_value( key):
    key = S3_FOLDER + '/' + key
    try:
        value = S3.get_object( Bucket=S3_BUCKET, Key=key)
    except:
        return None
    res = json.loads( value['Body'].read())
    return res

#-----------------------------
def s3_get_keys( prefix=''):
    prefix = S3_FOLDER + '/' + prefix
    res = S3.list_objects( Bucket=S3_BUCKET, Prefix=prefix)
    llen = len(S3_FOLDER) + 1
    if 'Contents' in res:
        res = [ x['Key'][llen:] for x in res['Contents'] ]
        return res
    return []

if __name__ == '__main__':
    app.run(debug=True)
