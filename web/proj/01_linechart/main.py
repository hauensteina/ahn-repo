#!/usr/bin/env python

# /********************************************************************
# Filename: linechart/main.py
# Author: AHN
# Creation Date: Apr, 2019
# **********************************************************************/
#
# Serve static html and accept API calls, both with flask
#

from pdb import set_trace as BP
import os, sys, re
import numpy as np
from flask import Flask
from flask import jsonify
from flask import request

#---------------
def main():
    here = os.path.dirname( __file__)
    static_path = os.path.join( here, 'static')
    app = Flask( __name__, static_folder=static_path, static_url_path='/static')

    #-------------
    # Endpoints
    #-------------

    # Use api to ececute func
    @app.route('/api/<func>', methods=['POST','GET'])
    #---------------------------------------------------
    def run_func( func):
        if request.method == 'POST':
            content = request.json
        else:
            content = request.args

        if func == 'get_data':
            res = get_data( content)
        else:
            return jsonify( {'error':'Unknown API endpoint: %s' % func}, 500)

        return jsonify( res, 200)

    app.run( host='0.0.0.0')


#-------------
# Methods
#-------------

# Return array of N x and y values for a line chart
#------------------------------------------------------
def get_data( args):
    N = int(args['N'])
    x = np.arange( 0, N).tolist()
    y = np.random.uniform( 0, 100, N).tolist()
    res = list( zip( x,y))
    return res

#-------------
# Helpers
#-------------


if __name__ == '__main__':
    main()
