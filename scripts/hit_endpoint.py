#!/usr/bin/env python

# Python 3 script hitting a REST endpoint
# AHN, Jun 2019

import requests
from pdb import set_trace as BP

URL = 'https://ahaux.com/leela_server/select-move/leela_gtp_bot?tt=1234'
ARGS = {'board_size':19,'moves':[],'config':{'randomness':0.5,'request_id':'0.6834311059880898'}}

#-------------
def main():
    res = hit_endpoint( URL, ARGS)
    print( res)

# Hit an endpoint with a POST request
#----------------------------------------
def hit_endpoint( url, args):
    try:
        resp = requests.post( url, json=args)
        res = resp.json()
        return res
    except Exception as e:
        print( 'ERROR: hit_endpoint() failed: %s' % str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
