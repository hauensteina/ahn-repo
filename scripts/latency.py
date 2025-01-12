
'''
Measure internet latency by comparing a 1GB S3 download to 1000 downloads of size 1MB.
AHN, Jan 2025

RESULTS:
==========

TL;DR
Nothing beats a cable.

Mac connected by ethernet cable upload
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1GB: 11.12508487701416 seconds  (about 100 MB per sec)
1000 files of 1MB: 241.40392112731934 seconds (about 4 MB per sec)
upload latency per file: 0.23027883625030518 seconds

Mac connected by ethernet cable download
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1GB: 16.79356813430786 seconds (62 MB per sec)
1000 files of 1MB: 193.17018699645996 seconds (about 5MB per sec)
download latency per file: 0.1763766188621521 seconds

Mac connected by old wifi (Google Mesh) upload
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1GB: 45.35641384124756 seconds (22 MB per sec)
1000 files of 1MB: 261.99924302101135 seconds (3.8 MB per sec)
upload latency per file: 0.2166428291797638 seconds

Mac connected by old wifi (Google Mesh) download
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1GB: 113.60409712791443 seconds (9 MB per sec)
1000 files of 1MB: 583.9016978740692 seconds (1.7 MB per sec)
download latency per file: 0.4702976007461548 seconds

Mac connected by new wifi (Nest Wifi pro) upload
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1GB: 26.999014854431152 seconds 37.038388452009656 MB/sec
1000 files of 1MB: 249.85127305984497 seconds 4.002381047546144 MB/sec
upload latency per file: 0.22285225820541382 seconds

Mac connected by new wifi (Nest Wifi pro) download
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1GB: 23.684571027755737 seconds 42.22157956030147 MB/sec
1000 files of 1MB: 202.9877290725708 seconds 4.926406165381981 MB/sec
download latency per file: 0.17930315804481506 seconds

Mac connected by iphone personal hotspot upload (5G UC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1GB: 447.8588778972626 seconds 2.232846214180434 MB/sec
1000 files of 1MB: 664.756402015686 seconds 1.5043104466053767 MB/sec
upload latency per file: 0.21689752411842347 seconds

Mac connected by personal hotspot download (5G UC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1GB: 27.33896803855896 seconds 36.57782541716999 MB/sec
1000 files of 1MB: 279.3228530883789 seconds 3.5800865877723074 MB/sec
download latency per file: 0.25198388504981994 seconds

'''

from pdb import set_trace as BP
import pprint
import os
import argparse
import sys
import time
import boto3
from boto3.s3.transfer import S3Transfer, TransferConfig

BUCKET_NAME = 'ahn-uploads'
FOLDER_NAME = 'latency'
TMPFOLDER = '/tmp'

#--------------
def usage():
    name = os.path.basename( __file__)
    msg = f'''
    Name:
      {name}: Try to measure internet download latency by comparing a 1GB S3 download to 1000 downloads of size 1MB.

    Examples:
      python {name} --test_upload
      python {name} --test_download

''' 
    msg += '\n '
    return msg 

#-------------
def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( '--test_upload', action='store_true')
    parser.add_argument( '--test_download', action='store_true')

    args = parser.parse_args()
    
    if len(sys.argv) < 2:
        print( usage())
        sys.exit(1) 
    
    if args.test_upload:
        test_upload()
    if args.test_download:
        test_download()

#----------------------------------------        
def test_download():        
    # Download a 1GB file and measure how long it takes
    print('Testing download of 1GB')
    tstart = time.time()
    download_file_s3( f'1GB')
    tend = time.time()
    t_gig = tend - tstart
    os.remove( f'{TMPFOLDER}/1GB')
    print( f'Download of 1GB took {t_gig} seconds {1000 / t_gig} MB/sec')

    # Upload 1000 files of size 1MB
    print('Testing download of 1000 times 1MB ')
    NFILES = 1000
    tstart = time.time()
    for i in range(NFILES):
        download_file_s3( f'1MB_{i}')
    tend = time.time()
    t_meg = tend - tstart

    # Remove the files
    for i in range(NFILES):
        os.remove( f'{TMPFOLDER}/1MB_{i}')     

    print( f'Download of 1GB took {t_gig} seconds {1000 / t_gig} MB/sec')
    print( f'Download of 1000 files of 1MB took {t_meg} seconds {1000 / t_meg} MB/sec')
    print(f'Download latency per file is { (t_meg - t_gig) / 1000 } seconds')

#----------------------------------------        
def test_upload():        
    # Upload a 1GB file and measure how long it takes
    os.system( f'dd if=/dev/zero of={TMPFOLDER}/1GB bs=1M count=1000')
    tstart = time.time()
    upload_file_s3( f'{TMPFOLDER}/1GB')
    tend = time.time()
    t_gig = tend - tstart
    os.remove( f'{TMPFOLDER}/1GB')
    print( f'Upload of 1GB took {t_gig} seconds {1000 / t_gig} MB/sec')

    # Upload 1000 files of size 1MB
    NFILES = 1000
    for i in range(NFILES):
        os.system( f'dd if=/dev/zero of={TMPFOLDER}/1MB_{i} bs=1M count=1')
    tstart = time.time()
    for i in range(NFILES):
        upload_file_s3( f'{TMPFOLDER}/1MB_{i}')
    tend = time.time()
    t_meg = tend - tstart

    # Remove the files
    for i in range(NFILES):
        os.remove( f'{TMPFOLDER}/1MB_{i}')     

    print( f'Upload of 1GB took {t_gig} seconds {1000 / t_gig} MB/sec')
    print( f'Upload of 1000 files of 1MB took {t_meg} seconds {1000 / t_meg} MB/sec')
    print(f'Upload latency per file is { (t_meg - t_gig) / 1000 } seconds')

#----------------------------------------
def get_s3_client():
    if get_s3_client.client: return get_s3_client.client
    
    key = os.environ['AWS_KEY']
    secret = os.environ['AWS_SECRET']
    client = boto3.client('s3', aws_access_key_id=key, aws_secret_access_key=secret)
    get_s3_client.client = client
    return client

get_s3_client.client = None    

#------------------------------------------
def upload_file_s3(fname):
    client = get_s3_client()
    basename = os.path.basename( fname)
    client.upload_file(fname, BUCKET_NAME, FOLDER_NAME + '/' + basename)
        
#-----------------------------------------
def download_file_s3(fname):
    client = get_s3_client()         
    client.download_file(BUCKET_NAME, FOLDER_NAME + '/' + fname , TMPFOLDER + '/' + fname)
 
if __name__ == '__main__':    
    main()
