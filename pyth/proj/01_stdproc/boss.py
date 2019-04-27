#!/usr/bin/env python

# Start up a few workers and tell them to do things

from pdb import set_trace as BP
import sys, os, signal
import subprocess
from threading import Thread,Lock
from queue import Queue,Empty
import atexit

g_procs = []

#--------------
def main():
    global g_procs
    N_WORKERS = 2

    atexit.register( kill_procs)

    g_procs = []
    for idx in range( N_WORKERS):
        g_procs.append( subprocess.Popen( ['python', 'worker.py', str(idx)], stdin=subprocess.PIPE, stdout=subprocess.PIPE))
    listeners = []
    for idx,p in enumerate(g_procs):
        listeners.append( Listener( p.stdout, result_handler))
        print( 'made listener %d' % idx)

    cur_worker = 0
    while 1:
        # Command is 'w <n>', find a worker and
        # let it work n seconds before it replies.
        cmd = sys.stdin.readline()
        print( 'sending %s' % cmd)
        g_procs[cur_worker].stdin.write( cmd.encode('utf8'))
        g_procs[cur_worker].stdin.flush()
        cur_worker += 1
        cur_worker %= N_WORKERS

#-------------------
def kill_procs():
    for p in g_procs:
        if p.pid:
            os.kill( p.pid, signal.SIGTERM)


result_handler_lock = Lock()
#--------------------------------------
def result_handler( line):
    with result_handler_lock:
        print( line.decode(), end='')

# Listen on a stream in a separate thread until
# a line comes in. Process line in a callback.
#=================================================
class Listener:
    #---------------------------------------------
    def __init__( self, stream, result_handler):
        self.stream = stream
        self.callback = result_handler

        #------------------------------------
        def wait_for_line( stream, callback):
            while True:
                line = stream.readline()
                if line:
                    callback( line)
                else:
                    raise UnexpectedEndOfStream

        self.thread = Thread( target = wait_for_line,
                              args = (self.stream, self.callback))
        self.thread.daemon = True
        self.thread.start()

#========================================
class UnexpectedEndOfStream(Exception):
    pass

if __name__ == '__main__':
    main()
