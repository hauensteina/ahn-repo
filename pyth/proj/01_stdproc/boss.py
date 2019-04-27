#!/usr/bin/env python

# Start up a few workers and tell them to do things

from pdb import set_trace as BP
import sys, os, signal
import subprocess
from threading import Thread,Lock
from queue import Queue,Empty
import atexit

g_procs = []
g_listeners = []
error_handler_lock = Lock()

#--------------
def main():
    global g_procs
    global g_listeners
    N_WORKERS = 2

    atexit.register( kill_procs)

    g_procs = []
    g_listeners = []
    for idx in range( N_WORKERS):
        p = start_worker( idx)
        g_procs.append( p)
        g_listeners.append( Listener( p.stdout, result_handler, lambda idx=idx :error_handler( idx)))
        print( 'made worker %d' % idx)

    cur_worker = 0
    while 1:
        # Command is 'w <n>', find a worker and
        # let it work n seconds before it replies.
        cmd = sys.stdin.readline()
        print( 'sending %s' % cmd)
        with error_handler_lock: # Wait until they come back to life if they died
            g_procs[cur_worker].stdin.write( cmd.encode('utf8'))
            g_procs[cur_worker].stdin.flush()
        cur_worker += 1
        cur_worker %= N_WORKERS

#-------------------------
def start_worker( idx):
    res = subprocess.Popen( ['python', 'worker.py', str(idx)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    return res

#-------------------
def kill_procs():
    for p in g_procs:
        if p.pid: os.kill( p.pid, signal.SIGKILL)

result_handler_lock = Lock()
#--------------------------------------
def result_handler( line):
    with result_handler_lock:
        print( line.decode(), end='')

# Resurrect dead workers
#--------------------------------------
def error_handler( idx):
    with error_handler_lock:
        print( 'Worker %d died. Resurrecting.' % idx)
        p = g_procs[idx]
        if p.pid: os.kill( p.pid, signal.SIGKILL)
        p = start_worker( idx)
        g_procs[idx] = p
        g_listeners[idx] = Listener( p.stdout, result_handler, lambda:error_handler(idx,p))
        print( 'Worker %d resurrected' % idx)


# Listen on a stream in a separate thread until
# a line comes in. Process line in a callback.
#=================================================
class Listener:
    #------------------------------------------------------------
    def __init__( self, stream, result_handler, error_handler):
        self.stream = stream
        self.callback = result_handler

        #------------------------------------
        def wait_for_line( stream, callback):
            while True:
                line = stream.readline()
                if line:
                    callback( line)
                else: # probably my process died
                    error_handler()
                    break

        self.thread = Thread( target = wait_for_line,
                          args = (self.stream, self.callback))
        self.thread.daemon = True
        self.thread.start()

#========================================
class UnexpectedEndOfStream(Exception):
    pass

if __name__ == '__main__':
    main()
