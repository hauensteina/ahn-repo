#!/usr/bin/env python

# Listen on stdin, reply on stdout

from pdb import set_trace as BP
import sys,time

#--------------
def main():
    myid = int(sys.argv[1])
    while 1:
        cmd = sys.stdin.readline()
        #print('>>>' + cmd + '<<<')
        if cmd == 'quit':
            sys.exit(0)
        elif cmd.startswith( 'w '):
            secs = int( cmd.split()[1])
            time.sleep( secs)
            sys.stdout.write( '--> worker %d worked %d seconds\n' % (myid,secs))
            sys.stdout.flush()


if __name__ == '__main__':
    main()
