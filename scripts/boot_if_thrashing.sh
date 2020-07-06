#!/bin/bash

# Product of si and so
siso=`vmstat 10 2 | tail -n1 | awk  '{print $7*$8}'`
msg="`date` siso = $siso"
echo $msg
if [ $siso -gt 50 ]; then
    echo 'thrashing, rebooting'
    #shutdown -r now
fi
