#!/bin/bash

# tracks the memory load of a specific process as long as its running
# required parameters: 
#       param #1: PID of the process to track

pid=$1

max_mem=0

while ps --pid $pid &>/dev/null
do
    mem=$(ps -p $pid -o pmem=)
    if (($(echo "$mem > $max_mem" | bc -l)))
    then
        max_mem=$mem
    fi
done

echo "Maximum (resilient) memory usage:$max_mem GB"