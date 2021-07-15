#!/bin/bash

# tracks the memory load of a specific process as long as its running
# required parameters: 
#       param #1: PID of the process to track

pid=$1

max_mem=0
max_mem_gpu=0

while ps --pid $pid &>/dev/null
do
    mem=$(ps -p $pid -o pmem=)
    mem_gpu=$(gpustat -p | grep $pid | cut -d '|' -f3 | cut -d '/' -f1)
    if (($(echo "$mem > $max_mem" | bc -l)))
    then
        max_mem=$mem
    fi
    if [[ $mem_gpu -gt $max_mem_gpu ]]
    then
        max_mem_gpu=$mem_gpu
    fi
done

echo "Maximum (resilient) memory usage:$max_mem GB"
echo "Maximum GPU memory usage:$max_mem_gpu MB"