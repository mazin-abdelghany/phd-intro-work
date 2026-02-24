#!/bin/bash

# how many times should delta minimax run
num_runs=100

# delta minimax parameter values
null_hypothesis=0
alt_hypothesis=1
variance=3
alpha=0.05
power=0.9
num_stages=3

# run minimax with the above parameters and output to file names run_i
for ((i = 1; i <= num_runs; i++))
do
    ./finddeltaminimaxdesign $null_hypothesis $alt_hypothesis $variance \
                             $alpha $power $variance run_$i.txt
done
