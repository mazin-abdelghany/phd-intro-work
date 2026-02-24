#!/bin/bash

# how many times should delta minimax run
num_runs=2

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

if [ -f new_data.txt ]; then
    echo "Rename new_data.txt to avoid overwrites!"
else
    # above for loop outputs num_runs files; cat them all together into a single 
    # file with each line showing data from iteration of loop
    cat run*.txt > new_data.txt

    # add a row at the top, in place, that names each column created by the 
    # loops. this cannot be placed on multiple lines without adding extra space  
    # in output files
    sed -i "1i requiredtypeIerror requiredpower K seed typeIerror power expected_sample_size_null expected_sample_size_crd expected_sample_size_dm per_group lower1 upper1 lower2 upper2 lower3 upper3" new_data.txt

    # delete the num_run created files
    rm -rf run*.txt
fi

