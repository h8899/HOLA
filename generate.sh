#!/bin/bash

generate_variables() { 
   local input=()
   for (( i=1; i<=$1; i++))
   do
     gen_val= shuf -i1-200000 -n1
     input+=($gen_val)  
   done
   echo "${input[*]}"
}

arr=(1 1 3 1 2 2 2)

for i in {1..7}
do
    gcc 0$i.dig.c -o 0$i.out
    rm 0$i.output
    for j in {1..10000}
    do
	input=( $(generate_variables ${arr[$i-1]}) )
	echo "./0$i.out ${input[*]} >> 0$i.output"
        ./0$i.out ${input[*]} >> 0$i.output	
    done    
done
	
