#!/bin/bash

proj=berzelius-2026-136
proj=berzelius-2026-117
for f in $(cat targets_to_do.txt);
do
    #prefer to echo, then grep for the sbatch command to see what will be submitted, then just pipe the output to bash to actually submit
#    echo sbatch -A $proj -J $f ./submit_train.sh $f --network_type 2dconv
    echo $f

    id=$(sbatch -A $proj -J "$f" ./submit_train.sh "$f" --network_type 2dconv --no-pae | awk '{print $4}')      
    id=$(sbatch -A $proj -J "$f" -d afterany:$id ./submit_train.sh "$f" --network_type 2dconv --no-pae | awk '{print $4}')

    id=$(sbatch -A $proj -J "$f" ./submit_train.sh "$f" --network_type mlp --no-pae | awk '{print $4}')      
    id=$(sbatch -A $proj -J "$f" -d afterany:$id ./submit_train.sh "$f" --network_type mlp --no-pae | awk '{print $4}')

    id=$(sbatch -A $proj -J "$f" ./submit_train.sh "$f" --network_type mlp | awk '{print $4}')      
    id=$(sbatch -A $proj -J "$f" -d afterany:$id ./submit_train.sh "$f" --network_type mlp | awk '{print $4}')     
    id=$(sbatch -A $proj -J "$f" -d afterany:$id ./submit_train.sh "$f" --network_type mlp | awk '{print $4}')

done

