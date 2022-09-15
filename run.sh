#!/usr/bin/env bash

if [ ! -z "$3" ]
then
    echo $3
    python /data/condor_builds/users/jasonfan/3ml_sensitivity_v42/background_trial.py --nscramble 5000 --dec $1 --surfix $2 --extension $3 --wrkdir /data/condor_builds/users/jasonfan/3ml_sensitivity_v42/result_dir/
else
    echo "No extension"
    python /data/condor_builds/users/jasonfan/3ml_sensitivity_v42/background_trial.py --nscramble 5000 --dec $1 --surfix $2 --wrkdir /data/condor_builds/users/jasonfan/3ml_sensitivity_v42/result_dir/
fi
