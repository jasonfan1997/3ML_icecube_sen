#!/usr/bin/env bash

if [ ! -z "$3" ]
then
echo $3
python /data/condor_builds/users/jasonfan/3ml_sensitivity_v42/signal_trial.py --nscramble 300 --dec $1 --surfix $2 --extension $3 --index 2.5 --wrkdir /data/condor_builds/users/jasonfan/3ml_sensitivity_v42/result_dir/
else
echo "No extension"
python /data/condor_builds/users/jasonfan/3ml_sensitivity_v42/signal_trial.py --nscramble 300 --dec $1 --surfix $2 --index 2.5 --wrkdir /data/condor_builds/users/jasonfan/3ml_sensitivity_v42/result_dir/

fi
