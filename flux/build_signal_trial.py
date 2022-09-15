import numpy as np

jobbase = '3ml_convertflux'
script = '/data/condor_builds/users/jasonfan/3ml_sensitivity/flux/run.sh'

counter = 0
dec=np.arange(-85,85,2.5)

for d in dec:
    command = f'bash {script} {d}'
    job_id = f'{jobbase}_{d}'
    print(f'JOB {job_id} /data/condor_builds/users/jasonfan/3ml_sensitivity/flux/submit.sub')
    print(f'VARS {job_id} JOBNAME="{job_id}" command="{command}"')
    counter += 1
