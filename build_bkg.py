import numpy as np

jobbase = '3ml_sen'
script = '/data/condor_builds/users/jasonfan/3ml_sensitivity_v42/run.sh'

counter = 0
dec=np.arange(-85,85,2.5)

for d in dec:
    for surfix in range(10):
        command = f'bash {script} {d} {surfix} 1'
        job_id = f'{jobbase}_{d}_{surfix}_1'
        print(f'JOB {job_id} /data/condor_builds/users/jasonfan/3ml_sensitivity_v42/submit.sub')
        print(f'VARS {job_id} JOBNAME="{job_id}" command="{command}"')
        counter += 1
