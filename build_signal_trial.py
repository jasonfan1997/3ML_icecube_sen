import numpy as np

jobbase = '3ml_sen_signal'
script = '/data/condor_builds/users/jasonfan/3ml_sensitivity_v42/run_signal.sh'

counter = 0
dec=np.arange(-85,85,2.5)

for d in dec:
    for surfix in range(5):
        command = f'bash {script} {d} {surfix} 0.5'
        job_id = f'{jobbase}_{d}_{surfix}_05deg'
        print(f'JOB {job_id} /data/condor_builds/users/jasonfan/3ml_sensitivity_v42/submit.sub')
        print(f'VARS {job_id} JOBNAME="{job_id}" command="{command}"')
        counter += 1
