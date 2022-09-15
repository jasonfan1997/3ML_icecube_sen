import numpy as np

jobbase = '3ml_convertflux'
script = '/data/condor_builds/users/jasonfan/3ml_sensitivity/flux/run.sh'

counter = 0
dec=np.arange(-85,85,2.5)

for d in dec:
    command = f'bash {script} {d}'
    print(f'{command}')
    counter += 1
