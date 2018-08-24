import os
import numpy as np

if __name__ == "__main__":

## which have already been run
enumerate_dir = os.path.join(os.getcwd(),'enumerateOutput')
results_dir = os.path.join(enumerate_dir,os.listdir(enumerate_dir)[0])
print 'Results dir is: {}'.format(results_dir)

completed_inds = map(int,[i.split('_')[2] for i in os.listdir(results_dir)])

## which still need to be run
still_to_run = [i for i in np.arange(1,200,2) if i not in completed_inds]
print 'These still yet to run: {}'.format(still_to_run)

## identify how many different new RSA.py calls to make
## by getting list of straggler lower bounds
straggler_lb = np.where(np.diff(still_to_run) !=2)[0]
if len(straggler_lb)==0:
	print 'No stragglers, run them all.'

## which split?
which_split = os.listdir(results_dir)[0].split('_')[0]

	## go through and run RSA.py for earliest to latest in list of still to run
	lb = still_to_run[0]
	ub = still_to_run[-1]
	cmd_string = 'python RSA.py --wppl BDA-enumerate --sim_scaling_lb {} --sim_scaling_ub {} --step_size 2 --split_type {}'.format(lb, ub, which_split)
	print 'Running: {}'.format(cmd_string)
	os.system(cmd_string)
