from __future__ import division
import os
import numpy as np
import pandas as pd
import analysis_helpers as h

'''
Estimate uncertainty in estimates of key variables of interest that are derived from model predictions,
e.g., target rank, sketch cost. Estimate sampling uncertainty by resampling trials with replacement from the
test data set for each split, marginalizing out the parametric uncertainty (from param posterior). 

Generate, for each model in model_space and each split_type in split_types, a boot_vec that is 
nIter in length (nIter=1000), and can be used to estimate standard error both within split and to get
standard error estimate when combining across splits.

'''

split_types = ['balancedavg1','balancedavg2','balancedavg3','balancedavg4','balancedavg5']

model_space = ['human_combined_cost',
			   'human_S0_cost',
			   'human_combined_nocost',
               'multimodal_fc6_combined_cost',
               'multimodal_fc6_S0_cost',
               'multimodal_fc6_combined_nocost',
               'multimodal_conv42_combined_cost',
               'multimodal_pool1_combined_cost']

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()

	parser.add_argument('--model', type=str, 
	                               help='which model? human_combined_cost | \
	                                                  multimodal_fc6_combined_cost | \
	                                                  multimodal_conv42_combined_cost |\
	                                                  multimodal_fc6_S0_cost | \
	                                                  multimodal_fc6_combined_nocost | ', \
	                               default='human_combined_cost')
	parser.add_argument('--split_type', type=str, 
	                                    help='which split? balancedavg1 | \
	                                                       balancedavg2 | \
	                                                       balancedavg3 | \
	                                                       balancedavg4 | \
	                                                       balancedavg5 |', 
	                                    default='balancedavg1')


	parser.add_argument('--condition', type=str,
									   help='which condition(s)? all | closer | further',
									   default='all')

	parser.add_argument('--nIter', type=int, help='how many bootstrap iterations?', default=1000)
	parser.add_argument('--var_of_interest', type=str, 
											 help='which variable to get bootstrap estimates for?', 
											 default='cost')
	parser.add_argument('--out_dir', type=str,
									 help='where do you want to save out your bootstrap results?',
									 default='./bootstrap_results')


	args = parser.parse_args()


	## get name of model and split type to get predictions for, variable of interest, number of iterations
	model = args.model
	split_type = args.split_type
	var_of_interest = args.var_of_interest
	nIter = args.nIter   
	condition = args.condition 

	## load in model preds
	B = h.load_model_predictions(model=model,split_type=split_type)	
	B = B.sort_values(by=['sample_ind','trial']) ## make sure that B is sorted properly
	
	## subset by condition iff args.condition is either closer or further
	if args.condition in ['closer','further']:
		B = B[B['condition']==args.condition]
	else:
		B = B

	## run the bootstrap
	trial_list = np.unique(B.trial.values)
	num_trials = len(trial_list)
	num_samples = len(np.unique(B.sample_ind.values))

	print 'Running bootstrap with {} trials, for variable "{}" for {} iterations...'.format(num_trials,var_of_interest,nIter)

	boot_vec = []
	for boot_iter in np.arange(nIter):
		if boot_iter%10==0:
			print 'Now on boot iteration {}'.format(boot_iter)
		boot_ind = np.random.RandomState(boot_iter).choice(np.arange(num_trials),size=num_trials,replace=True)    
		grouped = B.groupby('sample_ind')
		_boot_vec = []
		for name, group in grouped:
			## append subsetted boot_vec to temp _boot_vec vector that is built up across groups
			_boot_vec = np.hstack((_boot_vec,group[var_of_interest].values[boot_ind])) 
		
		## add summary statistic to bootstrap vector
		if var_of_interest != 'sign_diff_rank':
			## compute boot sample mean, marginalizing over MCMC sample variability
			boot_vec.append(np.mean(_boot_vec))
		else:
			## if computing prop_congruent, get proportion congruent!
			prop_congruent = np.sum(_boot_vec)/len(_boot_vec)
			boot_vec.append(prop_congruent)


	## now save out boot_vec
	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	boot_vec = np.array(boot_vec)
	out_path = os.path.join(args.out_dir, 'bootvec_{}_{}_{}_{}_{}.npy'.format(model,split_type,var_of_interest,condition,nIter))
	print 'Now saving out boot_vec at path: {}'.format(out_path)
	np.save(out_path,boot_vec)





