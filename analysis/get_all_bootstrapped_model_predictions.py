from __future__ import division
import os
import numpy as np
import pandas as pd
import thread
import analysis_helpers as h

'''
Wrapper around bootstrap_model_predictions.py. 
It will spawn several threads to get bootstrap vectors from all splits and models.

Estimate uncertainty in estimates of key variables of interest that are derived from model predictions,
e.g., target rank, sketch cost. Estimate sampling uncertainty by resampling trials with replacement from the
test data set for each split, marginalizing out the parametric uncertainty (from param posterior). 

Generate, for each model in model_space and each split_type in split_types, a boot_vec that is 
nIter in length (nIter=1000), and can be used to estimate standard error both within split and to get
standard error estimate when combining across splits.

'''


if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--nIter', type=int, help='how many bootstrap iterations?', default=1000)
	args = parser.parse_args()

	split_types = ['balancedavg1','balancedavg2','balancedavg3','balancedavg4','balancedavg5']      
	model_space = ['human_combined_cost','multimodal_fc6_combined_cost','multimodal_conv42_combined_cost','multimodal_fc6_S0_cost','multimodal_fc6_combined_nocost']
        model_space=['multimodal_fc6_combined_nocost']
        conditions = ['all','closer','further']
	vois = ['target_rank','foil_rank','sign_diff_rank','cost']

	nIter = args.nIter

	print 'Now running ...'
	for split_type in split_types:	 
		for model in model_space:
			for condition in conditions:
				for var_of_interest in vois:
					cmd_string = 'python bootstrap_model_predictions.py --split_type {} \
																		--model {} \
																		--condition {} \
																		--nIter {} \
																		--var_of_interest {}\
																		'.format(split_type,model,\
																				 condition,nIter,\
																				 var_of_interest)
					print cmd_string
					thread.start_new_thread(os.system,(cmd_string,))   
