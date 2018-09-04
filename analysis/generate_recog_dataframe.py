from __future__ import division

import os
import urllib, cStringIO

import pymongo as pm

import matplotlib
from matplotlib import pylab, mlab, pyplot
from IPython.core.pylabtools import figsize, getfigs
plt = pyplot
import seaborn as sns
sns.set_context('poster')
sns.set_style('white')

import numpy as np
import pandas as pd
import json
import re

from PIL import Image
import base64
import json

import analysis_helpers as h
reload(h)

## get standardized object list
categories = ['bird','car','chair','dog']
obj_list = []
for cat in categories:
	for i,j in h.objcat.iteritems():
		if j==cat:
			obj_list.append(i)             

'''
Script to iterate through records on mongo and generate properly formatted
dataframe containing recognition experiment data.
'''

if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()


	parser.add_argument('--out_path', type=str, \
									  help='filepath to write dataframe to', 
									  default='./sketchpad_basic_recog_group_data_2.csv')

	args = parser.parse_args()

	# directory & file hierarchy
	iterationName = 'pilot2'
	exp_path = './'
	analysis_dir = os.getcwd()
	data_dir = os.path.abspath(os.path.join(os.getcwd(),'../../..','data',exp_path))
	exp_dir = './'
	sketch_dir = os.path.abspath(os.path.join(os.getcwd(),'../../..','analysis',exp_path,'sketches','pilot2'))

	# set vars 
	auth = pd.read_csv('auth.txt', header = None) # this auth.txt file contains the password for the sketchloop user
	pswd = auth.values[0][0]
	user = 'sketchloop'
	host = 'rxdhawkins.me' ## cocolab ip address

	# have to fix this to be able to analyze from local
	import pymongo as pm
	conn = pm.MongoClient('mongodb://sketchloop:' + pswd + '@127.0.0.1')
	db = conn['3dObjects']
	coll = db['sketchpad_basic_recog']

	stimdb = conn['stimuli']
	stimcoll = stimdb['sketchpad_basic_pilot2_sketches']

	# How many sketches have been retrieved at least once? equivalent to: coll.find({'numGames':{'$exists':1}}).count()
	x = stimcoll.find({'numGames':{'$gte':0}}).count()
	y = coll.count()
	print '{} sketches in the stimuli db that have been retrieved at least once'.format(x)
	print '{} records in the recognition experiment database'.format(y)
	### "pilot2" of recog experiment run on Sunday April 22 2018
	print '{} unique workers.'.format(len(coll.find({'iterationName': {'$in': ['pilot0', 'pilot1', 'pilot2']}}).distinct('wID')))

	# PREPROCESS RECOGNITION TASK DATA

	## retrieve records from db
	## notes: 
	## pilot0 = no feedback onscreen
	## pilot1 = bonus point counter onscreen

	## make lists from db
	gameID = []
	target = []
	choice = []
	correct = []
	correct_class = []
	rt = []
	fname = []

	d1 = []
	d2 = []
	d3 = []
	target_category = []
	chosen_category = []
	condition = []
	drawDuration = []
	original_gameID = []
	viewer_correct = []
	viewer_choice = []
	viewer_RT = []
	mean_intensity = []
	num_strokes = []

	bad_sessions = ['1571-00d11ddf-96e7-4aae-ba09-1a338b328c0e','9770-2f360e9a-7a07-4026-9c36-18b558c1da21']
	a = coll.find({'iterationName': {'$in': ['pilot0', 'pilot1', 'pilot2']}}).sort('gameID').batch_size(50)

	counter = 0
	for rec in a:
		if rec['gameID'] not in bad_sessions:
			try:
				if counter%500==0:
					print '{} out of {} records analyzed.'.format(counter,a.count())
				if rec['target'] is not None:
					gameID.append(rec['gameID'])
					target.append(rec['target'])
					choice.append(rec['choice'])
					correct.append(rec['correct'])
					rt.append(rec['rt'])
					f = rec['sketch'].split('/')[-1]
					fname.append(f)
					chosen_category.append(h.objcat[rec['choice']])

					## match up with corresponding record in stimuli collection
					b = stimcoll.find({'fname_no_target':f})[0]
					assert stimcoll.find({'fname_no_target':f}).count()==1
					d1.append(b['Distractor1'])
					d2.append(b['Distractor2'])
					d3.append(b['Distractor2'])
					target_category.append(b['category'])
					correct_class.append(h.objcat[rec['choice']]==b['category'])
					condition.append(b['condition'])
					drawDuration.append(b['drawDuration'])
					original_gameID.append(b['gameID'])
					viewer_correct.append(b['outcome'])
					viewer_choice.append(b['response'])
					viewer_RT.append(b['viewerRT'])
					mean_intensity.append(b['mean_intensity'])  
					num_strokes.append(b['numStrokes'])    
					counter += 1
			except:
			    print 'Something went wrong with {} {}'.format(rec['gameID'],rec['trialNum'])
				pass


	## organize data into dataframe
	X = pd.DataFrame([gameID,target,choice,correct,rt,fname,d1,d2,d3,target_category,chosen_category,condition,drawDuration, \
	                 original_gameID,viewer_correct,viewer_choice,viewer_RT,mean_intensity,num_strokes,correct_class])
	X = X.transpose()
	X.columns = ['gameID','target','choice','correct','rt','fname','d1','d2','d3','target_category','chosen_category','condition','drawDuration', \
	            'original_gameID','viewer_correct','viewer_choice','viewer_RT','mean_intensity','num_strokes','correct_class']
	print '{} annotations saved.'.format(X.shape[0])

	## save proto version of X as X0
	X0 = X

	## remove NaN rows from data matrix (no target recorded)
	X = X[X['target'].isin(obj_list)]

	## filter out games that were particularly low accuracy 
	X['correct']=pd.to_numeric(X['correct'])
	acc = X.groupby('gameID')['correct'].mean().reset_index()
	acc_games = acc[acc['correct']>0.25]['gameID'] ## amounts to around 6% of data ## np.percentile(acc['correct'].values,6)
	X = X[X['gameID'].isin(acc_games)]

	## filter out responses that took too long, or too short
	too_fast = 1000
	too_slow = 30000
	X = X[(X['rt']>=too_fast) & (X['rt']<=too_slow)]

	print '{} annotations retained.'.format(X.shape[0])

	## save out to CSV
	print 'Saving dataframe to path: {}'.format(args.out_path)
	X.to_csv(args.out_path)	
