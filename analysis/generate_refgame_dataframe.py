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
from __future__ import division
import scipy.stats as stats
import pandas as pd
import json
import re

from PIL import Image
import base64

import analysis_helpers as h
reload(h)

'''
Script to iterate through records on mongo and generate properly formatted
dataframe containing drawing reference game data.
'''

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()


	parser.add_argument('--out_path', type=str, \
									  help='filepath to write dataframe to', 
									  default='sketchpad_basic_pilot2_group_data')

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
	conn = pm.MongoClient('mongodb://sketchloop:' + pswd + '@127.0.0.1')
	db = conn['3dObjects']
	coll = db['sketchpad_basic']

	S = coll.find({ '$and': [{'iterationName':iterationName}, {'eventType': 'stroke'}]}).sort('time')
	C = coll.find({ '$and': [{'iterationName':iterationName}, {'eventType': 'clickedObj'}]}).sort('time')
	print str(S.count()) + ' stroke records in the database.'
	print str(C.count()) + ' clickedObj records in the database.'	

	# print unique gameid's
	unique_gameids = coll.find({ '$and': [{'iterationName':'pilot2'}, {'eventType': 'clickedObj'}]}).sort('time').distinct('gameid')
	# print map(str,unique_gameids)

	# filter out  records that match researcher ID's
	jefan = ['A1MMCS8S8CTWKU','A1MMCS8S8CTWKV','A1MMCS8S8CTWKS']
	hawkrobe = ['A1BOIDKD33QSDK']
	researchers = jefan + hawkrobe
	workers = [i for i in coll.find({'iterationName':'pilot2'}).distinct('workerId') if i not in researchers]	


	## get list of all gameids, tagging the ones that have been vetted manually as "valid."
	valid_gameids = []
	for i,g in enumerate(unique_gameids):
		W = coll.find({ '$and': [{'gameid': g}]}).distinct('workerId')
		for w in W:
			if w in workers:
				X = coll.find({ '$and': [{'workerId': w}, {'gameid': g}]}).distinct('trialNum') ## # of trials completed
				eventType = coll.find({ '$and': [{'workerId': w}]}).distinct('eventType')
				print i, w[:4], len(X), str(eventType[0])
				if (str(eventType[0])=='clickedObj') & (len(X)==32):
					valid_gameids.append(g)
	print '   ===========   '

	## filter if the pair did not follow instructions, consistently writing words or using other symbols in their drawings
	cheaty = ['8155-e46a25a3-9259-4b76-80e9-5bd79b6bdd97','6224-ab96ed5c-2a98-477c-aae2-7398b9e5b237',\
	         '5595-a00b8109-1910-43c4-9f14-00eb4945ac70','1697-7ab5b295-fae8-4f62-8cbd-72aa0e23b10e']
	motor = ['2829-820b338d-5720-4964-bd22-8ba38329569d'] # this person made multiples in several of their sketches, and appeared to suffer from strong tremors
	unfiltered_gameids = valid_gameids
	valid_gameids = [i for i in valid_gameids if i not in cheaty]
	valid_gameids = [i for i in valid_gameids if i not in motor]

	print str(len(valid_gameids)) + ' valid gameIDs (# complete games).'

	df = pd.DataFrame([valid_gameids])
	df = df.transpose()
	df.columns=['valid_gameids']
	df.to_csv('valid_gameids_pilot2.csv')

	df = pd.DataFrame([unfiltered_gameids])
	df = df.transpose()
	df.columns=['unfiltered_gameids']
	df.to_csv('unfiltered_gameids_pilot2.csv')

	## loop through all gameids to generate raw, unfiltered dataframe
	TrialNum = []
	GameID = []
	Condition = []
	Target = []
	Distractor1 = []
	Distractor2 = []
	Distractor3 = []
	Outcome = []
	Response = []
	numStrokes = []
	drawDuration = [] # in seconds
	viewerRT = []
	svgStringLength = [] # sum of svg string for whole sketch
	svgStringLengthPerStroke = [] # svg string length per stroke
	numCurvesPerSketch = [] # number of curve segments per sketch
	numCurvesPerStroke = [] # mean number of curve segments per stroke
	svgStringStd = [] # std of svg string length across strokes for this sketch
	Outcome = []
	Pose = []
	Svg = []

	these_gameids = unfiltered_gameids

	for g in these_gameids:
		print 'Analyzing game: ', g

		X = coll.find({ '$and': [{'gameid': g}, {'eventType': 'clickedObj'}]}).sort('time')
		Y = coll.find({ '$and': [{'gameid': g}, {'eventType': 'stroke'}]}).sort('time')

		for t in X:
			targetname = t['intendedName']
			distractors = [t['object2Name'],t['object3Name'],t['object4Name']]
			full_list = [t['intendedName'],t['object2Name'],t['object3Name'],t['object4Name']] 
			y = coll.find({ '$and': [{'gameid': g}, {'eventType': 'stroke'}, {'trialNum': t['trialNum']}]}).sort('time')
			ns = y.count()
			numStrokes.append(ns)
			drawDuration.append((y.__getitem__(ns-1)['time'] - y.__getitem__(0)['time'])/1000) # in seconds  
			y = coll.find({ '$and': [{'gameid': g}, {'eventType': 'stroke'}, {'trialNum': t['trialNum']}]}).sort('time')        
			z = coll.find({ '$and': [{'gameid': g}, {'eventType': 'clickedObj'}, {'trialNum': t['trialNum']}]}).sort('time')
			viewerRT.append((z.__getitem__(0)['time'] - y.__getitem__(ns-1)['time'])/1000)
			ls = [len(_y['svgData']) for _y in y]
			svgStringLength.append(reduce(lambda x, y: x + y, ls))
			y = coll.find({ '$and': [{'gameid': g}, {'eventType': 'stroke'}, {'trialNum': t['trialNum']}]}).sort('time')
			num_curves = [len([m.start() for m in re.finditer('c', _y['svgData'])]) for _y in y]
			y = coll.find({ '$and': [{'gameid': g}, {'eventType': 'stroke'}, {'trialNum': t['trialNum']}]}).sort('time')        
			_svg = []
			for _y in y:
			    _svg.append(_y['svgData'])
			Svg.append(_svg)
			numCurvesPerSketch.append(reduce(lambda x, y: x + y, num_curves))
			numCurvesPerStroke.append(reduce(lambda x, y: x + y, num_curves)/ns)
			svgStringLengthPerStroke.append(reduce(lambda x, y: x + y, ls)/ns)
			svgStringStd.append(np.std(ls))
			### aggregate game metadata
			TrialNum.append(t['trialNum'])
			GameID.append(t['gameid'])        
			Target.append(targetname)
			Condition.append(t['condition'])
			Response.append(t['clickedName'])
			Outcome.append(t['correct'])
			Distractor1.append(distractors[0])
			Distractor2.append(distractors[1])
			Distractor3.append(distractors[2])
			Pose.append(t['pose'])

	## compose pandas dataframe from lists
	iteration = ['pilot2']*len(GameID)

	_D = pd.DataFrame([GameID,TrialNum,Condition, Target, drawDuration, Outcome, Response, numStrokes, \
						svgStringLength, svgStringLengthPerStroke, svgStringStd, Distractor1, Distractor2, \
						Distractor3, Pose, iteration, Svg, viewerRT]) 

	D =_D.transpose()
	D.columns = ['gameID','trialNum','condition', 'target', 'drawDuration','outcome', 'response', \
				 'numStrokes', 'svgStringLength', 'svgStringLengthPerStroke', 'svgStringStd', \
				 'Distractor1', 'Distractor2', 'Distractor3', 'pose', 'iteration', 'svg','viewerRT']


	## add png to D dataframe
	png = []
	for g in these_gameids:
		X = coll.find({ '$and': [{'gameid': g}, {'eventType': 'clickedObj'}]}).sort('time')
		Y = coll.find({ '$and': [{'gameid': g}, {'eventType': 'stroke'}]}).sort('time')
		# print out sketches from all trials from this game
		for t in X: 
			png.append(t['pngString'])
	D = D.assign(png=pd.Series(png).values)

	iteration = ['pilot2']*len(D['gameID'].values)
	D = D.assign(iteration=pd.Series(iteration).values)

	## add another cost-related dependent measure: mean pixel intensity (amount of ink spilled) -- to handle
	## some weird glitches in the num stroke count
	mean_intensity = []
	imsize = 100
	numpix = imsize**2
	thresh = 250

	for i,_d in D.iterrows():
		imgData = _d['png']
		filestr = base64.b64decode(imgData)
		fname = os.path.join('sketch.png')
		with open(fname, "wb") as fh:
			fh.write(imgData.decode('base64'))
		im = Image.open(fname).resize((imsize,imsize))
		_im = np.array(im)
		mean_intensity.append(len(np.where(_im[:,:,3].flatten()>thresh)[0])/numpix)

	# add mean_intensity to the main D dataframe 
	### btw: mean_intensity and numStrokes is about 0.43 spearman correlated.
	D = D.assign(mean_intensity=pd.Series(mean_intensity).values)
	print stats.spearmanr(D['mean_intensity'].values,D['numStrokes'].values)

	category = [h.objcat[t] for t in D.target.values]
	D = D.assign(category=pd.Series(category).values)

	# save D out as group_data.csv 
	if len(np.unique(D.gameID.values))==len(valid_gameids):
		D.to_csv(os.path.join(analysis_dir,'{}.csv'.format(args.out_path)))
		print 'Saving out valid games csv at: {}.csv'.format(args.out_path)
	elif len(np.unique(D.gameID.values))==len(unfiltered_gameids):
		D.to_csv(os.path.join(analysis_dir,'{}_unfiltered.csv'.format(args.out_path)))
		print 'Saving out unfiltered games csv at {}_unfiltered.csv'.format(args.out_path) 

