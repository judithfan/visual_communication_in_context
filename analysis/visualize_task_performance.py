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
dataframe containing experimental data.
'''

if __name__ == "__main__":
    import argparse

    parser.add_argument('--data_fname', type=str, \
    								  help='filepath to retrieve data from', 
    								  default='sketchpad_basic_pilot2_group_data.csv')

    args = parser.parse_args()

	## directory & file hierarchy
	iterationName = 'pilot2'
	exp_path = './'
	analysis_dir = os.getcwd()
	data_dir = os.path.abspath(os.path.join(os.getcwd(),'../../..','data',exp_path))
	exp_dir = './'
	sketch_dir = os.path.abspath(os.path.join(os.getcwd(),'../../..','analysis',exp_path,'sketches','pilot2'))


	## read group data csv in as D
	D = pd.read_csv(os.path.join(analysis_dir,args.data_fname))

	# extract some basic descriptive statistics and assign them to variables
	all_games = np.unique(D['gameID'])
	further_strokes = []
	closer_strokes = []
	further_drawDuration = []
	closer_drawDuration = []
	further_accuracy = []
	closer_accuracy = []
	closer_meanintensity = []
	further_meanintensity = []
	closer_viewerRT = []
	further_viewerRT = []

	for game in all_games:    
	    further_strokes.append(D[(D['gameID']== game) & (D['condition'] == 'further')]['numStrokes'].mean())
	    closer_strokes.append(D[(D['gameID']== game) & (D['condition'] == 'closer')]['numStrokes'].mean())   
	    further_drawDuration.append(D[(D['gameID']== game) & (D['condition'] == 'further')]['drawDuration'].mean())
	    closer_drawDuration.append(D[(D['gameID']== game) & (D['condition'] == 'closer')]['drawDuration'].mean())
	    further_accuracy.append(D[(D['gameID']== game) & (D['condition'] == 'further')]['outcome'].mean())
	    closer_accuracy.append(D[(D['gameID']== game) & (D['condition'] == 'closer')]['outcome'].mean())
	    closer_meanintensity.append(D[(D['gameID']== game) & (D['condition'] == 'closer')]['mean_intensity'].mean())    
	    further_meanintensity.append(D[(D['gameID']== game) & (D['condition'] == 'further')]['mean_intensity'].mean())        
	    closer_viewerRT.append(D[(D['gameID']== game) & (D['condition'] == 'closer')]['viewerRT'].mean())    
	    further_viewerRT.append(D[(D['gameID']== game) & (D['condition'] == 'further')]['viewerRT'].mean())     
	    
	further_strokes, closer_strokes, \
	further_drawDuration, closer_drawDuration, further_accuracy, closer_accuracy, \
	closer_meanintensity,further_meanintensity, closer_viewerRT, further_viewerRT = map(np.array, \
	[further_strokes, closer_strokes, \
	further_drawDuration, closer_drawDuration, further_accuracy, closer_accuracy, \
	closer_meanintensity, further_meanintensity, closer_viewerRT, further_viewerRT])


	## make communication task performance plot 
	sns.set_context('talk')
	plt.figure(figsize=(12,4))
	plt.subplot(1,5,1)
	ax = sns.barplot(data=D,x='condition',y='numStrokes')
	plt.ylabel('num strokes')
	plt.ylim(0,16)
	plt.subplot(1,5,2)
	sns.barplot(data=D,x='condition',y='mean_intensity')
	plt.ylabel('mean pixel intensity')
	plt.ylim(0,0.06)
	plt.subplot(1,5,3)
	sns.barplot(data=D,x='condition',y='outcome')
	plt.ylabel('accuracy')
	plt.ylim([0,1.01])
	plt.subplot(1,5,4)
	sns.barplot(data=D,x='condition',y='drawDuration')
	plt.ylabel('draw duration (s)')
	plt.ylim(0,35)
	plt.subplot(1,5,5)
	ax = sns.barplot(data=D,x='condition',y='viewerRT')
	plt.ylabel('viewer RT (s)')
	plt.ylim(0,10)
	plt.tight_layout()
	if not os.path.exists('./plots'):
	    os.makedirs('./plots')
	plt.savefig('./plots/sketchpad_basic_pilot2_taskperformance.pdf')
	plt.savefig('../manuscript/figures/raw/sketchpad_basic_taskperformance.pdf')	


	## Print out stats & confidence intervals of interest to the console

	## accuracy
	overall_accuracy = np.mean(D['outcome'].values)
	print 'Overall accuracy (collapsing across conditions) = {}'.format(np.round(overall_accuracy,3))

	accuracy_by_game = D.groupby('gameID')['outcome'].mean()
	boot, p, lb, ub = bootstrap(accuracy_by_game)
	print '95% CI for accuracy across games: ({}, {}), p = {}'.format(np.round(lb,3),np.round(ub,3),p)	
	
	boot, p, lb, ub = bootstrap(further_strokes - closer_strokes)
	print '95% CI for closer vs. further strokes: [{}, {}], p = {}'.format(np.round(lb,3),np.round(ub,3),p)

	boot, p, lb, ub = bootstrap(further_meanintensity - closer_meanintensity)
	print '95% CI for closer vs. further mean intensity: [{}, {}], p = {}'.format(np.round(lb,3),np.round(ub,3),p)

	boot, p, lb, ub = bootstrap(further_drawDuration - closer_drawDuration)
	print '95% CI for closer vs. further draw duration: [{}, {}], p = {}'.format(np.round(lb,3),np.round(ub,3),p)

	boot, p, lb, ub = bootstrap(further_viewerRT - closer_viewerRT)
	print '95% CI for closer vs. further viewer RT: [{}, {}], p = {}'.format(np.round(lb,3),np.round(ub,3),p)

