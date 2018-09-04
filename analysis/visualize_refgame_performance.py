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

import warnings
warnings.filterwarnings("ignore")

import analysis_helpers as h
reload(h)

'''
Script to iterate through records on mongo and generate properly formatted
dataframe containing experimental data.
'''

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()

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

	## compute mean accuracy and cost by game and condition
	further_strokes, closer_strokes = h.get_mean_by_condition_and_game(D, var='numStrokes')
	further_drawDuration, closer_drawDuration = h.get_mean_by_condition_and_game(D, var='drawDuration')
	further_meanintensity, closer_meanintensity = h.get_mean_by_condition_and_game(D, var='mean_intensity')
	further_accuracy, closer_accuracy = h.get_mean_by_condition_and_game(D, var='outcome')
	further_viewerRT, closer_viewerRT = h.get_mean_by_condition_and_game(D, var='viewerRT')

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
	accuracy_by_game = D.groupby('gameID')['outcome'].mean()
	boot, p, lb, ub = h.bootstrap(accuracy_by_game)
	print '95% CI for accuracy across games: ({}, {}), p = {}'.format(np.round(lb,3),np.round(ub,3),p)	
	print 'mean accuracy (collapsing acorss conditions) = {}'.format(np.round(overall_accuracy,3))	

	ACG = D.groupby(['gameID','condition'])['outcome'].mean().reset_index()
	grouped = ACG.groupby('condition')['outcome']
	for name,group in grouped:
		boot, p, lb, ub = h.bootstrap(group)
		print '    95% CI for accuracy for {} condition: [{}, {}], p = {}'.format(name,np.round(lb,3),np.round(ub,3),p)
		print '    mean = {}'.format(np.round(np.mean(group),3))

	print ' ======= '
	## cost	
	boot, p, lb, ub = h.bootstrap(further_strokes - closer_strokes)
	print '95% CI for closer vs. further strokes: [{}, {}], p = {}'.format(np.round(lb,3),np.round(ub,3),p)
	print 'mean difference in number of strokes = {}'.format(np.round(np.mean(further_strokes - closer_strokes),3))

	boot, p, lb, ub = h.bootstrap(further_meanintensity - closer_meanintensity)
	print '95% CI for closer vs. further mean intensity: [{}, {}], p = {}'.format(np.round(lb,3),np.round(ub,3),p)
	print 'mean difference in pixel intensity = {}'.format(np.round(np.mean(further_meanintensity - closer_meanintensity),3))

	boot, p, lb, ub = h.bootstrap(further_drawDuration - closer_drawDuration)
	print '95% CI for closer vs. further draw duration: [{}, {}], p = {}'.format(np.round(lb,3),np.round(ub,3),p)
	print 'mean draw difference in draw duration = {}'.format(np.round(np.mean(further_drawDuration - closer_drawDuration),3))

	boot, p, lb, ub = h.bootstrap(further_viewerRT - closer_viewerRT)	
	print '95% CI for closer vs. further viewer RT: [{}, {}], p = {}'.format(np.round(lb,3),np.round(ub,3),p)
	print 'mean difference in viewer RT = {}'.format(np.round(np.mean(further_viewerRT - closer_viewerRT),3))

	ACG = D.groupby(['gameID','condition'])['viewerRT'].mean().reset_index()
	grouped = ACG.groupby('condition')['viewerRT']
	for name,group in grouped:
		boot, p, lb, ub = h.bootstrap(group)
		print '    95% CI for viewer RT for {} condition: [{}, {}], p = {}'.format(name,np.round(lb,3),np.round(ub,3),p)
		print '    mean = {}'.format(np.round(np.mean(group),3))
