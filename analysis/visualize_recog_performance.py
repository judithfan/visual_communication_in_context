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


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()

	parser.add_argument('--data_fname', type=str, \
									  help='filepath to retrieve data from', 
									  default='./sketchpad_basic_recog_group_data_2.csv')

	args = parser.parse_args()	
	
	## load in CSV
	X = pd.read_csv('./sketchpad_basic_recog_group_data_2.csv')

	## make plots dir if does not already exist
	if not os.path.exists('./plots'):
		os.makedirs('./plots')

	## how many sessions (proxy for number of observers) do we have?
	print 'Number of sessions: {}'.format(len(np.unique(X.gameID.values)))    	

	## histogram of annotations per sketch
	fig = plt.figure(figsize=(6,6))
	x = X.groupby('fname')['gameID'].count().reset_index()
	h = sns.distplot(x['gameID'].values,bins=16,kde=False)
	plt.xlim(0,15)
	xt = plt.xticks(np.arange(0,16))
	plt.xlabel('number of annotations')
	plt.ylabel('number of sketches')
	plt.tight_layout()
	plt.savefig('./plots/num_annotations_per_sketch.pdf')

	## what is object-level accuracy broken out condition?
	print 'What is object-level accuracy?'
	print X.groupby('condition')['correct'].apply(lambda x: np.mean(x)).reset_index()

	## what is class-level accuracy?
	print 'What is category-level accuracy?'
	print X.groupby('condition')['correct_class'].apply(lambda x: np.mean(x)).reset_index()	