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
import scipy
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

	## is accuracy different for close and far and sketches?
	p1 = X.groupby('condition')['correct'].apply(lambda x: np.mean(x))['closer']
	p2 = X.groupby('condition')['correct'].apply(lambda x: np.mean(x))['further']
	n1 = X.groupby('condition')['correct'].count()['closer']
	n2 = X.groupby('condition')['correct'].count()['further']
	p_hat = ((n1*p1)+(n2*p2))/(n1+n2)
	z = (p1 - p2) / (np.sqrt(p_hat * (1 - p_hat) * (1/n1 + 1/n2)))
	print 'Is accuracy different for close and far and sketches?'
	print 'Z stat for difference between two proportions (binomially distributed): {}'.format(z)

	## classical hypothesis test
	p = scipy.stats.binom_test(p1*n1, n=n1, p=1/32, alternative='two-sided')
	print 'Closer proportion diff from chance? p = {}'.format(p)

	p = scipy.stats.binom_test(p2*n2, n=n2, p=1/32, alternative='two-sided')
	print 'Further proportion diff from chance? p = {}'.format(p)

	##### MAKE PLOTS AND SAVE OUT 

	## plot recognition accuracy by condition
	sns.set_context('poster')
	fig = plt.figure(figsize=(4,4))
	redgld=[(0.8, 0.2, 0.2),(0.9, 0.7, 0.3)]
	sns.factorplot(y='correct',
					x='target_category',
					hue='condition',
					hue_order=['closer','further'],
					order=['bird','car','chair','dog'],
					data=X,kind='bar',palette=redgld)
	plt.ylim([0,1])
	plt.ylabel('proportion correct')
	plt.xlabel('category')
	h = plt.axhline(1/32,linestyle='dashed',color='black')
	plt.savefig('./plots/accuracy_by_category_and_condition.pdf')
	plt.close(fig)

	## plot recognition accuracy by condition
	plt.figure(figsize=(2,4))
	# sns.set_context('poster')
	redgld=[(0.8, 0.2, 0.2),(0.9, 0.7, 0.3)]
	sns.factorplot(y='correct',
					x='condition',
					# hue='condition',
					order=['further','closer'],
					# order=['bird','car','chair','dog'],
					data=X,kind='bar',palette=redgld, size=4.2, aspect=0.65)
	plt.ylim([0,1])
	plt.ylabel('proportion correct')
	plt.xlabel('category')
	h = plt.axhline(1/32,linestyle='dashed',color='black')
	plt.tight_layout()
	plt.savefig('./plots/accuracy_by_condition.pdf')
	plt.savefig('../manuscript/figures/raw/accuracy_by_condition.pdf')
	plt.close(fig)	

	## plot RT by condition
	sns.set_context('poster')
	# fig = plt.figure(figsize=(8,8))
	fig = plt.figure(figsize=(4,4))
	redgld=[(0.8, 0.2, 0.2),(0.9, 0.7, 0.3)]
	sns.factorplot(y='rt',
					x='target_category',
					hue='condition',
					data=X,kind='bar',palette=redgld)
	plt.ylim([0,8000])
	plt.ylabel('RT')
	plt.xlabel('category')
	plt.savefig('./plots/RT_by_category_and_condition.pdf')
	plt.close(fig)


	## subset by full games only
	all_games = np.unique(X.gameID.values)
	full_games = [i for i in all_games if np.sum(X['gameID']==i)>50]
	_X = X[X['gameID'].isin(full_games)]

	game_acc_close = _X[_X['condition']=='closer'].groupby('gameID')['correct'].apply(lambda x: np.mean(x))
	game_acc_far = _X[_X['condition']=='further'].groupby('gameID')['correct'].apply(lambda x: np.mean(x))
	fig = plt.figure(figsize=(6,6))
	plt.scatter(game_acc_close,game_acc_far)
	plt.xlim([0,1])
	plt.ylim([0,1])
	plt.plot([0,1],[0,1],linestyle='dashed')
	plt.title('accuracy by condition and game')
	plt.xlabel('close accuracy')
	plt.ylabel('far accuracy')
	plt.savefig('./plots/accuracy_by_condition_and_game.pdf')
	plt.close(fig)	

	## accuracy by condition and object
	sns.set_context('poster')
	objs = np.unique(X['target'].values)
	objs = [o for o in objs if o is not None]
	obj_acc_close = X[X['condition']=='closer'].groupby('target')['correct'].apply(lambda x: np.mean(x))
	obj_acc_far = X[X['condition']=='further'].groupby('target')['correct'].apply(lambda x: np.mean(x))
	fig = plt.figure(figsize=(6,6))
	plt.scatter(obj_acc_close,obj_acc_far,color='gray')
	for i, txt in enumerate(objs):
	    plt.annotate(txt, (obj_acc_close[i],obj_acc_far[i]))
	plt.xlim([0,1])
	plt.ylim([0,1]) 
	plt.plot([0,1],[0,1],linestyle='dashed',color='gray')
	plt.xlabel('close accuracy',fontsize=30)
	plt.ylabel('far accuracy',fontsize=30)
	plt.title('accuracy by condition and object',fontsize=30)
	plt.savefig('./plots/accuracy_by_condition_and_object.pdf')
	plt.close(fig)

	## RT by condition and object
	objs = np.unique(X['target'].values)
	objs = [o for o in objs if o is not None]
	obj_acc_close = X[X['condition']=='closer'].groupby('target')['rt'].apply(lambda x: np.mean(x))
	obj_acc_far = X[X['condition']=='further'].groupby('target')['rt'].apply(lambda x: np.mean(x))
	fig = plt.figure(figsize=(6,6))
	plt.scatter(obj_acc_close,obj_acc_far)
	# for i, txt in enumerate(objs):
	#     plt.annotate(txt, (obj_acc_close[i],obj_acc_far[i]))

	plt.xlim([0,15000])
	plt.ylim([0,15000])
	plt.plot([0,15000],[0,15000],linestyle='dashed')
	plt.xlabel('close RT')
	plt.ylabel('far RT')
	plt.title('RT by condition and object')
	plt.savefig('./plots/RT_by_condition_and_object.pdf')
	plt.close(fig)	


	## subset by full games only
	all_games = np.unique(X.gameID.values)
	full_games = [i for i in all_games if np.sum(X['gameID']==i)>50]
	_X = X[X['gameID'].isin(full_games)]

	game_acc_close = _X[_X['condition']=='closer'].groupby('gameID')['rt'].apply(lambda x: np.median(x))
	game_acc_far = _X[_X['condition']=='further'].groupby('gameID')['rt'].apply(lambda x: np.median(x))
	fig = plt.figure(figsize=(6,6))
	plt.scatter(game_acc_close,game_acc_far)
	# plt.xlim([0,20000])
	# plt.ylim([0,20000])
	# plt.plot([0,20000],[0,20000],linestyle='dashed')
	plt.title('RT by condition and game')
	plt.xlabel('close RT')
	plt.ylabel('far RT')
	plt.savefig('./plots/RT_by_condition_and_game.pdf')
	plt.close(fig)


	## subset by full games only
	all_games = np.unique(X.gameID.values)
	full_games = [i for i in all_games if np.sum(X['gameID']==i)>50]
	_X = X[X['gameID'].isin(full_games)]

	acc = _X.groupby('gameID')['correct'].apply(lambda x: np.mean(x))
	rt = _X.groupby('gameID')['rt'].apply(lambda x: np.mean(x))
	fig = plt.figure(figsize=(6,6))
	plt.scatter(acc,rt)
	plt.xlabel('accuracy')
	plt.ylabel('RT')
	plt.title('RT vs. accuracy by game')
	plt.savefig('./plots/RT_vs_accuracy_by_game.pdf')
	plt.close(fig)



	## MAKE CONFUSION MATRIX

	## all sketches

	## initialize confusion matrix
	confusion = np.zeros((len(obj_list),len(obj_list)))

	## generate confusion matrix by incrementing in each cell
	for i,d in X.iterrows():
	    targ_ind = obj_list.index(d['target'])
	    choice_ind = obj_list.index(d['choice'])
	    confusion[targ_ind,choice_ind] += 1
	    
	## normalized confusion matrix    
	normed = np.zeros((len(obj_list),len(obj_list)))
	for i in np.arange(len(confusion)):
	    normed[i,:] = confusion[i,:]/np.sum(confusion[i,:])    
	    
	## plot confusion matrix
	from matplotlib import cm
	fig = plt.figure(figsize=(8,8))
	ax = plt.subplot(111)
	cax = ax.matshow(normed,vmin=0,vmax=1,cmap=cm.viridis)
	plt.xticks(range(len(normed)), obj_list, fontsize=12,rotation='vertical')
	plt.yticks(range(len(normed)), obj_list, fontsize=12)
	plt.colorbar(cax,shrink=0.8)
	plt.tight_layout()
	plt.savefig('./plots/confusion_matrix_all.pdf')
	plt.close(fig)


	## divided by condition	

	conds = ['closer','further']

	for cond in conds:
	    ## initialize confusion matrix 
	    confusion = np.zeros((len(obj_list),len(obj_list)))

	    _X = X[X['condition']==cond]
	    ## generate confusion matrix by incrementing in each cell
	    for i,d in _X.iterrows():
	        targ_ind = obj_list.index(d['target'])
	        choice_ind = obj_list.index(d['choice'])
	        confusion[targ_ind,choice_ind] += 1

	    ## normalized confusion matrix    
	    normed = np.zeros((len(obj_list),len(obj_list)))
	    for i in np.arange(len(confusion)):
	        normed[i,:] = confusion[i,:]/np.sum(confusion[i,:])    

	    ## plot confusion matrix
	    from matplotlib import cm
	    fig = plt.figure(figsize=(8,8))
	    ax = plt.subplot(111)
	    cax = ax.matshow(normed,vmin=0,vmax=1,cmap=cm.viridis)
	    plt.xticks(range(len(normed)), obj_list, fontsize=12,rotation='vertical')
	    plt.yticks(range(len(normed)), obj_list, fontsize=12)
	    plt.colorbar(cax,shrink=0.8)
	    plt.tight_layout()
	    plt.savefig('./plots/confusion_matrix_{}.pdf'.format(cond))
	    plt.close(fig)


	# ##### plot difference between close and far conditions

	conds = ['closer','further']
	normed = np.zeros((len(obj_list),len(obj_list),2))

	for k,cond in enumerate(conds):
	    ## initialize confusion matrix 
	    confusion = np.zeros((len(obj_list),len(obj_list)))

	    _X = X[X['condition']==cond]
	    ## generate confusion matrix by incrementing in each cell
	    for i,d in _X.iterrows():
	        targ_ind = obj_list.index(d['target'])
	        choice_ind = obj_list.index(d['choice'])
	        confusion[targ_ind,choice_ind] += 1

	    ## normalized confusion matrix    
	    for i in np.arange(len(confusion)):
	        normed[i,:,k] = confusion[i,:]/np.sum(confusion[i,:])    

	## plot difference in confusion matrix
	from matplotlib import cm
	fig = plt.figure(figsize=(8,8))
	ax = plt.subplot(111)
	cax = ax.matshow(normed[:,:,0]-normed[:,:,1],vmin=-0.2,vmax=0.2,cmap=cm.BrBG)
	plt.xticks(range(len(normed)), obj_list, fontsize=12,rotation='vertical')
	plt.yticks(range(len(normed)), obj_list, fontsize=12)
	plt.colorbar(cax,shrink=0.8)
	plt.tight_layout()
	plt.savefig('./plots/confusion_matrix_close_minus_far.pdf')
	plt.close(fig)

	# save out to npy 
	np.save('./human_confusion.npy',normed)

	## plot diagonals of diff between confusion matrices
	fig = plt.figure(figsize=(8,8))	
	h = plt.hist(np.diagonal(normed[:,:,0]-normed[:,:,1]),20)
	plt.xlim(-0.5,0.5)
	plt.axvline(0,linestyle='dashed',color='black')
	plt.title('diff diagonal')
	plt.savefig('./plots/diagonals_of_diff_between_confusion_matrices.pdf')	
	plt.close(fig)

	## save out object order 
	x = pd.DataFrame([obj_list])
	x = x.transpose()
	x.columns = ['object']
	x.to_csv('./human_confusion_object_order.csv')

	## save out idealized human confusion for comparison
	def gen_block_diagonal(num_blocks,num_objs):
	    '''
	    num_blocks = how many square blocks do you want?
	    num_objs = number of rows in resulting matrix (equal to number of columns)
	    '''
	    assert num_objs%num_blocks==0
	    tmp = np.zeros([num_objs,num_objs])
	    ub = map(int,np.linspace(0,num_objs,num_blocks+1)[1:])
	    lb = map(int,np.linspace(0,num_objs,num_blocks+1)[:-1])
	    partitions = zip(lb,ub)
	    for l,u in partitions:
	        tmp[l:u,l:u] = 1
	    out = np.zeros([num_objs,num_objs])
	    for i,row in enumerate(tmp):
	        out[i] = row/np.sum(row)
	    return out

	idealized = np.zeros([32,32,2])
	idealized[:,:,0] = np.identity(32)
	idealized[:,:,1] = gen_block_diagonal(4,32)

	# save out to npy 
	np.save('./human_confusion_idealized.npy',idealized)
