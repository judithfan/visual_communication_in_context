
# coding: utf-8

# In[ ]:

import os
import urllib, cStringIO

import pymongo as pm

import matplotlib
from matplotlib import pylab, mlab, pyplot
get_ipython().magic(u'matplotlib inline')
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
import json


# ## INPUT: preprocessed group data CSV from mongo (see pilot2_analysis_sketchpad_basic)
# ## OUTPUT: full set of bdaInput files 

# __list of bdaInpput files__: 
# * sketchData CSV (generated inside generate_bdaInput_csv)
# * test_examples TXT/JSON/CSV (generated inside generate_bdaInput_csv)
# * costs JSON
# * condition-lookup JSON
# * similarity JSON

# ## Because using human perceptual encoder, we need to generate similarity json from the annotation data

# ### read in human annotations csv (prepped in pilot2_analysis_sketchpad_basic_recog.ipynb)

# In[ ]:

X = pd.read_csv('./sketchpad_basic_recog_group_data_2.csv')
print 'Shape of un-augmented (additional columns) annotation csv: {}'.format(X.shape)

X = pd.read_csv('./sketchpad_basic_recog_group_data_2_augmented.csv')
print 'Shape of augmented annotation csv: {}'.format(X.shape)


# ### prep similarity jsons for evaluation by passing forward to RSA

# Format for json is a dictionary of dictionaries, where each top-level key refers to one of the renders, e.g. "cuckoo". For each render, you can look up the similarity with each sketch, referenced with an abbreviated ID taken by trimming the last 12-character string, and appending an underscore, and the trial number. E.g., 'gameID_9903-d6e6a9ff-a878-4bee-b2d5-26e2e239460a_trial_9.npy' ==> '26e2e239460a_9'

# In[ ]:

def load_json(json_path):
    with open(json_path) as fp:
        data = json.load(fp)  
    return data

def simplify_sketch(path): ## example path: 'gameID_9903-d6e6a9ff-a878-4bee-b2d5-26e2e239460a_trial_9.npy' ==> '26e2e239460a_9'
    path = '_'.join(os.path.splitext(os.path.basename(path))[0].split('_')[1:])
    path = path.split('-')[-1]
    path = path.replace('_trial', '')
    return path

from collections import Counter
import analysis_helpers as h
reload(h)
## get standardized object list
categories = ['bird','car','chair','dog']
obj_list = []
for cat in categories:
    for i,j in h.objcat.iteritems():
        if j==cat:
            obj_list.append(i)  


# In[ ]:

reallyRun = 1
if reallyRun:
    ## define list of renders and sketches
    render_list = obj_list
    sketch_list = np.unique(X['sketchID'])

    out_json = {}
    for i, this_render in enumerate(render_list):
        print i, this_render
        out_json[this_render] = {}
        for j,this_sketch in enumerate(sketch_list):        
            counts = np.zeros(len(obj_list)) ## initialize the probability vector
            choices = X[X['sketchID']==this_sketch]['choice'].values ## get list of all choices

            ## get counter dictionary
            cdict = Counter(choices)
            ## populate count vector accordingly
            for i,o in enumerate(obj_list):
                if o in cdict:
                    counts[i] = cdict[o]

            ## get probability vector by dividing by sum
            prob = counts/np.sum(counts)
            out_json[this_render][this_sketch] = prob[i]    

    ## output json in the same format as the other similarity jsons
    output_path = '../models/refModule/json/similarity-human_full25k-centroid.json'
    with open(output_path, 'wb') as fp:
        json.dump(out_json, fp)            


# In[ ]:




# ### generate bdaInput CSV

# In[ ]:

# directory & file hierarchy
iterationName = 'pilot2'
exp_path = './'
analysis_dir = os.getcwd()
data_dir = os.path.abspath(os.path.join(os.getcwd(),'../../..','data',exp_path))
exp_dir = './'
sketch_dir = os.path.abspath(os.path.join(os.getcwd(),'../../..','analysis',exp_path,'sketches','pilot2'))


# In[ ]:

D = pd.read_csv(os.path.join(analysis_dir,'sketchpad_basic_pilot2_group_data.csv'))
DUNFIL = pd.read_csv(os.path.join(analysis_dir,'sketchpad_basic_pilot2_group_data_unfiltered.csv'))

# filter out incorrect and invalid trials as well
incorrects = pd.read_csv('./incorrect_trial_paths_pilot2.txt',header=None)[0].values
invalids = pd.read_csv('./invalid_trial_paths_pilot2.txt',header=None)[0].values

def add_fnames(D):
    fname = []
    for i,_d in D.iterrows():
        fname.append('gameID_' + _d['gameID'] + '_trial_' + str(_d['trialNum']) + '_' + _d['target'] +'.png')    
    D = D.assign(fname=pd.Series(fname).values)  

    fname_no_target = []
    for i,_d in D.iterrows():
        fname_no_target.append('gameID_' + _d['gameID'] + '_trial_' + str(_d['trialNum']) +'.png')    
    D = D.assign(fname_no_target=pd.Series(fname_no_target).values) 
    return D

def filter_out_incorrect(D, incorrects):
    D = D[~D['fname'].isin(incorrects)]
    return D

def filter_out_invalids(D, invalids):
    D = D[~D['fname_no_target'].isin(invalids)] 
    return D
    
## add some filename columns
D = add_fnames(D)
DUNFIL = add_fnames(DUNFIL) ## version of dataframe with ALL trials, garbage games, incorrect trials, invalid trials

DNOINC = filter_out_incorrect(DUNFIL, incorrects) ## save version of D containing with incorrect trials filtered out only
DNOINV = filter_out_invalids(DUNFIL, invalids) ## save version of D containing with invalid trials filtered out only
D = filter_out_invalids(filter_out_incorrect(D, incorrects), invalids) ## both kinds of garbage filtered out

print np.shape(D)
print str(np.shape(D)[0]) + ' records in merged dataframe'

print '{} incorrect trials'.format(len(incorrects))
print '{} invalid trials'.format(len(invalids))
print ' '
print '{} trials with NO GARBAGE filtered out'.format(DUNFIL.shape[0])
print '{} trials with incorrects filtered out'.format(DNOINC.shape[0])
print '{} trials with invalids filtered out'.format(DNOINV.shape[0])
print '{} trials with ALL GARBAGE filtered out'.format(D.shape[0])


# In[ ]:

## now assign sketchID and class_label for the other versions of D
import analysis_helpers as h
reload(h)
def add_extra_label_columns(D):
    sketchID = [(i[-12:] + '_' + str(j)) for i,j in zip(D['gameID'].values,D['trialNum'].values)]
    D = D.assign(sketchID=pd.Series(sketchID).values)

    # add class label
    category = []
    classes = ['bird','car','chair','dog']
    for i,d in D.iterrows():
        category.append(h.objcat[d['target']])
    D = D.assign(category=pd.Series(category).values)
    return D

D = add_extra_label_columns(D)
DUNFIL = add_extra_label_columns(DUNFIL)
DNOINC = add_extra_label_columns(DNOINC)
DNOINV = add_extra_label_columns(DNOINV)


# In[ ]:




# In[ ]:

def generate_bdaInput_csv(D,filtration_level,train_test_split=True,
                          adaptor_type='human_full25k',
                          split_type='splitbyobject'):

    ### filter out training examples if using a trained (not human) encoder and if train_test_split==True
#     if (train_test_split==True) & (adaptor_type[:5]!='human'):
    if (train_test_split==True):        
        test_examples = pd.read_json('pilot2_{}_test_examples.json'.format(adaptor_type),orient='records')
        test_examples = list(test_examples[0].values)
        test_examples = [i.split('.')[0] + '.png' for i in test_examples]                
        keep_examples = test_examples 
    else:
        keep_examples = D['fname_no_target'].values ## keep all datapoints

    D0 = D[D['fname_no_target'].isin(keep_examples)]

    ## generate lists to compose new bdaInput CSV
    _sketchLabel = []
    _Condition = []
    _Target = []
    _Distractor1 = []
    _Distractor2 = []
    _Distractor3 = []
    _coarseGrainedSketchInfo = [] # condition_objectName ... e.g., further_knob
    for i, _d in D0.iterrows():
        _sketchLabel.append(_d['sketchID'])
        _Condition.append(_d['condition']) 
        _Target.append(_d['target'])
        distractor1 = _d['Distractor1']
        distractor2 = _d['Distractor2']
        distractor3 = _d['Distractor3']      
        
        d_list = sorted([distractor1, distractor2, distractor3])
        _Distractor1.append(d_list[0])
        _Distractor2.append(d_list[1])    
        _Distractor3.append(d_list[2])  
        _coarseGrainedSketchInfo.append('{}_{}'.format(_d['condition'],_d['target'])) 

    D2 = pd.DataFrame([_Condition,_sketchLabel,_Target,_Distractor1,_Distractor2,_Distractor3,_coarseGrainedSketchInfo])
    D2 = D2.transpose()
    D2.columns = ['condition','sketchLabel','Target','Distractor1','Distractor2','Distractor3','coarseGrainedSketchInfo']
    print '{} datapoints x {} columns'.format(D2.shape[0],D2.shape[1])
    
#     if (train_test_split==True) & (adaptor_type[:5]!='human'):
    if (train_test_split==True):        
        print 'saving CSV with only test data'
        if len(filtration_level)==0:
            D2.to_csv('../models/bdaInput/sketchData_fixedPose_{}_{}_pilot2.csv'.format(split_type,adaptor_type))
        else:  
            D2.to_csv('../models/bdaInput/sketchData_fixedPose_{}_{}_pilot2_{}.csv'.format(split_type,adaptor_type,filtration_level))
    else: ## run bda on ALL datapoints (not just test split)
        print 'saving CSV including all datapoints'
        if len(filtration_level)==0:
            D2.to_csv('../models/bdaInput/sketchData_fixedPose_alldata_{}_pilot2.csv'.format(adaptor_type))
        else:  
            D2.to_csv('../models/bdaInput/sketchData_fixedPose_alldata_{}_pilot2_{}.csv'.format(adaptor_type,filtration_level))        
    print 'Saved out bdaInput CSV ... {}'.format(filtration_level) 


# ## SIMILARITY FUNCTION PARAMS

# In[ ]:

## params
adaptor_type = 'human_full25k'
split_type = 'splitbyobject'    ### to compare with sketch_unroll_full25k


# In[ ]:

## only generate these CSV's if using a learned encoder

# first split versions
generate_bdaInput_csv(D,'',adaptor_type = adaptor_type,split_type = split_type)
generate_bdaInput_csv(DNOINC,'no_incorrect',adaptor_type = adaptor_type,split_type = split_type)
generate_bdaInput_csv(DNOINV,'no_invalid',adaptor_type = adaptor_type,split_type = split_type)
generate_bdaInput_csv(DUNFIL,'unfiltered',adaptor_type = adaptor_type,split_type = split_type)    

# now alldata versions
generate_bdaInput_csv(D,'',train_test_split=False,adaptor_type = adaptor_type,split_type = split_type)
generate_bdaInput_csv(DNOINC,'no_incorrect',train_test_split=False,adaptor_type = adaptor_type,split_type = split_type)
generate_bdaInput_csv(DNOINV,'no_invalid',train_test_split=False,adaptor_type = adaptor_type,split_type = split_type)
generate_bdaInput_csv(DUNFIL,'unfiltered',train_test_split=False,adaptor_type = adaptor_type,split_type = split_type)


# ### remove cost outliers

# In[ ]:

## load data in again and filter out cost outliers

def remove_outliers(X,column):
    mu = np.mean(X[column].values)
    sd = np.std(X[column].values)
    thresh = mu + 5*sd        
    X = X.drop(X[X[column] > thresh].index)
    return X
 
## make copy of D that has cost outliers removed    
D2 = remove_outliers(D,'drawDuration')    
D2 = remove_outliers(D2,'mean_intensity')
D2 = remove_outliers(D2,'numStrokes')

# print D2.shape

splits = ['splitbyobject','alldata']
    
for split in splits:
    ### subset drawing data csv by sketches that are accounted for here (i.e., that were not cost outliers)
    B = pd.read_csv('../models/bdaInput/sketchData_fixedPose_{}_{}_pilot2.csv'.format(split_type,adaptor_type))    
#     print B.shape

    remaining_sketches = list(np.unique(D2['sketchID'].values))
    _B = B[B['sketchLabel'].isin(remaining_sketches)]
#     print _B.shape
    _B.to_csv('../models/bdaInput/sketchData_fixedPose_{}_{}_pilot2_costOutliersRemoved.csv'.format(split,adaptor_type))


# In[ ]:




# In[ ]:




# ### generate cost dictionary

# In[ ]:

def sigmoid(x,k=1,x0=0.5):
    return 1 / (1 + np.exp(-k * (x - x0)))

def add_rescaled_metric(X,metric,transform='maxnorm',k=5):
    '''
    input: X is a data frame, metric is the name of one of the (cost) metrics that you want to scale between 0 and 1
            transform options include:
                :'maxnorm', which means dividing each value by maximum in list
                :'minmaxnorm', look at it
                :'sigmoid', which means passing each value through logistic function with mean
    output: X with additional column that has the rescaled metric
    '''
    if metric=='drawDuration': ## if handling drawDuration, log first -- no wait, maybe not 
        vals = X[metric].values
    else:
        vals = X[metric].values
    X['vals'] = vals
    if transform=='maxnorm':
        top_val = np.max(vals)
        rescaled_val = []
        for i,d in X.iterrows():
            rescaled_val.append(d['vals']/top_val)
    elif transform=='minmaxnorm':
        bottom_val = np.min(vals)
        top_val = np.max(vals)
        rescaled_val = []
        for i,d in X.iterrows():
            rescaled_val.append((d['vals']-bottom_val)/(top_val-bottom_val))        
    elif transform=='sigmoid':
        median_val = np.median(vals)
        rescaled_val = []
        for i,d in X.iterrows():
            rescaled_val.append(sigmoid(d['vals'],k=k,x0=median_val))
    X['rescaled_{}'.format(metric)] = rescaled_val          
    return X


# In[ ]:

## actually add rescaled metric
D2 = add_rescaled_metric(D2,'numStrokes',transform='minmaxnorm')
D2 = add_rescaled_metric(D2,'mean_intensity',transform='minmaxnorm')
D2 = add_rescaled_metric(D2,'drawDuration',transform='minmaxnorm')


# In[ ]:

## generate cost dictionaries to try out with pragmatics model
print len(np.unique(D2['sketchID'].values))
sketchID_list = np.unique(D2['sketchID'].values)
metrics = ['drawDuration','mean_intensity','numStrokes']

for metric in metrics:    
    print metric
    cost_json = {}
    for i,d in enumerate(sketchID_list):
        assert len(np.unique(D2[D2['sketchID']==d]['rescaled_{}'.format(metric)].values))==1
        cost_json[d] = D2[D2['sketchID']==d]['rescaled_{}'.format(metric)].values[0]    

    ## output json in the same format as the other cost json
    output_path = '../models/refModule/json/costs-fixedPose96-{}.json'.format(metric)
    with open(output_path, 'wb') as fp:
        json.dump(cost_json, fp)        


# ### generate condition-lookup.json to be able to pair sketches with condition

# In[ ]:

## generate condition-lookup.json to be able to pair sketches with condition
cond_json = {}
sketchID_list = np.unique(D2['sketchID'].values)
for i,d in enumerate(sketchID_list):
    cond = D2[D2['sketchID']==d]['condition'].values[0]
    obj = D2[D2['sketchID']==d]['target'].values[0]
    cond_json[d] = '{}_{}'.format(cond,obj)
    
## output json in the same format as the other cost json
output_path = '../models/bdaInput/condition-lookup.json'
with open(output_path, 'wb') as fp:
    json.dump(cond_json, fp)


# In[ ]:



