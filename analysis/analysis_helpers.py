from __future__ import division
import os
import json
import numpy as np
import re
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pylab, mlab, pyplot
from IPython.core.pylabtools import figsize, getfigs
plt = pyplot
import seaborn as sns
sns.set_context('poster')
sns.set_style('white')

exp_dir = './'

objcat = dict({'basset':'dog',
               'beetle':'car',
               'bloodhound':'dog',
               'bluejay':'bird',
               'bluesedan':'car',
               'bluesport':'car',
               'brown':'car',
               'bullmastiff':'dog',
               'chihuahua':'dog',
               'crow':'bird',
               'cuckoo':'bird',
               'doberman':'dog',
               'goldenretriever':'dog',
               'hatchback':'car',
               'inlay':'chair',
               'knob':'chair',
               'leather':'chair',
               'nightingale':'bird',
               'pigeon':'bird',
               'pug':'dog',
               'redantique':'car',
               'redsport':'car',
               'robin':'bird',
               'sling':'chair',
               'sparrow':'bird',
               'squat':'chair',
               'straight':'chair',
               'tomtit':'bird',
               'waiting':'chair',
               'weimaraner':'dog',
               'white':'car',
               'woven':'chair'
              })


def get_mean_by_condition_and_game(D,var='numStrokes'):
    '''
    Input: dataframe D and name of variable of interest (which is a column of D)
    Output: two vectors, one for close and one for far condition,
            with mean for each game
    '''

    d = D.groupby(['gameID','condition'])[var].mean().reset_index()
    far_d = d[d['condition']=='further'][var].values
    close_d = d[d['condition']=='closer'][var].values
    return far_d, close_d

    
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def sort_filelist(files):
    return files.sort(key=natural_keys)    
    
def get_close_accuracy_by_category(D, all_games):
    car_accuracy = []
    dog_accuracy = []    
    chair_accuracy = []
    bird_accuracy = []        
    for game in all_games:    
        car_accuracy.append(D[(D['category']=='car') & (D['condition']=='closer') & (D['gameID']== game) ]['outcome'].mean())
        dog_accuracy.append(D[(D['category']=='dog') & (D['condition']=='closer') & (D['gameID']== game) ]['outcome'].mean())     
        chair_accuracy.append(D[(D['category']=='chair') & (D['condition']=='closer') & (D['gameID']== game) ]['outcome'].mean())  
        bird_accuracy.append(D[(D['category']=='bird') & (D['condition']=='closer') & (D['gameID']== game) ]['outcome'].mean())   
    return bird_accuracy, car_accuracy, chair_accuracy, dog_accuracy
    
def get_canonical(category):    
    stimFile = os.path.join(exp_dir,'stimList_subord.js')
    with open(stimFile) as f:
        stimList = json.load(f)    
    allviews = [i['filename'] for i in stimList if i['basic']==category]
    canonical = [a for a in allviews if a[-8:]=='0035.png']    
    return canonical

def get_actual_pose(subordinate,pose):
    stimFile = os.path.join(exp_dir,'stimList_subord.js')
    with open(stimFile) as f:
        stimList = json.load(f)
    inpose = [i['filename'] for i in stimList if (i['subordinate']==subordinate) and (i['pose']==pose)]
    return inpose
    
def get_subord_names(category):
    full_names = get_canonical(category)    
    return [c.split('_')[2] for c in full_names]

def get_basic_names(subordinate):
    stimFile = os.path.join(exp_dir,'stimList_subord.js')
    with open(stimFile) as f:
        stimList = json.load(f)   
    allviews = [i['filename'] for i in stimList if i['subordinate']==subordinate]
    canonical = [a for a in allviews if a[-8:]=='0035.png']      
    return canonical[0].split('_')[0]

def build_url_from_category(category):
    full_names = get_canonical(category)
    url_prefix = 'https://s3.amazonaws.com/sketchloop-images-subord/'
    urls = []
    for f in full_names:
        urls.append(url_prefix + f)
    return urls

def build_url_from_filenames(filenames):
    url_prefix = 'https://s3.amazonaws.com/sketchloop-images-subord/'
    urls = []
    for f in filenames:
        urls.append(url_prefix + f)
    return urls

def plot_from_url(URL):
    file = cStringIO.StringIO(urllib.urlopen(URL).read())
    img = Image.open(file)    

def plot_gallery(category):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.figure(figsize = (8,8))
    gs1 = gridspec.GridSpec(8, 8)
    gs1.update(wspace=0.025, hspace=0.05)

    url_prefix = 'https://s3.amazonaws.com/sketchloop-images-subord/'
    for (i,c) in enumerate(category):
        URL = url_prefix + c
        file = cStringIO.StringIO(urllib.urlopen(URL).read())
        img = Image.open(file)
        p = plt.subplot(3,3,i+1)
        plt.imshow(img)
        p.get_xaxis().set_ticklabels([])
        p.get_yaxis().set_ticklabels([])
        p.get_xaxis().set_ticks([])
        p.get_yaxis().set_ticks([])
        p.set_aspect('equal')
        subord = c.split('_')[2]
        plt.title(subord)
    plt.tight_layout()

    
def softmax(x):
    denom = np.sum(map(np.exp,x))
    return [i/denom for i in x]
        
def get_superblock_bounds(M_shape,block_size):
    bounds = np.linspace(0,M_shape,M_shape/block_size+1)
    bound_tuples = []
    for i,j in enumerate(bounds[:len(bounds)-1]):
        bound_tuples.append(tuple((int(bounds[i]), np.int(bounds[i+1]))))
    return bound_tuples

    
def get_superblocks(M,num_cats):
    '''
    accepts: 
        square similarity matrix M of dim (m,m)
        num_cats: number of categories (dim of superblock is m/num_cats)
    '''
    M_shape = M.shape[0]
    block_size = M_shape/num_cats
    bound_tuples = get_superblock_bounds(M_shape,block_size)
    _M = []
    for block in bound_tuples:
        _M.append(M[block[0]:block[1]])    
    
###### MODEL COMPARISON HELPERS ######

def load_json(path):
    with open(path) as f:
        J = json.load(f)   
    return J

def sumlogprob(a,b):
    if (a > b):
        return a + np.log1p(np.exp(b-a))
    else:
        return b + np.log1p(np.exp(a-b))  
    
dogs = sorted(['weimaraner', 'chihuahua', 'basset', 'doberman', 'bloodhound', 'bullmastiff', 'goldenretriever', 'pug'])
chairs = sorted(['leather', 'straight', 'squat', 'sling', 'woven', 'waiting', 'inlay','knob'])
birds = sorted(['crow', 'pigeon', 'robin', 'sparrow', 'tomtit', 'nightingale', 'bluejay', 'cuckoo'])
cars = sorted(['beetle', 'bluesport', 'brown', 'white', 'redsport', 'redantique', 'hatchback', 'bluesedan'])

def flatten_mcmc_to_samples(raw_params,num_samples=1000):
    flat_params = pd.DataFrame(columns=raw_params.columns)
    counter = 0
    for i,d in raw_params.iterrows():
        multiples = int(np.round(np.exp(d['posteriorProb'])*num_samples))
        for m in np.arange(multiples):
            flat_params.loc[counter] = d
            counter += 1

    ## correct the posteriorProb column so that each sample has prob 1/num_samples, where num_samples prob is 1000
    flat_params.drop(labels=['posteriorProb'], axis="columns", inplace=True)
    flat_params['posteriorProb'] = np.tile(np.log(1/num_samples),len(flat_params))
    assert len(flat_params)==num_samples
    return flat_params 

def flatten(x):
    return [item for sublist in x for item in sublist]

def trim_outliers(x):
    mu = np.mean(x)
    sd = np.std(x)
    thresh = mu + sd*3
    y = [i for i in x if i<thresh]
    return y

def bootstrap(w,nIter=10000):
    boot = []
    for i in np.arange(nIter):
        boot.append(np.mean(np.random.RandomState(i).choice(w,len(w),replace=True)))
    boot = np.array(boot) 
    p1 = sum(boot<0)/len(boot) * 2
    p2 = sum(boot>0)/len(boot) * 2
    p = np.min([p1,p2])        
    lb = np.percentile(boot,2.5)
    ub = np.percentile(boot,97.5)
    return boot, p, lb, ub

def make_category_by_obj_palette():
    import itertools
    col = []
    for j in sns.color_palette("hls", 4):
        col.append([i for i in itertools.repeat(j, 8)])
    return flatten(col)

def model_comparison_bars(model_prefixes,adaptor_type='human',split_type='balancedavg'):
    '''
    loads in model param posterior by adaptor type
    '''
    all_param_paths = sorted(os.listdir('../models/bdaOutput/{}_{}/raw/'.format(adaptor_type,split_type)))
    model_zoo = [i for i in all_param_paths for pre in model_prefixes if pre in i]
    model_zoo = [i for i in model_zoo if i[-1] != '~']
    model_zoo = [i for i in model_zoo if '.csv' in i]
    model_zoo = [i for i in model_zoo if 'S1' not in i.split('_')] ## do not consider S1
    
#     assert len(model_zoo) == len(model_prefixes)*4
    
    import analysis_helpers as h
    reload(h)

    LL = []
    model_name = []
    for this_model in model_zoo:

        ## define paths to model predictions
        if adaptor_type=='human':
            model_dirname = ('_').join(this_model.split('_')[:3])
        else:
            model_dirname = ('_').join(this_model.split('_')[:4])

        ## get file with params from this model
        this_params = os.path.join('../models/bdaOutput/{}_{}/raw/'.format(adaptor_type,split_type),this_model)
        params = pd.read_csv(this_params)
        assert np.round(np.sum(np.exp(params.posteriorProb.values)),12)==1

        ## append MAP LL
        LL.append(params.sort_values(by=['logLikelihood'],ascending=False).iloc[0]['logLikelihood'])
        model_name.append(model_dirname) 
        
    ## make dataframe
    PP = pd.DataFrame.from_records(zip(model_name,LL))
    PP.columns=['model','logLikelihood']
    if adaptor_type=='human':
        PP['perception'], PP['pragmatics'], PP['production'] = PP['model'].str.split('_', 3).str
    else:
        PP['adaptor'],PP['perception'], PP['pragmatics'], PP['production'] = PP['model'].str.split('_', 3).str
    return PP        
    
    
def plot_human_bars(PP):
    sns.catplot(data=PP,x='pragmatics',y='logLikelihood',
                   hue='production',kind='bar',
                   order=['S0','combined'],
                   hue_order=['nocost','cost'],
                   palette='Paired',
                   legend=False,
                   ci=None)
    plt.ylabel('log likelihood')
    locs, labels = plt.xticks([0,1],['insensitive','sensitive'],fontsize=14)
    plt.xlabel('context')
    # plt.ylim([-3000,0])
    plt.tight_layout()
    plt.savefig('./plots/loglikelihood_models_human.pdf')
    # plt.close()  
    
def plot_multimodal_bars(PP):
    sns.catplot(data=PP,x='perception',y='logLikelihood',
                   hue='pragmatics',kind='bar',
                   order=['pool1','conv42','fc6'],
                   palette='Paired',
                   legend=False,
                   ci=None)
    plt.ylabel('log likelihood')
    locs, labels = plt.xticks([0,1,2],['early','mid','high'],fontsize=14)
    plt.xlabel('visual features')
    # plt.ylim([-3000,0])
    plt.tight_layout()
    plt.savefig('./plots/loglikelihood_models_multimodal.pdf')
    # plt.close()      
    
def flatten_mcmc_to_samples(raw_params,num_samples=1000):
    flat_params = pd.DataFrame(columns=raw_params.columns)
    counter = 0
    for i,d in raw_params.iterrows():
        multiples = int(np.round(np.exp(d['posteriorProb'])*num_samples))
        for m in np.arange(multiples):
            flat_params.loc[counter] = d
            counter += 1

    ## correct the posteriorProb column so that each sample has prob 1/num_samples, where num_samples prob is 1000
    flat_params.drop(labels=['posteriorProb'], axis="columns", inplace=True)
    flat_params['posteriorProb'] = np.tile(np.log(1/num_samples),len(flat_params))
    assert len(flat_params)==num_samples
    return flat_params    

def check_mean_LL_for_cost_vs_nocost(model_prefixes=['multimodal_fc6'],
                                     adaptor_type = 'multimodal_fc6',
                                     split_type='balancedavg1',
                                     plot=True):
    
    all_param_paths = sorted(os.listdir('../models/bdaOutput/{}_{}/raw'.format(adaptor_type,split_type)))
    model_zoo = [i for i in all_param_paths for pre in model_prefixes if pre in i]
    model_zoo = [i for i in model_zoo if i[-1] != '~']
    model_zoo = [i for i in model_zoo if '.csv' in i]
    # model_zoo = [i for i in model_zoo if 'S1' not in i.split('_')] ## do not consider S1

    # assert len(model_zoo) == len(model_prefixes)*6

    ## get file with params from this model
    this_params = os.path.join('../models/bdaOutput/{}_{}'.format(adaptor_type,split_type),'raw',model_zoo[4])
    params1 = pd.read_csv(this_params)

    this_params = os.path.join('../models/bdaOutput/{}_{}'.format(adaptor_type,split_type),'raw',model_zoo[5])
    params2 = pd.read_csv(this_params)

    print 'Hold tight, running this check takes a little while...'
    ## "flatten" params file so that we have all 1000 samples in the params file itself
    fparams1 = flatten_mcmc_to_samples(params1)
    fparams2 = flatten_mcmc_to_samples(params2)
    fparams1.reset_index(inplace=True,drop=True)
    fparams2.reset_index(inplace=True,drop=True)

    print '{} cost version mean LL: {}'.format(adaptor_type, np.mean(fparams1.logLikelihood.values))
    print '{} nocost version mean LL: {}'.format(adaptor_type, np.mean(fparams2.logLikelihood.values))
    
    if plot==True:
        
        ## plot LL distribution comparing cost and nocost verisons
        plt.figure(figsize=(8,4))
        plt.subplot(121)
        h = sns.distplot(fparams1.logLikelihood.values,color=(0.6,0.2,0.2),label='cost')
        h = sns.distplot(fparams2.logLikelihood.values,color=(0.9,0.6,0.6),label='nocost')
        plt.xlabel('loglikelihood')
        # plt.xlim(-650,-400)
        plt.title(split_type)
        plt.legend()

        ## plot cost weight distribution
        plt.subplot(122)
        h = sns.distplot(fparams1.costWeight.values,color=(0.6,0.2,0.2))
        plt.title('cost param posterior')
        plt.xlabel('cost weight')
        
def flatten_param_posterior(adaptor_types = ['multimodal_pool1','multimodal_conv42','multimodal_fc6', 'human'],
                            verbosity=1):
    '''
    "flattening" means making sure that we have 1000 rows in each of the param posterior files, 
    corresponding to each sample. In "raw" form, there may be fewer than 1000 samples, b/c some samples
    might just be associated with higher posteriorProb.        
    '''
    model_prefixes = adaptor_types
    split_types = ['balancedavg{}'.format(i) for i in map(str,np.arange(1,6))]

    for adaptor_type in adaptor_types:
        if verbosity==1:
            print 'Flattening all splits and models of adaptor type {}...'.format(adaptor_type)        
        for split_type in split_types:
            if verbosity==1:
                print 'Now flattening models in split {}'.format(split_type)                         
            all_param_paths = sorted(os.listdir('../models/bdaOutput/{}_{}/raw'.format(adaptor_type,split_type)))
            model_zoo = [i for i in all_param_paths for pre in model_prefixes if pre in i]
            model_zoo = [i for i in model_zoo if i[-1] != '~']
            model_zoo = [i for i in model_zoo if '.csv' in i]    

            for i,model in enumerate(model_zoo):
                if verbosity>1:
                    print 'flattening {}'.format(model)
                ## get file with params from this model
                this_params = os.path.join('../models/bdaOutput/{}_{}'.format(adaptor_type,split_type),'raw',model)
                params = pd.read_csv(this_params)

                ## "flatten" params file so that we have all 1000 samples in the params file itself
                fparams = flatten_mcmc_to_samples(params)
                fparams.reset_index(inplace=True,drop=True)
                fparams = fparams.rename(columns={'id':'sample_id'}) ## rename id column to be sample id
                fparams = fparams.reindex(fparams.index.rename('id')) ## rename index column to be id column

                ## write out
                out_path = os.path.join('../models/bdaOutput/{}_{}'.format(adaptor_type,split_type),'flattened',model.split('.')[0] + 'Flattened.csv')
                if not os.path.exists(os.path.join('../models/bdaOutput/{}_{}'.format(adaptor_type,split_type),'flattened')):
                    os.makedirs(os.path.join('../models/bdaOutput/{}_{}'.format(adaptor_type,split_type),'flattened'))
                if verbosity>1:
                    print 'out_path = {}'.format(out_path)
                fparams.to_csv(out_path)
            
def get_sense_for_param_range_across_splits():
    
    '''
    Before running bda-enumerate in order to do model comparison, wanted to get a sense of the range that
    the various params fell into in the posterior, to ensure that our enumerate mesh captures this range for 
    all splits.
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

    # model_space = ['multimodal_fc6_combined_cost']

    # ## define paths to model predictions
    # split_type = 'balancedavg1'
    # model = 'multimodal_conv42_combined_cost'

    for model in model_space:
        print ' '
        for split_type in split_types:

            path_to_evaluate = '/data5/jefan/sketchpad_basic_model_output/evaluateOutput/{}_{}'.format(model,split_type)
            pred_files = [os.path.join(path_to_evaluate,i) for i in os.listdir(path_to_evaluate)]

            ## get file with params from this model
            if model.split('_')[0]=='human':
                bdaOutDir = '_'.join(model.split('_')[:1]) + '_{}'.format(split_type)
            else:
                bdaOutDir = '_'.join(model.split('_')[:2]) + '_{}'.format(split_type)
            params_fname = model + '_' + split_type + 'ParamsFlattened.csv'
            params_path = os.path.join('../models/bdaOutput',bdaOutDir,'flattened',params_fname)
            params = pd.read_csv(params_path)

            maxSim = np.max(params.simScaling.values)
            maxPrag = np.max(params.pragWeight.values)
            maxCost = np.max(params.costWeight.values)
            maxInf = np.max(params.infWeight.values)

            print 'model {} split {}'.format(model,split_type)
            print 'max | sim: {} prag: {} cost: {} inf: {}'.format(maxSim,maxPrag,maxCost,maxInf) 
            
def weight_cost_by_modelProb(x):
    '''
    in order to determine the average sketch cost predicted by the model for this trial,
    take mean cost across all sketch categories weighted by the probability assigned to that sketch category
    
    note: modelProb is in log space, so you need to exponentiate before multiplying by cost
    '''
    return np.exp(x['modelProb']) * x['cost']    


def convert_numeric(X):
    ## make numeric types for aggregation
    X['target_rank'] = pd.to_numeric(X['target_rank'])
    X['foil_rank'] = pd.to_numeric(X['foil_rank'])
    X['logprob'] = pd.to_numeric(X['logprob'])
    return X

def add_diff_rank(X):
    X['diff_rank'] = X['target_rank'] - X['foil_rank']
    X['sign_diff_rank'] = X['diff_rank']<0
    return X

def load_model_predictions(model='human_combined_cost',
                           split_type='balancedavg1'):
    model_preds = pd.read_csv('./csv/{}_{}_predictions.csv'.format(model,split_type))
    model_preds = add_diff_rank(convert_numeric(model_preds))
    if 'Unnamed: 0' in model_preds.columns:
        model_preds.drop(columns=['Unnamed: 0'],inplace=True)    
    
    return model_preds
    
def load_all_model_preds(split_types = ['balancedavg1','balancedavg2','balancedavg3','balancedavg4','balancedavg5'],
                         model_space = ['human_combined_cost',
                                        'human_S0_cost',
                                        'human_combined_nocost',
                                        'multimodal_fc6_combined_cost',
                                        'multimodal_fc6_S0_cost',
                                        'multimodal_fc6_combined_nocost',
                                        'multimodal_conv42_combined_cost',
                                        'multimodal_pool1_combined_cost'],
                         verbosity=2):      
    
    '''
    Load all model predictions from all five splits into a dictionary called P.
    P is a nested dictionary containing all predictions dataframes for all five primary models of interest and five splits.    
    P.keys() = ['multimodal_conv42_combined_cost', 'human_combined_cost', 'multimodal_fc6_combined_cost', 'multimodal_fc6_S0_cost', 'multimodal_fc6_combined_nocost']
    Nested inside each model are dataframes containing model predictions from each split.    
    '''
    P = {}
    for model in model_space:
        if verbosity>=1:
            print 'Loading model preds from: {}'.format(model)
        P[model] = {}    
        for split_type in split_types:
            if verbosity >=2:
                print 'Loading split {}'.format(split_type)
            preds = load_model_predictions(model=model,
                                       split_type=split_type)

            P[model][split_type] = preds 
            
    return P

def get_convenient_handles_on_model_preds(P,split_type='balancedavg1'):
    H = P['human_combined_cost'][split_type]
    H0 = P['human_S0_cost'][split_type]
    H1 = P['human_combined_nocost'][split_type]    
    M = P['multimodal_fc6_combined_cost'][split_type]
    M0 = P['multimodal_conv42_combined_cost'][split_type]
    M1 = P['multimodal_fc6_S0_cost'][split_type]
    M2 = P['multimodal_fc6_combined_nocost'][split_type]
    M3 = P['multimodal_pool1_combined_cost'][split_type]
    return H,H0,H1,M,M0,M1,M2,M3

def load_and_check_bootstrapped_model_preds(results_dir = './bootstrap_results'):
    ## get how many boot files there are
    path_to_bootstrap_results = results_dir
    boot_files = os.listdir(path_to_bootstrap_results)
    print 'There are {} files in the bootstrap_results directory.'.format(len(boot_files))

    ## ground truth list of how many bootvec file we *should* have
    split_types = ['balancedavg1','balancedavg2','balancedavg3','balancedavg4','balancedavg5']
    model_space = ['human_combined_cost','human_S0_cost','human_combined_nocost',\
                  'multimodal_fc6_combined_cost', \
                  'multimodal_fc6_S0_cost','multimodal_fc6_combined_nocost',
                  'multimodal_conv42_combined_cost',\
                  'multimodal_pool1_combined_cost']
    conditions = ['all','closer','further']
    vois = ['target_rank','foil_rank','sign_diff_rank','cost']
    nIter = 1000

    full_path_list = []
    for split_type in split_types:
        for model in model_space:
            for condition in conditions:
                for var_of_interest in vois:
                    out_path = 'bootvec_{}_{}_{}_{}_{}.npy'.format(model,split_type,var_of_interest,condition,nIter)
                    full_path_list.append(out_path)

    ## check to make sure that there are no boot filenames that *should* be in the list but are not
    if len([i for i in full_path_list if i not in boot_files])>0:
        print 'Not all of the bootstrap results are available. Please try running: python get_all_bootstrapped_model_predictions.py again.'
        print 'These are the missing files: '
        for i in full_path_list:
            if i not in boot_files:
                print i
                   
    return boot_files    


def generate_bootstrap_model_preds_dataframe(boot_files, out_dir='./bootstrap_results'):
    '''
    Input: list of boot_file paths containing .npy with bootstrapped sampling distributions of each variable of interest.
    Output: dataframe collecting these bootstrapped data with associated metadata, including: 
            model names, split number, condition, variable of interest, bootstrap filename, 
            and the bootstrapped distribution itself.
    '''
    model_list = []
    split_type_list = []
    condition_list = []
    var_of_interest_list = []
    fname_list = []
    bootvec_list = []
    for i,bf in enumerate(boot_files):
        if bf.split('_')[1]=='human':
            divider_ind = 4
        else:
            divider_ind = 5
        ## divider_ind distinguishes between human/multimodal vision encoder names, which differ in length
        model = '_'.join(bf.split('_')[1:divider_ind])
        split_type = bf.split('_')[divider_ind]
        remainder = bf.split('_')[divider_ind+1:]

        condition = remainder[-2]
        var_of_interest = '_'.join(remainder[:len(remainder)-2]) 
        bootvec = np.load(os.path.join(out_dir,bf))

        model_list.append(model)
        split_type_list.append(split_type)
        condition_list.append(condition)
        var_of_interest_list.append(var_of_interest)
        fname_list.append(os.path.join(out_dir,bf))
        bootvec_list.append(bootvec)

    ## construct dataframe containing all bootvecs and associated metadata    
    X = pd.DataFrame([model_list,split_type_list,condition_list,var_of_interest_list,fname_list,bootvec_list])
    X = X.transpose()
    X.columns = ['model','split_type','condition','var_of_interest','fname','bootvec']
    assert len(np.unique(X.condition.values))==3 
    return X





def plot_target_vs_foil_rank_by_object(P,split_type='balancedavg1'):
    '''
    What is the rank of the correct sketch category (correct object + correct context) 
    vs. wrong sketch category (correct object + wrong context)?
    '''     
    
    H,H0,H1,M,M0,M1,M2,M3 = get_convenient_handles_on_model_preds(P,split_type=split_type)

    fig = plt.figure(figsize=(20,14))
    plt.subplot(241)
    targ = pd.DataFrame(H.groupby(['object'])['target_rank'].mean())['target_rank'].values
    foil = pd.DataFrame(H.groupby(['object'])['foil_rank'].mean())['foil_rank'].values
    h = plt.scatter(targ,foil,s=24)
    plt.plot([1,14],[1,14],color='k',linestyle='dashed')
    plt.xlim([1,14])
    plt.ylim([1,14])
    plt.xlabel('target rank')
    plt.ylabel('foil rank')
    plt.title('human')
    plt.subplot(243)
    targ = pd.DataFrame(H0.groupby(['object'])['target_rank'].mean())['target_rank'].values
    foil = pd.DataFrame(H0.groupby(['object'])['foil_rank'].mean())['foil_rank'].values
    h = plt.scatter(targ,foil,s=24)
    plt.plot([1,14],[1,14],color='k',linestyle='dashed')
    plt.xlim([1,14])
    plt.ylim([1,14])
    plt.xlabel('target rank')
    plt.ylabel('foil rank')
    plt.title('human S0') 
    plt.subplot(244)
    targ = pd.DataFrame(H1.groupby(['object'])['target_rank'].mean())['target_rank'].values
    foil = pd.DataFrame(H1.groupby(['object'])['foil_rank'].mean())['foil_rank'].values
    h = plt.scatter(targ,foil,s=24)
    plt.plot([1,14],[1,14],color='k',linestyle='dashed')
    plt.xlim([1,14])
    plt.ylim([1,14])
    plt.xlabel('target rank')
    plt.ylabel('foil rank')
    plt.title('human nocost')              
    plt.subplot(245)
    targ = pd.DataFrame(M.groupby(['object'])['target_rank'].mean())['target_rank'].values
    foil = pd.DataFrame(M.groupby(['object'])['foil_rank'].mean())['foil_rank'].values
    h = plt.scatter(targ,foil,s=24)
    plt.plot([1,14],[1,14],color='k',linestyle='dashed')
    plt.xlim([1,14])
    plt.ylim([1,14])
    plt.xlabel('target rank')
    plt.ylabel('foil rank')
    plt.title('model')
    plt.subplot(246)
    targ = pd.DataFrame(M0.groupby(['object'])['target_rank'].mean())['target_rank'].values
    foil = pd.DataFrame(M0.groupby(['object'])['foil_rank'].mean())['foil_rank'].values
    h = plt.scatter(targ,foil,s=24)
    plt.plot([1,14],[1,14],color='k',linestyle='dashed')
    plt.xlim([1,14])
    plt.ylim([1,14])
    plt.xlabel('target rank')
    plt.ylabel('foil rank')
    plt.title('model conv42')
    plt.subplot(247)
    targ = pd.DataFrame(M1.groupby(['object'])['target_rank'].mean())['target_rank'].values
    foil = pd.DataFrame(M1.groupby(['object'])['foil_rank'].mean())['foil_rank'].values
    h = plt.scatter(targ,foil,s=24)
    plt.plot([1,14],[1,14],color='k',linestyle='dashed')
    plt.xlim([1,14])
    plt.ylim([1,14])
    plt.xlabel('target rank')
    plt.ylabel('foil rank')
    plt.title('model S0')
    plt.subplot(248)
    targ = pd.DataFrame(M2.groupby(['object'])['target_rank'].mean())['target_rank'].values
    foil = pd.DataFrame(M2.groupby(['object'])['foil_rank'].mean())['foil_rank'].values
    h = plt.scatter(targ,foil,s=24)
    plt.plot([1,14],[1,14],color='k',linestyle='dashed')
    plt.xlim([1,14])
    plt.ylim([1,14])
    plt.xlabel('target rank')
    plt.ylabel('foil rank')
    plt.title('model nocost')
    plt.tight_layout()
    
def get_avg_rank_across_samples(X):
    '''
    Input: X is a summary dataframe of model predictions, e.g., ones named H,M,M0,M1,M2 in notebook
           And loaded using function: P = h.load_all_model_preds()
    make another dataframe which computes, for each MCMC sample, the average rank of the target
    (congruent context category)
    '''
    XM = pd.DataFrame(X.groupby('sample_ind')['target_rank'].mean()).reset_index()
    adaptor = np.unique(X['adaptor'].values)[0]
    adaptor = list(np.tile(adaptor,len(XM)))
    XM = XM.assign(adaptor=pd.Series(adaptor).values)    
    return XM  

def get_avg_rank_all_models(P,split_type='balancedavg1'):
    H,H0,H1,M,M0,M1,M2,M3 = get_convenient_handles_on_model_preds(P,split_type=split_type)
    HU,H0U,H1U,MU,M0U,M1U,M2U,M3U = map(get_avg_rank_across_samples,[H,H0,H1,M,M0,M1,M2,M3])
    return HU,H0U,H1U,MU,M0U,M1U,M2U,M3U

def plot_avg_rank_all_models(P,split_type='balancedavg1',saveout=True):
    '''
    Generate bar plot of average rank (out of 64) of correct sketch category, by model, for a particular split.
    Wrapper around get_avg_rank_all_models, which itself wraps around get_avg_rank_across_samples.
    '''
    HU,H0U,H1U,MU,M0U,M1U,M2U,M3U = get_avg_rank_all_models(P,split_type=split_type)
    sns.set_context('talk')
    sns.set_style("ticks")
    fig = plt.figure(figsize=(4,8))
    ax = fig.add_subplot(111)
    U = pd.concat([HU,H0U,H1U,MU,M0U,M1U,M2U,M3U],axis=0)
    sns.barplot(data=U,
                x='adaptor',
                y='target_rank',
                ci='sd',
                order = ['human_combined_cost','human_S0_cost','human_combined_nocost',\
                         'multimodal_fc6_combined_cost', \
                         'multimodal_fc6_S0_cost','multimodal_fc6_combined_nocost',
                         'multimodal_conv42_combined_cost',\
                         'multimodal_pool1_combined_cost'])
    plt.ylabel('mean rank of congruent sketch')
    plt.ylim([1,32])
    xticklabels=['Context Cost Human','NoContext Cost Human','Context NoCost Human','Context Cost HighAdaptor',
                 'NoContext Cost HighAdaptor','Context NoCost HighAdaptor', 'Context Cost MidAdaptor',\
                 'Context Cost LowAdaptor']
    plt.xlabel('')
    l = ax.set_xticklabels(xticklabels, rotation = 90, ha="left")
    plt.tight_layout()
    if saveout:
        plt.savefig('./plots/avg_rank_all_models_{}.pdf'.format(split_type))

    
    
def get_prop_congruent(X):
    '''
    make another data frame that computes, for each MCMC sample, the proportion of trials 
    on which model assigns better rank to congruent sketch than sketch from opposite context category ('foil')
    '''
    XM = pd.DataFrame(X.groupby('sample_ind')['sign_diff_rank'].apply(lambda x: sum(x)/len(x))).reset_index()
    adaptor = np.unique(X['adaptor'].values)[0]
    adaptor = list(np.tile(adaptor,len(XM)))
    XM = XM.assign(adaptor=pd.Series(adaptor).values)  
    return XM

def get_prop_congruent_all_models(P, split_type='balancedavg1'):
    '''
    Apply get_prog_congruent to all models
    '''
    H,H0,H1,M,M0,M1,M2,M3 = get_convenient_handles_on_model_preds(P,split_type=split_type)
    HU,H0U,H1U,MU,M0U,M1U,M2U,M3U = map(get_prop_congruent,[H,H0,H1,M,M0,M1,M2,M3])
    return HU,H0U,H1U,MU,M0U,M1U,M2U,M3U

def plot_prop_congruent_all_models(P,split_type='balancedavg1',saveout=True):
    '''
    Generate bar plot of proportion of trials for which context-congruent sketch preferred over incongruent sketch.
    Wrapper around get_prop_congruent_all_models, which itself wraps around get_prop_congruent.
    '''
    HU,H0U,H1U,MU,M0U,M1U,M2U,M3U = get_prop_congruent_all_models(P,split_type=split_type)
    sns.set_context('talk')
    fig = plt.figure(figsize=(4,8))
    ax = fig.add_subplot(111)     
    D = pd.concat([HU,H0U,H1U,MU,M0U,M1U,M2U,M3U],axis=0)    
    sns.barplot(data=D,
                x='adaptor',
                y='sign_diff_rank',ci='sd')
    plt.axhline(y=0.5,linestyle='dashed',color='k')
    plt.ylim([0,1])
    plt.ylabel('proportion context-congruent sketch preferred')

    xticklabels=['Context Cost Human','NoContext Cost Human','Context NoCost Human','Context Cost HighAdaptor',
                 'NoContext Cost HighAdaptor','Context NoCost HighAdaptor', 'Context Cost MidAdaptor',\
                 'Context Cost LowAdaptor']
    plt.xlabel('')
    l = ax.set_xticklabels(xticklabels, rotation = 90, ha="left")
    plt.tight_layout()
    if saveout:
        plt.savefig('./plots/prop_congruent_all_models_{}.pdf'.format(split_type))
        
    

def get_top_k_predictions(P, split_type='balancedavg1',verbosity=1):
    '''
    Save out CSV files containing, for various levels of k, the proportion of trials for which
    the target rank was less than or equal to k. 
    '''

    model_space = P.keys()

    for j,model in enumerate(model_space):
        if verbosity>=1:
            print 'Currently extracting topk predictions for model: {}'.format(model)
        D = P[model][split_type]   

        sample_inds = np.unique(D.sample_ind.values)
        prop = []
        sid = []
        K = []
        for i,sample_ind in enumerate(sample_inds):
            if (i%250==0) & (verbosity>=2):                
                print 'evaluating {}'.format(i)                  
            these_rows = D[D['sample_ind']==sample_ind]
            num_trials = these_rows.shape[0]
            for k in np.arange(1,65):
                prop.append(sum(these_rows['target_rank']<=k)/num_trials)
                sid.append(sample_ind)
                K.append(k)

        ## make dataframe and save out
        sid = map(int,sid)
        K = map(int,K)
        adaptor = list(np.tile(model,len(sid)))
        Q = pd.DataFrame([prop,sid,K,adaptor])
        Q = Q.transpose()
        Q.columns = ['prop','ind','k','adaptor']
        print 'Saving out topk prediction file for: {} {}'.format(model, split_type)
        Q.to_csv('./csv/{}_{}_topk.csv'.format(model,split_type),index=False)  
    if verbosity>=1:
        print 'Finished saving out all topk prediction files.'
        
def load_all_topk_predictions():
    try:
        QH = pd.read_csv('./csv/human_combined_cost_balancedavg1_topk.csv')
        QH0 = pd.read_csv('./csv/human_S0_cost_balancedavg1_topk.csv')
        QH1 = pd.read_csv('./csv/human_combined_nocost_balancedavg1_topk.csv')        
        QM = pd.read_csv('./csv/multimodal_fc6_combined_cost_balancedavg1_topk.csv')
        QM0 = pd.read_csv('./csv/multimodal_conv42_combined_cost_balancedavg1_topk.csv')
        QM1 = pd.read_csv('./csv/multimodal_fc6_S0_cost_balancedavg1_topk.csv')
        QM2 = pd.read_csv('./csv/multimodal_fc6_combined_nocost_balancedavg1_topk.csv')
        QM3 = pd.read_csv('./csv/multimodal_pool1_combined_cost_balancedavg1_topk.csv')        
        Q = pd.concat([QH,QH0,QH1,QM,QM0,QM1,QM2,QM3],axis=0)
    except Exception as e: 
        print 'Make sure that you have already run get_top_k_predictions.'
        print(e)
        Q = []

    return Q

def plot_topk_all_models():
    '''
    Generate line plot that visualizes, for various values of k, the proportion of trials
    for which the model assigned the correct sketch category a rank of <= k.
    '''
    Q = load_all_topk_predictions()
    krange = 64 ## how many values of k to plot
    sns.set_context('poster')
    fig = plt.figure(figsize=(8,8))
    colors = [(0.2,0.2,0.2),(0.8,0.3,0.3),(0.3,0.3,0.8),(0.5,0.5,0.5),(0.6,0.2,0.6)]
    sns.pointplot(x='k',
                  y='prop',
                  hue='adaptor',
                  data=Q,
                  palette=colors,
                  markers = '.',
                  ci='sd',              
                  join=True)
    plt.ylabel('proportion',fontsize=24)
    plt.xlabel('k',fontsize=24)
    plt.title('% correct within top k')
    plt.ylim([0,1.1])
    # plt.xlim([-0.1,krange])
    plt.xlim([0,18])
    # locs, labels = plt.xticks(np.linspace(0,krange-1,9),map(int,np.linspace(0,krange-1,9)+1),fontsize=16)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.0, 0.9))
    
def get_avg_cost_across_samples(X):
    '''
    make another dataframe which computes, for each MCMC sample, the average sketch cost produced
    '''
    XM = pd.DataFrame(X.groupby(['sample_ind','condition'])['cost'].mean()).reset_index()
    adaptor = np.unique(X['adaptor'].values)[0]
    adaptor = list(np.tile(adaptor,len(XM)))
    XM = XM.assign(adaptor=pd.Series(adaptor).values)    
    return XM
        
def get_avg_cost_all_models(P, split_type='balancedavg1'):
    '''
    Apply get_avg_cost_across_samples to all models
    '''
    H,H0,H1,M,M0,M1,M2,M3 = get_convenient_handles_on_model_preds(P,split_type=split_type)
    HU,H0U,H1U,MU,M0U,M1U,M2U,M3U = map(get_avg_cost_across_samples,[H,H0,H1,M,M0,M1,M2,M3])
    return HU,H0U,H1U,MU,M0U,M1U,M2U,M3U

def generate_aggregated_estimate_dataframe(B, 
                                           condition_list = ['all'],
                                           model_space = ['human_combined_cost','human_S0_cost','human_combined_nocost',\
                                                          'multimodal_fc6_combined_cost', \
                                                          'multimodal_fc6_S0_cost','multimodal_fc6_combined_nocost',\
                                                          'multimodal_conv42_combined_cost',
                                                          'multimodal_pool1_combined_cost'],
                                           split_types = ['balancedavg1','balancedavg2',\
                                                          'balancedavg3','balancedavg4','balancedavg5'],
                                           var_of_interest='target_rank',
                                           condition='all'):    
    
    TR = B[(B['var_of_interest'] == var_of_interest) & (B['condition'].isin(condition_list))]

    joint_mu = []
    joint_var = []
    joint_sd = []
    joint_model_list = []
    joint_condition_list = []
    joint_split_list = []
    for this_model in model_space:
        TRM = TR[TR['model']==this_model] 
        ## initialize agg_mu and agg_var
        agg_mu = []
        agg_var = []
        for this_condition in condition_list:
            for this_split in split_types:
                TRMS = TRM[(TRM['split_type']==this_split) & (TRM['condition']==this_condition)]
                this_boot = TRMS.bootvec.values[0]
                split_mu = np.mean(this_boot)
                split_sd = np.std(this_boot)
                split_var = np.var(this_boot)
                agg_mu.append(split_mu)
                agg_var.append(split_var)

            ## apply inverse-weighting of variance to get combined mean, se (which is joint_sd)
            ## https://en.wikipedia.org/wiki/Inverse-variance_weighting
            w = [1/v for v in agg_var] ## weights
            wX = np.sum([i*j for (i,j) in zip(w,agg_mu)]) ## numerator of weighted mean
            W = np.sum(w) ## denom of weighted mean
            joint_mu.append(wX/W) ## weighted average

            ## get combined estimate of variance
            inv_var = [1/v for v in agg_var]
            sum_inv_var = np.sum(inv_var)
            joint_var.append(1/sum_inv_var)
            joint_sd.append(np.sqrt(1/sum_inv_var))

            ## metadata
            joint_model_list.append(this_model)
            joint_condition_list.append(this_condition)

    ## bundle into dataframe        
    sort_inds = list(np.repeat([0,1,2,3,4,5,6],len(condition_list))) ## to plot models in a nice order    
    R = pd.DataFrame([joint_mu,joint_sd,joint_model_list,joint_condition_list,sort_inds])
    R = R.transpose()
    R.columns=['mu','sd','model','condition','sort_inds']
    R.sort_values(by=['sort_inds','condition'],inplace=True)     

    return R

def plot_average_target_rank_across_splits(R,
                                           var_of_interest='target_rank',
                                           condition_list = ['all'],
                                           model_space = ['human_combined_cost','human_S0_cost','human_combined_nocost',\
                                                          'multimodal_fc6_combined_cost', \
                                                          'multimodal_fc6_S0_cost','multimodal_fc6_combined_nocost',\
                                                          'multimodal_conv42_combined_cost',
                                                          'multimodal_pool1_combined_cost'],
                                           split_types = ['balancedavg1','balancedavg2',\
                                                          'balancedavg3','balancedavg4','balancedavg5'],
                                           condition='all',
                                           sns_context='talk',
                                           figsize=(6,6),
                                           errbar_multiplier=1,
                                           ylabel='avg sketch cost',
                                           saveout=True):

    '''
    bar plot of average target_rank, aggregating across splits
    '''    
    sns.set_context(sns_context)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)  
    sns.barplot(x='model',
                y='mu',
                order=['human_combined_cost','human_S0_cost','human_combined_nocost',\
                       'multimodal_fc6_combined_cost',\
                       'multimodal_fc6_S0_cost','multimodal_fc6_combined_nocost',\
                       'multimodal_conv42_combined_cost',
                       'multimodal_pool1_combined_cost'],
                ci=None,
                data=R)

    ## plot custom error bars
    x = np.arange(len(R['mu'].values))
    y = R['mu']
    plt.errorbar(x,
                 y,
                 yerr=R['sd']*errbar_multiplier,
                 ecolor='black',
                 linestyle='',
                 linewidth=4,
                 capsize=0)

    plt.ylabel(ylabel)
    plt.ylim(0,34)
    ax.yaxis.set_ticks(np.arange(0, 36, 4))
    plt.axhline(y=32,linestyle='dashed',color='k')
    xticklabels=['Context Cost Human','NoContext Cost Human','Context NoCost Human','Context Cost HighAdaptor',
                 'NoContext Cost HighAdaptor','Context NoCost HighAdaptor', 'Context Cost MidAdaptor',\
                 'Context Cost LowAdaptor']
    plt.xlabel('')
    l = ax.set_xticklabels(xticklabels, rotation = 90, ha="left")
    plt.tight_layout()
    if saveout:
        plt.savefig('./plots/average_target_rank_across_splits.pdf')

def plot_prop_congruent_across_splits(R,
                                      var_of_interest='sign_diff_rank',
                                      condition_list = ['all'],
                                      model_space = ['human_combined_cost','human_S0_cost','human_combined_nocost',\
                                                     'multimodal_fc6_combined_cost', 
                                                     'multimodal_fc6_S0_cost','multimodal_fc6_combined_nocost',
                                                     'multimodal_conv42_combined_cost',\
                                                     'multimodal_pool1_combined_cost'],
                                      split_types = ['balancedavg1','balancedavg2',\
                                                        'balancedavg3','balancedavg4','balancedavg5'],
                                      condition='all',
                                      sns_context='talk',
                                      figsize=(6,6),
                                      errbar_multiplier=1.,
                                      ylabel='avg sketch cost',
                                      saveout=True):

    sns.set_context(sns_context)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)  
    sns.barplot(x='model',
                y='mu',
                order = ['human_combined_cost','human_S0_cost','human_combined_nocost',\
                         'multimodal_fc6_combined_cost', \
                         'multimodal_fc6_S0_cost','multimodal_fc6_combined_nocost',\
                         'multimodal_conv42_combined_cost',
                         'multimodal_pool1_combined_cost'],
                ci=None,
                data=R)

    ## plot custom error bars
    x = np.arange(len(R['mu'].values))
    y = R['mu']
    plt.errorbar(x,
                 y,
                 yerr=R['sd']*errbar_multiplier,
                 ecolor='black',
                 linestyle='',
                 linewidth=4,
                 capsize=0)

    plt.ylabel(ylabel)
    plt.axhline(y=0.5,linestyle='dashed',color='k')
    plt.ylim(0,0.8)

    xticklabels=['Context Cost Human','NoContext Cost Human','Context NoCost Human','Context Cost HighAdaptor',
                 'NoContext Cost HighAdaptor','Context NoCost HighAdaptor', 'Context Cost MidAdaptor',
                 'Context Cost LowAdaptor']
    plt.xlabel('')
    l = ax.set_xticklabels(xticklabels, rotation = 90, ha="left") 
    plt.tight_layout()
    if saveout:
        plt.savefig('./plots/prop_congruent_across_splits.pdf')
    
def plot_cost_by_condition_across_splits(R,
                                        var_of_interest='cost',
                                        condition_list = ['closer','further'],
                                        model_space = ['human_combined_cost','human_S0_cost','human_combined_nocost',\
                                                     'multimodal_fc6_combined_cost',\
                                                     'multimodal_fc6_S0_cost','multimodal_fc6_combined_nocost',\
                                                     'multimodal_conv42_combined_cost',
                                                     'multimodal_pool1_combined_cost'],
                                        split_types = ['balancedavg1','balancedavg2',\
                                                     'balancedavg3','balancedavg4','balancedavg5'],
                                        condition='all',
                                        sns_context='talk',
                                        figsize=(6,6),
                                        errbar_multiplier=1.,
                                        ylabel='avg sketch cost',
                                        saveout=True):

    sns.set_context(sns_context)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)  
    sns.barplot(x='model',
                y='mu',
                hue='condition',
                ci=None,
                data=R)

    ## plot custom error bars
    x_inds = []
    offset=1/5
    for i in np.arange(8):    
        x_inds.append(i-offset)
        x_inds.append(i+offset)
    x = x_inds
    y = R['mu']
    plt.errorbar(x,
                 y,
                 yerr=R['sd']*errbar_multiplier,
                 ecolor='black',
                 linestyle='',
                 linewidth=4,
                 capsize=0)

    plt.ylabel(ylabel)
    plt.ylim(0,0.3)

    xticklabels=['Context Cost Human','NoContext Cost Human','Context NoCost Human','Context Cost HighAdaptor',
                 'NoContext Cost HighAdaptor','Context NoCost HighAdaptor', 'Context Cost MidAdaptor',\
                 'Context Cost LowAdaptor']
    plt.xlabel('')
    l = ax.set_xticklabels(xticklabels, rotation = 90, ha="left")
    # load average costs in and plot as a baseline
    avg_cost = np.mean(load_json('../models/refModule/json/balancedavg1/costs-fixedPose96-cost_duration-average.json').values())
    plt.axhline(y=avg_cost,linestyle='dashed',color='k')    
    plt.tight_layout()
    if saveout:
        plt.savefig('./plots/cost_by_condition_across_splits.pdf')