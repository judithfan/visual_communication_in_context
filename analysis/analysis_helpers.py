import os
import json
import numpy as np
import re

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

def get_summary_stats(_D, all_games, correct_only=True):
    '''
    Get summary stats for sketchpad_basic experiment. 
    If correct_only is True, then filter to only include correct trials... except when calculating accuracy, which considers all trials.
    '''
    further_strokes = []
    closer_strokes = []
    further_svgLength = []
    closer_svgLength = []
    further_svgStd = []
    closer_svgStd = []
    further_svgLengthPS = []
    closer_svgLengthPS = []
    further_drawDuration = []
    closer_drawDuration = []
    further_accuracy = []
    closer_accuracy = []
    further_pixelintensity = []
    closer_pixelintensity = []
    for game in all_games:    
        if correct_only:
            D = _D[_D['outcome']==1]
        else:
            D = _D
        thresh = np.mean(D['numStrokes'].values) + 3*np.std(D['numStrokes'].values)
        tmp = D[(D['gameID']== game) & (D['condition'] == 'further') & (D['numStrokes'] < thresh)]['numStrokes']            
        further_strokes.append(tmp.mean())        
        tmp = D[(D['gameID']== game) & (D['condition'] == 'closer') & (D['numStrokes'] < thresh)]['numStrokes']
        closer_strokes.append(tmp.mean())
        further_svgLength.append(D[(D['gameID']== game) & (D['condition'] == 'further')]['svgStringLength'].mean())
        closer_svgLength.append(D[(D['gameID']== game) & (D['condition'] == 'closer')]['svgStringLength'].mean())
        further_svgStd.append(D[(D['gameID']== game) & (D['condition'] == 'further')]['svgStringStd'].mean())
        closer_svgStd.append(D[(D['gameID']== game) & (D['condition'] == 'closer')]['svgStringStd'].mean())    
        further_svgLengthPS.append(D[(D['gameID']== game) & (D['condition'] == 'further')]['svgStringLengthPerStroke'].mean())
        closer_svgLengthPS.append(D[(D['gameID']== game) & (D['condition'] == 'closer')]['svgStringLengthPerStroke'].mean())
        further_drawDuration.append(D[(D['gameID']== game) & (D['condition'] == 'further')]['drawDuration'].mean())
        closer_drawDuration.append(D[(D['gameID']== game) & (D['condition'] == 'closer')]['drawDuration'].mean())
        further_accuracy.append(_D[(_D['gameID']== game) & (_D['condition'] == 'further')]['outcome'].mean())
        closer_accuracy.append(_D[(_D['gameID']== game) & (_D['condition'] == 'closer')]['outcome'].mean())
        further_pixelintensity.append(D[(D['gameID']== game) & (D['condition'] == 'further')]['mean_intensity'].mean())
        closer_pixelintensity.append(D[(D['gameID']== game) & (D['condition'] == 'closer')]['mean_intensity'].mean())

    further_strokes, closer_strokes, further_svgLength, closer_svgLength, \
    further_svgStd, closer_svgStd, further_svgLengthPS, closer_svgLengthPS, \
    further_drawDuration, closer_drawDuration, further_accuracy, closer_accuracy, \
    further_pixelintensity, closer_pixelintensity = map(np.array, \
    [further_strokes, closer_strokes, further_svgLength, closer_svgLength,\
     further_svgStd, closer_svgStd, further_svgLengthPS, closer_svgLengthPS, \
    further_drawDuration, closer_drawDuration, further_accuracy, closer_accuracy, \
    further_pixelintensity, closer_pixelintensity])
    
    return further_strokes, closer_strokes, further_svgLength, closer_svgLength,\
     further_svgStd, closer_svgStd, further_svgLengthPS, closer_svgLengthPS, \
    further_drawDuration, closer_drawDuration, further_accuracy, closer_accuracy, \
    further_pixelintensity, closer_pixelintensity

    
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

    
###### MODEL COMPARISON HELPERS ######

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

def bootstrapCI(x,nIter=1000):
    '''
    input: x is an array
    '''
    u = []
    for i in np.arange(nIter):
        inds = np.random.RandomState(i).choice(len(x),len(x))
        boot = x[inds]
        u.append(np.mean(boot))

    u = np.array(u)
    p1 = sum(u<0)/len(u) * 2
    p2 = sum(u>0)/len(u) * 2
    p = np.min([p1,p2])
    U = np.mean(u)
    lb = np.percentile(u,2.5)
    ub = np.percentile(u,97.5)
    return U,lb,ub,p

def make_category_by_obj_palette():
    import itertools
    col = []
    for j in sns.color_palette("hls", 4):
        col.append([i for i in itertools.repeat(j, 8)])
    return flatten(col)
