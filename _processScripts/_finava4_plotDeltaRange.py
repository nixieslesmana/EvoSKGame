# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:29:51 2024

@author: nixie
"""

import numpy as np
import pandas as pd
idx = pd.IndexSlice

import powerlaw
import matplotlib.pyplot as plt

############################## System Parameters ##############################

s_ub = 9

nAgents = 256 #512
alpha = .9

usebound = True
dbound = {(128, .7): (-.076, -.023),#(-.076, -.004),
          (128, None): (-.009, -.003),
          (256, None): (-.031, -.013), #(-.039, -.03), #(-.04, -.003),
          (256, .9): (-.065, -.035),
          (256, .8): (-.065, -.035),
          (256, .77): (-.05, -.03), #(-.035, -.01), #(-.065, -.035),
          (256, .75): (-.035, -.01), #(-.065, -.035), #(-.069, -.057), #  #(-.07, -.02),
          (256, .7): (-.065, -.035), #(-.035, -.02), #(-.07, -.02),
          (256, .5): (-.013, -.009), #(-.026, -.003), #(-.05, -.003),
          (256, .3): (-.04, -.003), #(-.016, -.003),
          (512, .7): (-.03, -.012), #(-.038, -.003),
          (512, None): (-.027, -.003),
          (1024, .3): (-.007, -.005), #(-.012, -.0066),
          (1024, .5): (-.0068, -.0061), #(-.014, -.007),
          (1024, .7): (-.012, -.008), #(-.012, -.009), #(-.026, -.0031),
          (1024, .77): (-.009, -.003),
          (1024, .75): (-.021, -.0208),
          (1024, .8): (-.041, -.013),
          (1024, .9): (),
          (1024, None): (-.026, -.009), #(-.009, -.0031) #(-.015, -.0031)
          }

'''
removestr = 'J01234-pl'
JsRanges = [(range(5), range(5, 10)),
            (range(5), [5]), 
            (range(5), [6]), 
            (range(5), [7]),
            (range(5), [8]), 
            (range(5), [9])]
'''

removestr = 's56789-pl'
JsRanges = [(range(5), range(5, 10)),
            ([0], range(5, 10)), 
            ([1], range(5, 10)), 
            ([2], range(5, 10)),
            ([3], range(5, 10)), 
            ([4], range(5, 10))]
'''
removestr = 'J0-pl'
JsRanges = [([0], range(5, 10)), 
            ([0], [5]), 
            ([0], [6]), 
            ([0], [7]),
            ([0], [8]), 
            ([0], [9])]

removestr = 'J1-pl'
JsRanges = [([1], range(5, 10)), 
            ([1], [5]), 
            ([1], [6]), 
            ([1], [7]),
            ([1], [8]), 
            ([1], [9])]

removestr = 'J2-pl'
JsRanges = [([2], range(5, 10)), 
            ([2], [5]), 
            ([2], [6]), 
            ([2], [7]),
            ([2], [8]), 
            ([2], [9])]

removestr = 'J3-pl'
JsRanges = [([3], range(5, 10)), 
            ([3], [5]), 
            ([3], [6]), 
            ([3], [7]),
            ([3], [8]), 
            ([3], [9])]

removestr = 'J4-pl'
JsRanges = [([4], range(5, 10)), 
            ([4], [5]), 
            ([4], [6]), 
            ([4], [7]),
            ([4], [8]), 
            ([4], [9])]
'''

#'nofgap-init.1-rm45484917' 

truncated = True
if nAgents==128:
    t_str_def = 100
elif nAgents==256:
    t_str_def = 150
elif nAgents==512:
    t_str_def=200
#elif alpha == .7:
#    t_str_def = 252
else:
    t_str_def=300 #252
    
#t_strDict = {(0,5): 300, (0,6): 300}
t_strDict = {}

tunesteady = False #True: 15, 18 overpower?
startSteady = 1000
if tunesteady:
    removestr += '-{}'.format(startSteady)
    
############################## Fitting Parameters #############################
fit_discrete = True
if fit_discrete:
    str_fitdiscr = 'dTflip' #'djump0-20k' #'d128jump0' 'd128.1' 'd1024' #'d' #'d_2389'
else:
    str_fitdiscr = 'c'

slack = .2 # .2
xmin, fit_method = 2, 'MLE'

if alpha is None:
    foldername = 'compressed/ava_N{}a{}Tadapt'.format(nAgents, alpha) #'dResults_{}/'.format(str_fitdiscr)
else:    
    foldername = 'compressed/ava_N{}a{}Tadapt'.format(nAgents, str(alpha)[2:]) #'dResults_{}/'.format(str_fitdiscr)

if nAgents==1024 and alpha in [.75]:
    foldername += '_full/'
elif nAgents==1024 and alpha in [.7]:
    foldername += '_300k/'
elif nAgents==1024 and alpha in [.8, .9]:
    foldername += '_dLarge/'
elif nAgents==1024 and alpha in [.3, .5]:
    foldername += '_win60k/'
else:
    foldername += '/'

print('folder:', foldername)

filename = foldername + 'N{}a{}_xmin{}_fit{}'.format(nAgents, alpha, xmin, fit_method)

slopeTarget = -1.5
slopeLow = slopeTarget + slack
slopeHigh = slopeTarget - slack
slopeOut = -1.    
dSeed = 0

maxSize = None
pmf = False
loggedPdf = True
numlog = 20
##################################

alldelta = pd.read_csv(filename+'.csv', header=None, index_col=[0, 1, 2, 3])
alldelta = alldelta.loc[(slopeLow, slopeHigh, slopeOut, dSeed)].to_numpy(dtype=float)
if len(alldelta.shape) > 1:
    alldelta = alldelta[-1, :]

if usebound:
    d_l, d_u = dbound[(nAgents, alpha)]
else:
    d_l, d_u = -alldelta[0], -alldelta[-1]    

removestr = '{}-{}'.format(-np.round(d_l, 4), -np.round(d_u, 4)) + '_' + removestr

dArr = alldelta[(alldelta >= -d_u)*(alldelta <= -d_l)] 
print('dArr:', dArr, len(dArr), ', is = unique len:', len(np.unique(dArr)))

# Resample delta init - 10
deltaSampleNum = 50
np.random.seed(0)
dArr_ = []
dAll = np.linspace(dArr[0], dArr[-1], min(deltaSampleNum, len(dArr))+1)
for i in range(len(dAll)-1):
    dRange = dArr[(dArr <= dAll[i]) * (dArr >= dAll[i+1])]
    if len(dRange) >= 1:
        dArr_ += [np.random.choice(dRange, 1)[0]]

dArr = np.array(dArr_)
print('dArr:', dArr, len(dArr), ', is = unique len:', len(np.unique(dArr)))
#raise ValueError()
#'''
for xmin in [2]: #[20, 22, 24, 28, 30]: #[2]: #[10, 2]:
    nrow = 1 + 5
    ncol = 5
    fig, ax_ = plt.subplots(nrow, ncol, figsize=(5*ncol, 3*nrow))
    colId = -1
    
    for i in np.linspace(0, len(dArr)-1, ncol).astype(int):
        colId += 1
        rowId = -1
    
        delta = dArr[i]
        print('===', i, delta, '===')
        #continue
        
        for JRange, sRange in JsRanges:
            
            rowId += 1        
            ax = ax_[rowId, colId]
    
            sizes = []
            
            for startSeedJ in JRange: 
                for startSeed in sRange: 
                    #if (startSeedJ, startSeed) in preSteady[(nAgents, alpha)]:
                    #    continue
                    
                    if startSeed > s_ub:
                        continue
                    
                    if truncated:
                        if (startSeedJ, startSeed) in t_strDict.keys():
                            t_str = t_strDict[(startSeedJ, startSeed)]
                            print(t_str)
                        else:
                            t_str = t_str_def
                        
                        filename_ =  filename+'_l{}h{}_{}_J{}s{}_{}k.csv'.format(-slopeLow, -slopeHigh, dSeed, startSeedJ, startSeed, t_str)                        
                    else:
                        filename_ =  filename+'_l{}h{}_{}_J{}s{}.csv'.format(-slopeLow, -slopeHigh, dSeed, startSeedJ, startSeed)
                    
                    print(filename_)
                    try:
                        df = pd.read_csv(filename_, header=0)
                    except:
                        print(filename_ + ' do not exist.')
                        print('skip.')
                        raise ValueError()
                    
                        if nAgents == 1024:
                            filename_ =  filename+'_l{}h{}_{}_J{}s{}.csv'.format(-slopeLow, -slopeHigh, dSeed, startSeedJ, startSeed)
                        elif nAgents == 128:
                            filename_ =  filename+'_l{}h{}_{}_J{}s{}_{}k.csv'.format(-slopeLow, -slopeHigh, dSeed, startSeedJ, startSeed, t_str_def)                        

                        df = pd.read_csv(filename_, header=0)
                    ############### SET DELTA HERE, DEPENDING ON JS ############
                    # NO RANGE???
                    try:
                        sizeJs = df.loc[idx[:], idx[str(delta)]].fillna(0).to_numpy(dtype=int)
                    except:
                        sizeJs = df.loc[idx[:], idx[str(np.round(delta, 9))]].fillna(0).to_numpy(dtype=int)
                    sizeJs = sizeJs[sizeJs > 0]
                    
                    if len(sizeJs) > 0:
                        
                        if tunesteady:
                            print('startSteady', startSteady)
                            sizeJs = sizeJs[-startSteady:]
                        
                        print(startSeedJ, startSeed, '| max ', max(sizeJs), ', len', len(sizeJs))
                        print('--')
                    else:
                        print(startSeedJ, startSeed, '| len', len(sizeJs))
                        print('--')
                        
                    sizes += list(sizeJs)
            
            # Sizes < maxSize for all delta + Exponents
            sizes = np.array(sizes)
            data = sizes[:] #[sizes<maxSize]
            maxSize = None
            
            if len(np.unique(data)) < 5:
                print('len data < 5')
                continue
            
            #print('sizes max:', max(sizes))
            
            if fit_method == 'MLE':
                fit_method = 'Likelihood'
            
            if maxSize is not None:
                data = data[data<=maxSize]
            else:
                maxSize = max(data)
                data = data[data<=maxSize]
            data_ = data[:]
            
            data = data[data>=xmin]
            #print('data max:', max(data))
            
            d = np.diff(np.unique(data)).min()
            
            left_of_first_bin = data.min() - float(d)/2
            right_of_last_bin = data.max() + float(d)/2
            
            P, binEdges = np.histogram(data, density=True, bins=np.arange(left_of_first_bin, right_of_last_bin + d, d))
            if pmf:
                P /= 1/np.diff(binEdges)
            
            C = np.cumsum(P)
            X = (binEdges[:-1] + binEdges[1:])/2
            X = X.astype('int')
            
            if loggedPdf:
                ### pdf log ####
                binEdges = np.logspace(np.log10(xmin), np.log10(maxSize), numlog) #np.logspace(0, 4, 50)
                Plog, _ = np.histogram(data, density=True, bins=binEdges)
                if pmf:
                    widths = (binEdges[1:] - binEdges[:-1])
                    Plog /= 1/widths
                
                Xlog = (binEdges[:-1] + binEdges[1:])/2
                Xlog = Xlog.astype('int')
            else:
                Xlog, Plog = X, P
            
            Xbool_ = Plog > 0
            #print(Xbool_)
            ################################### start fitting #############################
            print('bf powerlaw fitting, size min, max', min(data), max(data))
            #print('Plog', min(Plog[Xbool_]))
            print('--')
            
            ax.scatter(X, P, alpha=.2)
            ax.scatter(Xlog, Plog, marker='x')
            try:
                results = powerlaw.Fit(data, discrete=True, xmin=xmin, fit_method=fit_method)
                #results = powerlaw.Fit(data, discrete=True, xmin=None, fit_method=fit_method)
                
            except:
                continue
            colors = ['black','red', 'green']
            count = 0
            
            for fittedDistr in [results.power_law, results.truncated_power_law, results.lognormal]:
                
                if count==0:
                    slope=fittedDistr.alpha
                    cutoff=0.
                else:
                    slope=fittedDistr.parameter1
                    cutoff=fittedDistr.parameter2
                
                pdf_d = fittedDistr.pdf(Xlog)
                
                label_ = fittedDistr.name + ':{},{}'.format(np.round(slope,2), np.round(cutoff,4))
                if count == 2:
                    score1, score2 = results.distribution_compare('truncated_power_law', 'lognormal')
                    label_ += ',{},{}'.format(np.round(score1, 2), np.round(score2, 2))
                    #print('exp check:', results.distribution_compare('lognormal', 'exponential'))
                
                '''
                print(fittedDistr.xmin)
                count+=1
                continue
            
                #if count==0:
                Xbool = Xbool_[len(Xlog)-len(pdf_d):][:]
                Xlog_ = Xlog[len(Xlog)-len(pdf_d):][:]
                
                ax.plot(Xlog_[Xbool], pdf_d[Xbool], color=colors[count], #'red', 
                        label=label_)
                '''
                ax.plot(Xlog[Xbool_], pdf_d[Xbool_], color=colors[count], #'red', 
                        label=label_)
                
                count += 1
                
            ax.set_ylim((1e-10, 1))#(min(Plog[Xbool_]), 1))
            ax.set_yscale('log')
            ax.set_xscale('log')
            if rowId == 0:
                ax.set_title('delta:{}'.format(delta))
            else:
                ax.set_title('J{}s{}'.format(JRange[0], sRange[0]))
                
            ax.legend(loc='upper right') #, prop={'size': 7.5})
            
    plt.tight_layout()
    if truncated:
        plt.savefig(foldername + 'lognorm_N{}a{}_xmin{}_{}k_{}.png'.format(nAgents, alpha, xmin, t_str_def, removestr))
    else:
        plt.savefig(foldername + 'lognorm_N{}a{}_xmin{}_{}.png'.format(nAgents, alpha, xmin, removestr))
    plt.close()


  
'''ADD TRUNCATED POWER LAW FIT

import powerlaw


##### PRINT 'HERE' INSIDE TRUNCATED POWER LAW
##### WHY ONLY POWER LAW IS CALLED IN CLASS FIT?

fit_method = 'Likelihood' #'KS'
xmax = None
xmin = 6

# Problematic: Js05N1024 - delta=0.020379563503939158 #(Js=05 -> apply to 15 wht happens?)
q_l, q_u = .34, .78

for xmin in [3, 4,5]:

    print('---xmin:{}---'.format(xmin))
    
    dArr = plotting.get_dArray(fmin_, q_l, q_u)
    dArr = np.random.choice(dArr, size=min(deltaSampleNum, len(dArr)))
    dArr[::-1].sort()
    
    for delta in dArr: #[0.028418564420840568, 0.020379563503939158]:
        fdelta = plotting.get_fdelta(fmin_, delta)
        sizes, startTimes, endTimes = plotting.getAva(qTraj_, fdelta)
        sizes = np.array(sizes) #sizes = sizes[sizes<maxSize]
        data = sizes
        
        results = powerlaw.Fit(data, discrete=True, xmin=xmin, fit_method=fit_method)#,
                               #parameter_range={'alpha':[1.49, 1.51]})
        
        slope_d = results.truncated_power_law.parameter1 
        beta_d = results.truncated_power_law.parameter2
        
        ### LINE 156 SETATTR FUNCTION!!! -> here set tpl
        ## where is discrete? in Distribution.pdf(data) to compute likelihoods in tpl.pdf
        ## but where is tpl.pdf called? 
        inside def likelihoods in def loglikelihoods in def fit_function inputted to scipy.optimize.fmin
        ## where is the param_optimize part? @class tpl, there is _init_params, _in_std-range()
        l634, scipy optimize fmin
        ## to conclude: discrete-tpl-pdf is in the fmin for scipy
        
        ## discrete-vs-cont-tpl: l1431, 1417, 
        
        # Discrete n cont both goes to Distr.pdf
        l960 in Distrib.likelihoods
        l1435 in tpl.pdf
        l828 in Distribution.pdf
        likelihood=f*C (same f, diff C)
        
        # Discrete C: lerchphi?? is zeta function in denom of Likelihood.. 
        # zeta for cutoff denom modif formula..???
        # Continuous C: cdf formula for pl exp cutoff
        
        ###### CHECK WITH SIZES FROM C_0506 #########
        d .0017
        
        ## go to parent class: Distribution, def pdf(): self._pdf_discrete_normalizer()
        ## there is __init__ there, calling fit(data)
        
        print(slope_d, beta_d)
        print('R, p:', results.distribution_compare('truncated_power_law', 'exponential'))
        print('KS (lowest), loglikelihood (highest):', results.truncated_power_law.KS(sizes[sizes>=xmin]), results.truncated_power_law.loglikelihoods(sizes[sizes>=xmin]).sum())
        
        
pdf_d = results.truncated_power_law.pdf(data)
ccdf_d = results.truncated_power_law.ccdf(sizes)


fig, ax1 = plt.subplots(1,1)

#results.truncated_power_law.plot_pdf(ax=ax1)
#ax1.scatter(sizes, Psizes) # X_=sizes, Y_=Psizes
ax1.scatter(sizes[sizes>=2], pdf_d, alpha=.2, label='fit,pdf')
#ccdf_d = np.cumsum(pdf_d)
ax1.scatter(sizes[sizes>=2], ccdf_d, color='red', alpha=.2, label='fit,cdf')
ax1.set_ylim(top=1)
ax1.set_ylim(bottom = 10**-4)
ax1.set_xlim((1, 1000))
ax1.set_xscale('log')
ax1.set_yscale('log')
'''