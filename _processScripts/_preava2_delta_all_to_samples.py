# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:07:15 2024

@author: nixie
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import glob

def delta_sample_binned(dArr, deltaSampleNum=None, dbound_=None):
    
    if len(dArr)==0:
        return dArr
    
    dArr[::-1].sort()
    
    if dbound_ is None:
        d_l, d_u = -dArr[0], -dArr[-1]
    else:
        d_l, d_u = dbound_
        
    dArr = dArr[(dArr >= -d_u)*(dArr <= -d_l)] 
    
    if deltaSampleNum is None:
        return dArr
    
    dArr_ = []
    
    dAll = np.linspace(dArr[0], dArr[-1], min(deltaSampleNum, len(dArr))+1)
    
    for i in range(len(dAll)-1):
        dRange = dArr[(dArr <= dAll[i]) * (dArr >= dAll[i+1])]
        #print(dRange)
        if len(dRange) >= 1:
            dArr_ += [np.random.choice(dRange, 1)[0]]
            
    return np.array(dArr_)


###############################################################################

alpha = .5
lrbs = alpha
fit_discrete = True #False

if fit_discrete and lrbs != alpha:
    str_fitdiscr = 'dTflip{}'.format(lrbs)
elif lrbs == alpha:
    str_fitdiscr = 'dTflip'
else:
    str_fitdiscr = 'c'

#df = pd.read_csv('allDelta_{}.csv'.format(str_fitdiscr), header=0, index_col=[0,1,2,3,4,5,6])
df = pd.read_csv('allDelta_{}.csv'.format(str_fitdiscr), header=None, index_col=[0,1,2,3,4,5,6])
df = df.sort_index()
print(str_fitdiscr)
print(df.shape) # shape = 25?
print(df.index)

NRange = [256] #[512,2048]
JRange = [0,1,2,3,4]
sRange = [5,6,7,8,9]

for nAgents in NRange:
    for xmin in [2]:
        for fit_method in ['MLE']:
            
            filename = 'N{}a{}_xmin{}_fit{}'.format(nAgents, alpha, xmin, fit_method)
            
            # row filter: (nAgents, xmin, fit_method)
            filtered_df = df.loc[(nAgents, alpha, xmin, fit_method)]
            print(filtered_df.shape, '=50?') # (50, ..)
            
            idx = pd.IndexSlice
            #deltaMat = filtered_df.loc[idx[:, :, ["d"]], idx[:]]
            #slopeMat = filtered_df.loc[idx[:, :, ["slope"]], idx[:]]
            
            deltaMat = filtered_df.loc[idx[JRange, sRange, ["d"]], idx[:]]  
            slopeMat = filtered_df.loc[idx[JRange, sRange, ["slope"]], idx[:]]
            print(deltaMat.shape, slopeMat.shape, 'equal?')
            
            slopeLow = -1.5 + .2 #.05
            slopeHigh = -1.5 - .2 #.05
            slopeOut = -1. #1
            deltaTarget = deltaMat.to_numpy()[(slopeMat >= slopeHigh).to_numpy()*(slopeMat <=slopeLow).to_numpy()]
            deltaLow = deltaMat.to_numpy()[(slopeMat > slopeLow).to_numpy()*(slopeMat <= slopeOut).to_numpy()]
            deltaHigh = deltaMat.to_numpy()[(slopeMat < slopeHigh).to_numpy()]
            
            
            dSeed = 0
            np.random.seed(dSeed)
            
            filename_ = 'dResults_{}/d'.format(str_fitdiscr)+filename+'_l{}h{}_{}.png'.format(-slopeLow, -slopeHigh, dSeed)
            if not glob.glob(filename_):
                plt.figure()
                P, binEdges = np.histogram(deltaTarget, density=True, bins=10)
                X = (binEdges[1:]+binEdges[:-1])/2
                plt.bar(X, P, width = np.diff(binEdges)[0], label='[{},{}]'.format(slopeHigh, slopeLow))
                
                P, binEdges = np.histogram(deltaLow, density=True, bins=10)
                X = (binEdges[1:]+binEdges[:-1])/2
                plt.bar(X, P, width = np.diff(binEdges)[0], alpha=.5, label='>{}'.format(slopeLow))
                #plt.hist(deltaLow, bins=10, alpha=.2)
                
                P, binEdges = np.histogram(deltaHigh, density=True, bins=10)
                X = (binEdges[1:]+binEdges[:-1])/2
                plt.bar(X, P, width = np.diff(binEdges)[0], alpha=.5, label='<{}'.format(slopeHigh))
                #plt.hist(deltaHigh, bins=10, alpha=.2)
                plt.legend()
                figtitle = 'DELTA DISTRIBUTION: '+ filename
                plt.title(figtitle)
            
                plt.savefig(filename_)
                plt.close()
            
            numd = None # max: 750
            lowhighprop = .2
            dbound = {(256, .7): (-.043, -.016),
                      (512, .7): (-.044, -.033)} # IF USEALL, NO NEED TO REDO WITH DBOUND
            usebound, useall = False, True
            
            if useall:
                lowsize = None
                highsize = None
            else:
                lowsize = int(lowhighprop *numd)
                highsize = int(lowhighprop *numd)
                
            if usebound:
                dbound_ = dbound[(nAgents, alpha)]
            else:
                dbound_ = None
            
            dArr_ = []
            
            dArr_ += list(delta_sample_binned(deltaLow, lowsize, dbound_))
            print('low:', len(dArr_))
            dArr_ += list(delta_sample_binned(deltaHigh, highsize, dbound_))
            print('+ high:', len(dArr_))
            
            if useall:
                targetsize = None
            else:
                targetsize = int(numd - len(dArr_))
                
            dArr_ += list(delta_sample_binned(deltaTarget, targetsize, dbound_))
            print('+ near 1.5:', len(dArr_))
            
            '''
            if usebound:
                dArr_ += list(delta_sample_binned(deltaLow, int(lowhighprop *numd), dbound[(nAgents, alpha)]))
                dArr_ += list(delta_sample_binned(deltaHigh, int(lowhighprop *numd), dbound[(nAgents, alpha)]))
                dArr_ += list(delta_sample_binned(deltaTarget, int(numd-len(dArr_), dbound[(nAgents, alpha)])))
            else:
                dArr_ += list(delta_sample_binned(deltaLow, int(lowhighprop *numd)))
                dArr_ += list(delta_sample_binned(deltaHigh, int(lowhighprop *numd)))
                dArr_ += list(delta_sample_binned(deltaTarget, int(numd-len(dArr_))))
            '''
            
            dArr = np.unique(np.array(dArr_))
            dArr[::-1].sort()
            
            print(dArr, len(dArr))
            #raise ValueError()
            
            f = open('dResults_{}/'.format(str_fitdiscr)+filename+'.csv', 'a', newline = '')
            writer = csv.writer(f)
            
            to_append = [[slopeLow, slopeHigh, slopeOut, dSeed] + list(dArr)]
            writer.writerows(to_append)  
            
            f.close()
                     
        
        