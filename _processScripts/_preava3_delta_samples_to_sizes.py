# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:11:49 2024

@author: nixie
"""

import gc
import pkl_, plotting
import numpy as np
import pandas as pd
import glob
#import time
#from delta_all_to_samples import delta_sample_binned

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

#####################################################

windowDict = {}
'''
windowDict = {(1,6,.3,512):(100000, False, 1, 100000),
              (0,5,.3,2048): (144000, True, 9-8, 12000),
              (1,5,.3,2048): (144000, True, 8-7, 12000),
              (1,6,.3,2048): (100000, True, 2-1, 50000),
              (0,6,.3,2048): (100000, True, 5-4, 12000)}
'''
NRange = [256] #[512, 2048]
JRange = [0,1,2,3,4]
sRange = [5,6,7,8,9]

initType, tempflip, spinflip = 0., True, True # .1, False (default) | 0., False | 0., True

alpha = .5
lrbs = alpha

#foldername = 'Results/' #'E:/Results_Tflip/' #'Results/' 
#foldername = 'ResultsTflip_128256/'
foldername = 'ResultsT{}a{}/'.format(NRange[0], str(alpha)[2:]) 

ava_qty = 'fmin' # 'rmin', 'fmin' (default)
            
steadyDict = {(128, .3): 80000,
              (128, .5): 80000,
              (128, .7): 60000,
              (256, .3): 130000, #100000,
              (256, .5): 230000,#130000,
              (256, .7): 130000,#90000,
              (256, .75): 130000, #90000,
              (256, .77): 130000,#90000,
              (256, .8): 130000,#90000,
              (256, .9): 130000,#90000,
              (512, .3): 140000,
              (512, .5): 180000,
              (512, .7): 120000,
              (1024, .3): 240000,
              (1024, .5): 240000,
              (1024, .7): 240000,
              (1024, .75): 300000-60000,
              (1024, .77): 300000-60000,
              (1024, .8): 300000-60000,
              (1024, .9): 240000}

fit_discrete = True
#fit_discrete = False

if fit_discrete and lrbs != alpha:
    str_fitdiscr = 'dTflip{}'.format(lrbs) #'djump0-10k'
elif lrbs == alpha:
    str_fitdiscr = 'dTflip'
else:
    str_fitdiscr = 'c'
    
fdeltaFull = True #default: False
resampleNum = None #(default), small num=len of dArr to cut compute time

for nAgents in NRange:
    for startSeedJ in JRange: #, 1]:
        for startSeed in sRange:
            
            eps = 0.
            
            tMid = 0.
            adaptBS = True
            BSpool = False #True
            BSdist = 'avg'
            withEvent = False
            nrandom = False
            spinOnly = False
            numOfi = 1
            delta = 15
            if not withEvent:
                delta = None
            algo_p = (adaptBS, BSpool, BSdist, lrbs, initType, tempflip,spinflip, withEvent, delta, nrandom, spinOnly, numOfi, tMid)
            
            if (startSeedJ, startSeed, alpha, nAgents) in windowDict.keys():
                episodeLen, window, windowStart, windowLen = windowDict[(startSeedJ, startSeed, alpha, nAgents)]
            else:
                if nAgents == 128:
                    episodeLen = 100000 #50000
                    windowStart = 1 #3 9 1
                elif nAgents == 256: # or (nAgents == 1024 and (alpha, startSeedJ, startSeed) in exceptions):
                    episodeLen = 250000
                    windowStart = 1 #3 9 1
                elif nAgents == 512:
                    episodeLen = 200000 #200000
                    windowStart = 1
                elif nAgents == 1024:
                    episodeLen = 300000 #252000 
                    windowStart = 1 #15 #9
                else:
                    raise ValueError()
                print('Eplen:', episodeLen)
                
                windowStart = 1 #3 9 1
                window = True #False #True
                #windowLen = 100000 #25000
                windowLen = int(128*100000/nAgents)
                windowLen = min(episodeLen, 1000 * (windowLen // 1000))
            
            epLenSearch = True
            
            numSeedJs = 1
            seedJ_p = (startSeedJ, numSeedJs)
            
            numSeeds = 1
            seed_p = (startSeed, numSeeds)
            
            if ava_qty == 'fmin':
                print('ava type default, fmin')
                _, _, _, _, _, _, qTraj, _ = pkl_.read_results(nAgents, alpha, eps, algo_p, episodeLen, 
                                                                                           seedJ_p, seed_p, epLenSearch=True, 
                                                                                           window=window, windowLen=windowLen, windowStart=windowStart,
                                                                                           reads=[False] + [True] + [False]*5, foldername_ = foldername)
            
            elif ava_qty == 'rmin':
                print('setting fmin <- rmin')
                _, _, _, _, qTraj, _, _, _ = pkl_.read_results(nAgents, alpha, eps, algo_p, episodeLen, 
                                                                      seedJ_p, seed_p, epLenSearch=True, 
                                                                      window=window, windowLen=windowLen, windowStart=windowStart,
                                                                      reads=[False]*4 + [True]*2 + [False], foldername_=foldername)
                
            else:
                raise ValueError()
            
            iMinTraj = np.zeros(qTraj.shape)
            rmin = np.expand_dims(np.min(qTraj[:,:-1,:], axis=0), axis=0)
            iMinTraj[:, 1:, :] = (qTraj[:,:-1,:] == rmin).astype(int)
            
            t = qTraj.shape[1]  - 1 #50000 - 1 #qTraj.shape[1] - 1
            t_str = (t+1)//1000
            print('t, qshape:', t, qTraj.shape[1]-1) #t_:', t, t_)
            
            figtitle = ''
            
            if adaptBS:
                seed_ = 0
                #filename = 'Figures/N{}J{}s{}a{}adapt{}eve{}__{}'.format(nAgents, startSeedJ, startSeed, alpha, str(adaptBS)[0], str(delta)+str(nrandom)[0]+str(spinOnly)[0], t_)
                
                fmin = plotting.get_fmin(nAgents, t+1, seed_, qTraj[:, :t+1, :], iMinTraj[:, :t+1, :])
                
                fgap = np.maximum.accumulate(fmin)
                _, fgap_uniqueId = np.unique(fgap, return_index=True)
                
                print(fgap_uniqueId)
                #raise ValueError()
                step = .01
                colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
                
                steady = True
                pmf = False
                loggedPdf = True
                
                for fit_method in ['MLE']:
                    fgapEndId = 1
                    for pos in [1]: #, 3/4]: #, 1/2]: #, 0]:
                        
                        fval = fgap_uniqueId[0]*(1-pos) + fgap_uniqueId[-1]*(pos)
                        
                        prevEndId = fgapEndId
                        
                        fgapEndId = np.arange(0, len(fgap_uniqueId))[fgap_uniqueId <= fval][-1]
                        if fgapEndId > 0:
                            fgapEndId = -(len(fgap_uniqueId)-fgapEndId)
                        
                        if fgapEndId == prevEndId:
                            continue
                        
                        fval = fgap_uniqueId[fgapEndId] #[fgap_uniqueId <= fval][-1]
                        steadyIters = t+1 - fval
                        
                        if steadyIters > t+1 or (fgapEndId not in [0,-1] and episodeLen - (t+1) + fval < 20000): 
                            #steadyIters = min(steadyIters, t)
                            continue
                        
                        print('f0:', fgap[-steadyIters], t+1-steadyIters)
                        steadyIters = steadyDict[(nAgents, alpha)] #60000
                        print('rewrite steadyIters', steadyIters)
                        
                        if fdeltaFull:
                            fmin_ = fmin[:]
                        else:
                            fmin_ = fmin[-steadyIters:]
                        qTraj_ = qTraj[:, :t+1, :]
                        qTraj_ = qTraj_[:, -steadyIters:, 0]
                        
                        for xmin in [2]: #[6,8,10]:
                        #for xmin in [4,3,2, 6, 8, 10]: #[4, 3, 2]: #, False]:   
                            
                            print(fgapEndId, steadyIters, xmin, fit_method)
                            np.random.seed(0)
                            
                            t = qTraj.shape[1] - 1
                            earlyIters = t+1-steadyIters #20000
                            
                            filename = 'dResults_{}/N{}a{}_xmin{}_fit{}'.format(str_fitdiscr, nAgents, alpha, xmin, fit_method)
                            dArr = pd.read_csv(filename+'.csv', header=None, index_col=[0, 1, 2, 3])
                            
                            slopeLow = -1.5 + .2 #.05
                            slopeHigh = -1.5 - .2 #.05
                            slopeOut = -1
                            
                            dSeed = 0
                            dArr = dArr.loc[(slopeLow, slopeHigh, slopeOut, dSeed)].to_numpy(dtype=float)
                            if len(dArr.shape) > 1:
                                dArr = dArr[-1, :]
                            
                            dArr = delta_sample_binned(dArr, resampleNum)
                            print('check random seed for dresample:', len(dArr))
                            
                            '''
                            import csv
                            
                            f = open(filename+'.csv', 'a', newline = '')
                            writer = csv.writer(f)
                            
                            to_append = [[slopeLow, slopeHigh, slopeOut, dSeed] + list(dArr)]
                            writer.writerows(to_append)  
                            
                            f.close()
                            '''
                            
                            sizeDict = dict()
                            #start = time.time()
                            for delta in dArr: 
                                fdelta = plotting.get_fdelta(fmin_, delta)
                                #print(time.time() - start)
                                if fdeltaFull:
                                    fdelta = fdelta[-steadyIters:]
                                
                                sizes, _, _ = plotting.getAva(qTraj_, fdelta)
                                #print(time.time()-start)
                                
                                sizeDict[delta] = sizes
                                #print(time.time()-start)
                                #print('---')
                                
                            #raise ValueError()
                            filename_ = filename+'_l{}h{}_{}_J{}s{}_{}k.csv'.format(-slopeLow, -slopeHigh, dSeed, startSeedJ, startSeed, t_str)
                            if not glob.glob(filename_):
                                df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in sizeDict.items() ]))
                                df.to_csv(filename_, mode='a', index=False, header=True)
                            else:
                                print('Attempt rewrite:', filename_)
                                raise ValueError()
                    
            del qTraj, iMinTraj, rmin, qTraj_, fmin, fmin_, fgap, fdelta, df, sizeDict
            gc.collect()
                            
                            