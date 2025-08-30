# -*- coding: utf-8 -*-
"""
Created on Fri May 17 05:35:31 2024

@author: nixie
"""

import gc
import pkl_, plotting
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
#from SKGame import computeAcf
from chaos_func import computeAcf

windowDict={}
notdone = []

JRange = range(5) #range(5)
sRange = range(5,10) #[5,6,7,8,9] #range(5, 10)
NRange = [256]
alpha = .5 #None #.8

#fixedLrbs = True
#foldername = 'ResultsLrbs1/'
#lrbs = .7 #1. 
fixedLrbs = False
foldername = 'ResultsT{}a{}/'.format(NRange[0], str(alpha)[2:])
#'ResultsT1024aNone/'  #'Results/'# 'Results128-jump.1/' # (oldest) 'Results128/'
lrbs = alpha

ava_qty = 'fmin' #'rmin' # 'rmin', 'fmin' (default)

#d_l, d_u = -.05, -.003
d_l, d_u = -.1, -.003 # N=128, 256 or a>.75
#d_l, d_u = -.2, -.1
#d_l, d_u = -.3, -.2
print('dRange:', d_l, d_u)

startSteadyDict = {(128, None): 20000,
                   (256, None): 20000,
                   (512, None): 20000,
                   (1024, None): 60000,
                   (128, .3): 20000,
                   (128, .5): 30000,
                   (128, .7): 30000,
                   (256, .3): 20000,
                   (256, .5): 20000,
                   (256, .7): 20000,
                   (256, .75): 20000,
                   (256, .77): 20000,
                   (256, .8): 20000,
                   (256, .9): 20000,
                   (512, .3): 30000,
                   (512, .5): 20000,
                   (512, .7): 20000,
                   (1024, .3): 30000,
                   (1024, .5): 50000,
                   (1024, .7): 60000,
                   (1024, .75): 60000,
                   (1024, .77): 60000,
                   (1024, .8): 60000,
                   (1024, .9): 60000}
initType, tempflip, spinflip = 0., True, True
winMultiplier = 128

######## DEFAULT
#foldername = 'E:/SERVER/'
#winMultiplier = 128
#initType, tempflip, spinflip = .1, False, True

#initType, tempflip = 0., False, True
#initType, tempflip, spinflip = 0., True, False

fit_discrete = True # False
if fit_discrete:
    str_fitdiscr = 'd'
else:
    str_fitdiscr = 'c'

recordDelta = True
if tempflip:
    if fixedLrbs:
        str_fitdiscr += 'Tflip{}'.format(lrbs)
    else:
        str_fitdiscr += 'Tflip'
        
for nAgents in NRange: #, .7, .9]: 
    for startSeedJ in JRange: #[1,2,3,4]:
        for startSeed in sRange: #[5,6,7,8,9]:
            
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
            algo_p = (adaptBS, BSpool, BSdist, lrbs, initType, tempflip, spinflip, withEvent, delta, nrandom, spinOnly, numOfi, tMid)
            
            if (startSeedJ, startSeed, alpha, nAgents) in windowDict.keys():
                episodeLen, window, windowStart, windowLen = windowDict[(startSeedJ, startSeed, alpha, nAgents)]
            else:
                # SERVER
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
                                
                steady_def = episodeLen - startSteadyDict[(nAgents, alpha)]
                     
                window = True #False #True
                windowLen = int(winMultiplier*100000/nAgents)
                windowLen = min(episodeLen, 1000 * (windowLen // 1000))
            
            epLenSearch = True
            
            numSeedJs = 1
            seedJ_p = (startSeedJ, numSeedJs)
            
            numSeeds = 1
            seed_p = (startSeed, numSeeds)
            
            # full reads: NID, J, avgRTraj, mTraj, rTraj, bTraj, qTraj, iMinTraj = ..
            if ava_qty == 'fmin':
                print('ava type default fmin')
                _, _, avgRTraj, _, _, _, qTraj, _ = pkl_.read_results(nAgents, alpha, eps, algo_p, episodeLen, 
                                                                                           seedJ_p, seed_p, epLenSearch=True, 
                                                                                           window=window, windowLen=windowLen, windowStart=windowStart,
                                                                                           reads=[False] + [True] + [False]*2 + [True] + [False]*2, foldername_=foldername)
                
                
                
            elif ava_qty == 'rmin':
                print('setting fmin <- rmin')
                _, _, avgRTraj, _, qTraj, _, _, _ = pkl_.read_results(nAgents, alpha, eps, algo_p, episodeLen, 
                                                                      seedJ_p, seed_p, epLenSearch=True, 
                                                                      window=window, windowLen=windowLen, windowStart=windowStart,
                                                                      reads=[False]*4 + [True]*2 + [False], foldername_=foldername)
                
            else:
                raise ValueError()
            
            iMinTraj = np.zeros(qTraj.shape)
            rmin = np.expand_dims(np.min(qTraj[:,:-1,:], axis=0), axis=0)
            iMinTraj[:, 1:, :] = (qTraj[:,:-1,:] == rmin).astype(int)    
            
            t_ = episodeLen - 1
            t = qTraj.shape[1] - 1 #qTraj.shape[1] - 1 #episodeLen - 1 #windowLen - 1
            print('t, t_:', t, t_)
            
            avgRTraj_ = avgRTraj[:t+1,:]
            avgR = np.average(avgRTraj_[-20:, 0])
            print('>>avgR:', avgR)
            print(avgRTraj_[-20:, 0])
            
            figtitle = ''
            
            if adaptBS:
                seed_ = 0
                #filename = 'Figures/N{}J{}s{}a{}adapt{}eve{}__{}'.format(nAgents, startSeedJ, startSeed, alpha, str(adaptBS)[0], str(delta)+str(nrandom)[0]+str(spinOnly)[0], t_)
                
                fmin = plotting.get_fmin(nAgents, t+1, seed_, qTraj[:, :t+1, :], iMinTraj[:, :t+1, :])
                #raise ValueError()
                fgap = np.maximum.accumulate(fmin)
                _, fgap_uniqueId = np.unique(fgap, return_index=True)
                
                print('>> fgap index:', fgap_uniqueId)
                
                step = .01
                colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
                
                steady = True
                pmf = False
                loggedPdf = True #True # use -3,4; cek ydArr
                
                for fit_method in ['MLE']: #['KS', 'MLE']:
                
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
                        
                        while steadyIters < 1000:
                            fgapEndId -=1
                            fval = fgap_uniqueId[fgapEndId] #[fgap_uniqueId <= fval][-1]
                            steadyIters = t+1 - fval
                                
                        if steadyIters > t+1 or (fgapEndId not in [0,-1] and episodeLen - (t+1) + fval < 2000): 
                            #steadyIters = min(steadyIters, t)
                            continue
                        
                        print('f0, startSteady:', fgap[-steadyIters], t+1-steadyIters)
                        
                        for steadyIters in [steady_def, t+1]: #episodeLen]: #[5000, 50000, 100000]:
                            if steadyIters < steady_def:
                                continue
                            
                            if not recordDelta and steadyIters==steady_def:
                                continue
                            
                            if recordDelta and steadyIters==t+1:
                                continue
                            
                            steadyIters = min(steadyIters, t_)
                            #min(steadyIters, episodeLen) #50000 # CAN ALSO FGAP BUT WITH CAP 50K # max(steadyIters, 50000)
                            print('>>rewrite steadyIters', steadyIters)
                            
                            for xmin in [2]: #[2,3,4,6]: #[6,8,10]:
                            #for xmin in [4,3,2, 6, 8, 10]: #[4, 3, 2]: #, False]:   
                                
                                print(fgapEndId, xmin)
                                np.random.seed(0)
                                
                                t = qTraj.shape[1] - 1
                                earlyIters = t+1-steadyIters #20000
                                
                                ### Delta binning
                                deltaSampleNum = 10 # 20 
                                deltaBinNum = 3 # 1
                                maxSize = None #500 #300
                                print('maxSize:', maxSize)
                                divisor = 10 #5
                                slopeminId = 2
                                slopemaxId = math.floor(divisor*.7) #divisor-2
                                
                                nrow = deltaBinNum + 1
                                ncol = 3 + 2
                                fig, ax = plt.subplots(nrow, ncol, figsize=(5*ncol, 3*nrow)) #figure()
                                
                                axd = ax[0, 0]
                                ax04 = ax[0,4].twinx()
                                
                                ### Plot fmin vs fall
                                if steady:
                                    fmin_ = fmin[-steadyIters:]
                                    qTraj_ = qTraj[:, -steadyIters:, 0]
                                    fall_ = qTraj_.flatten()
                                
                                    xlim_fd = (-.2, max(fmin_)+step)
                                    xlim_fall = (-.35, max(fall_)+.01) 
                                    ylim_fd = (-.01, 62)
                                else:
                                    fmin_ = fmin[:earlyIters]
                                    qTraj_ = qTraj[:, :earlyIters, 0]
                                    fall_ = qTraj_.flatten()
                                    
                                    xlim_fd = (-.2, max(fmin_)+step)
                                    xlim_fall = (min(fmin_)-step, max(fall_)+step)
                                    ylim_fd = (-.01, 20)
                                
                                num = math.ceil((np.max(fmin_) - np.min(fmin_)) / step)
                                binEdges = np.min(fmin_) + np.arange(0, num) * step
                                
                                num = math.ceil((np.max(fall_) - np.max(fmin_)) / step)
                                binEdges = np.concatenate((binEdges, np.max(fmin_) + np.arange(0, num) * step))
                                
                                Pall, _ = np.histogram(fall_, density=True, bins=binEdges) # density p(x)
                                #Pall /= 1/widths # p(Dx) = p(x)Dx
                                Xall = (binEdges[:-1] + binEdges[1:])/2
                                
                                P0, _ = np.histogram(fmin_, density=True, bins=binEdges)
                                #P0 /= 1/widths
                                X0 = (binEdges[:-1] + binEdges[1:])/2
                                
                                ### Plot Delta distribution
                                alldelta = plotting.get_dArray(fmin_)
                                #alldelta = alldelta[alldelta<.1]
                                num_ = math.ceil((np.max(alldelta) - np.min(alldelta)) / step)
                                binEdges_ = np.min(alldelta) + np.arange(0, num_) * step
                                Pdelta, _ = np.histogram(alldelta, density=True, bins=binEdges_)
                                Xdelta = (binEdges_[:-1] + binEdges_[1:])/2
                                
                                ax[0,2].plot(Xdelta, Pdelta, color='black')
                                ax[0,2].set_title('{} {}{} delta distr'.format(nAgents, startSeedJ, startSeed))
                                
                                ### Plot Fmin portion
                                if steady:
                                    ax[0,1].scatter(range(episodeLen-steadyIters, episodeLen), fmin_, alpha=.1, color='grey')                
                                else:
                                    ax[0,1].scatter(range(earlyIters), fmin_, alpha=.1, color='grey')
                                ax[0,1].set_title('fmin portion, avgR: {}'.format(np.round(avgR, 4)))
                                ax[0,1].locator_params(axis='x', nbins=5)
                                
                                ### Plot imin correl
                                imin_ = np.argmax(iMinTraj[:, :, 0], axis=0)
                                X = imin_[-100:]
                                isStationary = True
                                numLags = 30 
                                acf, _, _ = computeAcf(np.expand_dims(X, axis=(0,2)), numLags=numLags, steadyIters=len(X))
                                ax[0,3].plot(acf, color='black')
                                ax[0,3].set_ylim((0,1.1))
                                ax[0,3].set_title('imin (last 50) correl')
                                
                                ### Plot avgR
                                #print(avgRTraj_.shape)
                                ax[1,1].plot(avgRTraj_, color='black')
                                ax[1,1].set_title('avgR:{}'.format([np.round(avgRTraj_[i].item(),4) for i in np.linspace(0, t_, 5).astype(int)]))
                                ax[1,1].set_ylim((1., 1.55))
                                ax[1,1].locator_params(axis='x', nbins=5)
                                plt.tight_layout()
                                
                                print('SKIP deltasearch >> use alldelta_dTflip.csv')
                                if recordDelta:
                                    plt.close()
                                
                                if recordDelta and steadyIters == steady_def:
                                    
                                    dArr = pkl_.delta_sample_binned(alldelta, 30, dbound_=(d_l, d_u), seed=0)
                                    tplAll = [list(dArr), [-1.5]*len(dArr)]
                                    tplAll = np.transpose(tplAll)
                                    
                                    f = open('allDelta_{}.csv'.format(str_fitdiscr), 'a', newline = '')
                                    writer = csv.writer(f)
                                    rownames = ['d', 'slope']
                                    for r in range(2):
                                        to_append = [[nAgents, alpha, xmin, fit_method, startSeedJ, startSeed, rownames[r]]+list(tplAll[:, r])]
                                        writer.writerows(to_append)
                                        
                                    f.close()
                    
            del avgRTraj, qTraj, iMinTraj, fmin, fgap, fmin_, fall_, qTraj_
            gc.collect()
                                    
                                