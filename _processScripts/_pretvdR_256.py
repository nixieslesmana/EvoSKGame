# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:22:16 2024

@author: nixie
"""

import numpy as np
import csv
#import pandas as pd
import itertools
from chaos_func import computeAcf_t0_Js, aggregateCorrelation, plotResults
import pkl_, pickle
#import matplotlib.pyplot as plt

notdone = dict()
exceptions = []

stat = 'corrM' #'temps' #'avgR' #'corrM' # 'maxR' # 'stdR'
corr_norm = True # False

nAgents = 256 #512
JRange = range(5)
sRange = range(5, 10)
for alpha in [.3]: 
    foldername_ = 'ResultsF256a{}/'.format(str(alpha)[2:]) 
    #'ResultsT256a{}/'.format(str(alpha)[2:]) 
    #'ResultsT256a{}_prtbX/'.format(str(alpha)[2:]) 
    #'ResultsT256a{}_prtbXinit/'.format(str(alpha)[2:]) 
    record_evoStop = False
    if record_evoStop:
        episodeLen, windowStart = 60000, 5
    else:
        #episodeLen, windowStart = 72000, 1 # 6 (confirm steady); 7, 8, 9, 10 (at 7 steady, but may age)
        episodeLen, windowStart = 100000, 1 # F
        #episodeLen, windowStart = 150000, 1 # _T: 2,4,6,7,8,9
        #episodeLen, windowStart = 15000, 1 #_Tprtb
        #episodeLen, windowStart = 10000, 1 #_Tprb_init
        
    adaptBS, tempRange = False, [.9,.15, .25] # .6, .7, .8,.9,1., #[0., .1, .2] #[.3, .4, .5]
    #adaptBS, tempRange = True, [0.] 
    initType, tempflip, spinflip = 0., True, True
    
    #'''
    evoStop_ = np.inf #int(.4*episodeLen) # 700, 900 # with freeze (default: np.inf)
    withPerturb = False
    perturb_pList = [('', '')] 
    perturbSeedRange = ['train'] #
    
    '''
    evoStop_ = 4000 + 1 #prtInit: 0+1 #4000 + 1
    withPerturb = True
    perturb_pList = list(itertools.product([nAgents], [0.])) # .5, .2, .1, 0., -.1, -.2, -.5
    perturbSeedRange = list(range(5, 10)) #list(range(10)) #list(range(5, 10))
    if record_evoStop:
        perturbSeedRange = [5]
    '''
    
    evoConst = 0 # 50, 100; default: 0
    evoStop = evoStop_ + evoConst
    
    Corr_ylim = {0.0: (.45, 1.05), 0.1: (.95, .995), 0.2: (.935, .972), 0.3: (.84, .945), 0.4: (.82, .915), 
                 0.5: (.7, .87), 0.6: (.66, .83), 0.8: (.4, .73), 1.: (0., .6)}
    with_ylim = False # check aging: set F; else: set T
    
    for tMid in tempRange:
    
        winMultiplier = 128
        window = True #False #True
        windowLen = int(winMultiplier*100000/nAgents)
        windowLen = min(episodeLen, 1000 * (windowLen // 1000))
        
        if windowStart > 1:
            shifter = 0
            startSteady = (windowStart-1) * windowLen + shifter
            steadyIters = windowLen * (episodeLen//windowLen-windowStart+1) - shifter # how many last iters
        else: # see <R> stabilize or not
            #startSteady = 6000+4000 # _Tprtb N128: 3000+800 | N256: 6000+4000
            startSteady = 10000 # _F N256: 2000, 4000, 10000, 30000, 50000
            steadyIters = episodeLen - startSteady
            
        assert steadyIters < episodeLen # tSteady=epLen/10
        
        steadyIters0 = steadyIters #48000 #steadyIters-1000 #100000
        steadyIters1 = steadyIters0 # (avgR) 5000, (mTraj) ..
        
        lenLags = 100 # N128: 5
        '''
        lagMax = 12000
        assert lagMax < steadyIters0
        lags = np.logspace(0, np.log10(lagMax), lenLags).astype(int)
        '''
        
        aLagMax = int(1000*np.ceil(.8*steadyIters0/1000)) #100000
        aLags = np.logspace(0, np.log10(aLagMax), lenLags)
        if record_evoStop or aLags.max() > alpha* .8*steadyIters0:
            aLags = aLags[aLags < alpha * .8*steadyIters0]
            
        lags = (aLags / alpha).astype(int)
        lagMax = max(lags)
        print('a lag max:', aLagMax)
        print('lag max:', lagMax, '< steady:', steadyIters0, '*.8:', .8*steadyIters0)
        assert lagMax < steadyIters0
        
        #raise ValueError()
        
        lags = sorted(list(set(lags)))
        if lags[0] > 0:
            lags = [0] + lags
        
        '''for pos in range(3):
            filename = '{}_perturb{}.csv'.format(stat, pos)
            f = open(filename, 'a', newline = '')
            writer = csv.writer(f)
            to_append = [['']*8 + list(alpha*np.array(lags))]
            writer.writerows(to_append)  
            f.close()
        '''
        
        lenLags = len(lags)
        numLags = len(lags)
        
        numt0 = 10 
        #t0s = np.logspace(0, np.log10(2/3*steadyIters0), numt0).astype(int)
        t0s = np.logspace(np.log10(startSteady+0), np.log10(startSteady+2/3*steadyIters0), numt0).astype(int)
        t0s -= startSteady
        t0s = sorted(list(set(t0s)))
        if t0s[0] < 0.:
            t0s[0] = 0
        numt0 = len(t0s)
        
        # tEnds --> default: list(range(steadyIters, episodeLen, 100))
        iterList = [startSteady+steadyIters0, episodeLen]
        
        avgRall = np.zeros((2 + len(lags), len(iterList)))
        #avgRall = []
        
        for col in [-1]: #range(len(iterList)): 
            tEnd = iterList[col] - startSteady
            print('startSteady:', startSteady, '+ tStart, tEnd:', tEnd-steadyIters0, tEnd)
            
            mTrajAll = None
            np.random.seed(0)
            agentSamples = np.random.randint(0, nAgents-1, 16) # range(nAgents)
            
            avgRTrajPlot = np.zeros((steadyIters0, len(JRange)*len(sRange), len(perturbSeedRange))) #None
            #tempTrajAll = []
            avgRTrajAll = []
            
            acvf_temp = np.zeros((numt0, numLags, 3))
            acvfStat_temp = np.zeros((numLags, 3))
            
            jsID = -1
            
            for startSeedJ in JRange:
                for startSeed in sRange:
                    jsID += 1
                    testID = -1
                    for spinScatterNum, tempScatterStrength in perturb_pList:
                        for perturbSeed in perturbSeedRange: #tMid in tRange:
                            testID += 1
                            
                            NID = nAgents
                            aID = alpha
                            eID = float(0)
                            
                            lrbs = alpha
                            
                            if adaptBS:
                                
                                bsID = str(adaptBS)[0] + 'None' + str(lrbs)
                                if initType != .1:
                                    bsID += str(float(initType))
                                    
                                if tempflip:
                                    bsID += str(tempflip)[0]
                                
                                if not spinflip:
                                    bsID += str(spinflip)[0]
                                    
                                if evoStop < np.inf and evoConst > 0:
                                    # add wPerturb
                                    perturbID = '({},{},{},{})'.format(evoConst, spinScatterNum, tempScatterStrength, perturbSeed)
                                    
                                elif evoStop < np.inf and evoConst == 0:
                                    perturbID = '({},{},{})'.format(spinScatterNum, tempScatterStrength, perturbSeed)
                                
                                else:
                                    perturbID = None
                                    
                            else:
                                bsID = str(adaptBS)[0] + str(tMid)
                                perturbID = None
                            
                            eps = 0
                            numSeedJs = 1
                            seedJ_p = (startSeedJ, numSeedJs)
                            
                            numSeeds = 1
                            seed_p = (startSeed, numSeeds)
                            
                            BSpool = False #True
                            BSdist = 'avg'
                            withEvent = False
                            numOfi = 1
                            delta = 15
                            if not withEvent:
                                delta = None
                            nrandom = False
                            spinOnly = False
                            algo_p = (adaptBS, BSpool, BSdist, lrbs, initType, tempflip, spinflip, withEvent, delta, nrandom, spinOnly, numOfi, tMid)
                            
                            '''
                            if adaptBS:
                                _, _, avgRTraj_, mTraj, _, bTraj, _, _ = pkl_.read_results(nAgents, alpha, eps, algo_p, episodeLen, 
                                                                                           seedJ_p, seed_p, epLenSearch=True, 
                                                                                           window=window, windowLen=windowLen, windowStart=windowStart,
                                                                                           reads = [True] + [False]*2 + [True]*2 + [False]*3,
                                                                                           foldername_=foldername_, perturbID=perturbID)
                            '''
                            #else:
                            _, _, avgRTraj_, mTraj, _, _, _, _ = pkl_.read_results(nAgents, alpha, eps, algo_p, episodeLen, 
                                                                                   seedJ_p, seed_p, epLenSearch=True, 
                                                                                   window=window, windowLen=windowLen, windowStart=windowStart,
                                                                                   reads = [False]*3 + [True]*2 + [False]*3,
                                                                                   foldername_=foldername_, perturbID=perturbID)
                            
                            #avgRTraj_.shape # {}k, 1
                            #mTraj.shape # N, {}k, 1
                            #bTraj.shape # N, {}k, 1, 1
                            #raise ValueError()
                            
                            if windowStart==1:
                                mTraj = mTraj[:, startSteady:, :]
                                avgRTraj_ = avgRTraj_[startSteady:, :]
                            
                            mTraj = mTraj[:, :tEnd, :]
                            
                            acvf_temp_Js = computeAcf_t0_Js(mTraj, lags, t0s, steadyIters0)
                            acvfStat_temp_Js = np.sum(acvf_temp_Js, axis=0)
                            
                            acvf_temp += acvf_temp_Js # not stationary
                            acvfStat_temp += acvfStat_temp_Js # stationary
                            
                            mSteadyDraw = min(2*lagMax, steadyIters0)
                            mSampled = mTraj[agentSamples, -mSteadyDraw:, :]
                            if mTrajAll is None:
                                mTrajAll = mSampled[:]
                            else:
                                mTrajAll = np.concatenate((mTrajAll, mSampled), axis=2)
                            '''  
                            if adaptBS:
                                #bTraj = np.array(pd.read_pickle(filename2)).T
                                bTraj = bTraj[:, tEnd-1].flatten()
                                tempTraj = 1/bTraj # 128, 
                                
                                tempTrajAll += [tempTraj.mean()]
                            '''
                            # \% avgR Traj
                            #avgRTraj_ = np.array(pd.read_pickle(filename1)) #--> recover 'df' --> arr_convert
                            #print(avgRTraj_.shape) # 25k, 1 
                            avgRTraj_ = avgRTraj_[:tEnd, :]
                            
                            avgRTrajPlot[:, jsID, testID] = avgRTraj_[-steadyIters1:, 0]
                            
                            #avgRTrajAll += [avgRTraj_[-steadyIters0:, 0].mean()]
                            
                        #print(alpha, avgR_bfPerturb, avgR)
            
            divisor = (nAgents*len(JRange)*len(sRange)*len(perturbSeedRange))
            acvfCumt0_, acvfCumt0_norm = aggregateCorrelation(acvf_temp, divisor)
            divisor = (numt0*nAgents*len(JRange)*len(sRange)*len(perturbSeedRange))
            acvfCum_, acvfCum_norm = aggregateCorrelation(acvfStat_temp, divisor)
            
            if corr_norm:
                acvfCum = acvfCum_norm
                acvfCumt0 = acvfCumt0_norm
                #figfoldername = 'FiguresPHASE/norm/'
            else:
                acvfCum = acvfCum_
                acvfCumt0 = acvfCumt0_
                #figfoldername = 'FiguresPHASE/notnorm/'
            
            # Saving the objects:
            datafoldername = 'compressed/'
            figfoldername = datafoldername
            
            if adaptBS and withPerturb:
                datafilename = 'N{}a{}T{}{}_ts{}k,{}k_alag{}'.format(nAgents, alpha, 'adapt', tempScatterStrength,
                                                                     (startSteady+tEnd-steadyIters0)//1000, 
                                                                     (startSteady+tEnd)//1000, aLagMax)
            elif adaptBS and not withPerturb:
                datafilename = 'N{}a{}T{}_ts{}k,{}k_alag{}'.format(nAgents, alpha, 'adapt',
                                                            (startSteady+tEnd-steadyIters0)//1000, 
                                                            (startSteady+tEnd)//1000, aLagMax)
            else:
                datafilename = 'N{}a{}T{}_ts{}k,{}k_alag{}'.format(nAgents, alpha, tMid,
                                                            (startSteady+tEnd-steadyIters0)//1000, 
                                                            (startSteady+tEnd)//1000, aLagMax)
            
            # PLOT
            plotResults(figfoldername, acvfCumt0, acvfCum, avgRTrajPlot, mTrajAll,
                        nAgents, adaptBS, alpha, tMid, JRange, sRange,
                        withPerturb, record_evoStop, tempScatterStrength, perturbSeedRange,
                        startSteady, tEnd, steadyIters0, steadyIters1, steadyIters,
                        lags, t0s)
            
            # Write to file:
            hyperparams = [nAgents, adaptBS, alpha, tMid, JRange, sRange,
                           withPerturb, record_evoStop, tempScatterStrength, perturbSeedRange,
                           startSteady, tEnd, steadyIters0, steadyIters1, steadyIters]
            
            #lags, t0s, acvfCumt0, acvfCum, avgRTrajPlot, mTrajAll,
            with open(datafoldername + datafilename + '.npy', 'wb') as f:
                np.save(f, np.array(hyperparams, dtype='object'))
                np.save(f, np.array(t0s))
                np.save(f, np.array(lags))
                np.save(f, acvfCumt0)
                np.save(f, acvfCum)
                np.save(f, mTrajAll)
                np.save(f, avgRTrajPlot)
                    

    
