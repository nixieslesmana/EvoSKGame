# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:10:46 2024

@author: nixie
"""

import pandas as pd
import numpy as np
import glob

import plotting

def delta_sample_binned(dArr, deltaSampleNum=None, dbound_=None, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    
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
    
    #print(dArr[0], dArr[-1])
    dAll = np.linspace(dArr[0], dArr[-1], min(deltaSampleNum, len(dArr))+1)
    
    for i in range(len(dAll)-1):
        dRange = dArr[(dArr <= dAll[i]) * (dArr >= dAll[i+1])]
        #print(dRange)
        if len(dRange) >= 1:
            dArr_ += [np.random.choice(dRange, 1)[0]]
            
    return np.array(dArr_)

def reinit_arr(nAgents, episodeLen, numSeeds, read=False):
    
    mTraj = np.zeros((nAgents, episodeLen, numSeeds))
    rTraj = np.zeros((nAgents, episodeLen, numSeeds))
    sTraj = np.zeros((nAgents, episodeLen, numSeeds))
    
    bTraj = np.zeros((nAgents, episodeLen, numSeeds))
    qTraj = np.zeros((nAgents, episodeLen, numSeeds))
    iMinTraj = np.zeros((nAgents, episodeLen, numSeeds), dtype='int')
    
    avgRTraj_ = np.zeros((episodeLen, numSeeds))
    f0Traj = np.zeros((episodeLen, numSeeds))
    isEventTraj = np.zeros((episodeLen, numSeeds))
    
    if read:
        return bTraj, qTraj, iMinTraj, mTraj, avgRTraj_, rTraj, sTraj
    
    return bTraj, qTraj, iMinTraj, mTraj, avgRTraj_, rTraj, sTraj, f0Traj, isEventTraj

def record_results(world, NID, aID, eID, bsID, seedID, seed, episodeLen, seedJ,
                   bTraj, qTraj, iMinTraj, mTraj, avgRTraj_, rTraj, sTraj, 
                   window=False, foldername_ = 'Results/', to_record = ['b', 'q', 'i', 'm', 'avgR', 'r', 's'],
                   adaptStr_=[], truePid=[1]):
    
    #print(bTraj.shape) # 3 --> original; 4 --> get bTraj.shape[3]
    
    if episodeLen%1000 == 0:
        windowID = str(episodeLen//1000)
    else:
        windowID = str(float(episodeLen/1000))
        
    if window:
        windowID = 'w' + windowID
    
    if 'b' in to_record:
        # % beta: truePid = [0, 1]
        if len(bTraj.shape) == 3:
            filename_ = foldername_ + 'b{}_{}_{}_{}_{}k_{}_{}.pkl'.format(NID, aID, eID, bsID, windowID, seedJ, seed)
            
            if not glob.glob(filename_):    
                df = pd.DataFrame(np.transpose(bTraj[:, :, seedID]))        
                df.to_pickle(filename_)
            else:
                print('attempt b-overwrite seed', seed)
                
        elif len(bTraj.shape) == 4:
            for evo_n in range(bTraj.shape[3]):
                evo_str = adaptStr_[evo_n]
                #print('evo_str', evo_str)
                #print('b-record', bTraj[:, :, seedID, evo_n][0, -10:])
                if evo_str == 'temp':
                    evo_str = 'b'
                
                filename_ = foldername_ + '{}{}_{}_{}_{}_{}k_{}_{}.pkl'.format(evo_str, NID, aID, eID, bsID, windowID, seedJ, seed)
                if not glob.glob(filename_):    
                    df = pd.DataFrame(np.transpose(bTraj[:, :, seedID, evo_n]))        
                    df.to_pickle(filename_)
                else:
                    print('attempt {}-overwrite seed'.format(evo_str), seed)
        
        else:
            raise ValueError()
    
    if 'q' in to_record:
        # % q   
        filename_ = foldername_ + 'q{}_{}_{}_{}_{}k_{}_{}.pkl'.format(NID, aID, eID, bsID, windowID, seedJ, seed)
        
        if not glob.glob(filename_):   
            df = pd.DataFrame(np.transpose(qTraj[:, :, seedID]))         
            df.to_pickle(filename_)
        else:
            print('attempt q-overwrite seed', seed)
        
    if 'i' in to_record:
        # % i (iters, 256)
        filename_ = foldername_ + 'i{}_{}_{}_{}_{}k_{}_{}.pkl'.format(NID, aID, eID, bsID, windowID, seedJ, seed)
        
        if not glob.glob(filename_):
            df = pd.DataFrame(np.transpose(iMinTraj[:, :, seedID]))            
            df.to_pickle(filename_)
        else:
            print('attempt i-overwrite seed', seed)
    
    #### both EVO, notEVO #### avgRTraj_, mTraj, rTraj, cycleInset, M
    if 'm' in to_record:
        # % m
        filename_ = foldername_ + 'm{}_{}_{}_{}_{}k_{}_{}.pkl'.format(NID, aID, eID, bsID, windowID, seedJ, seed)
        
        if not glob.glob(filename_):     
            df = pd.DataFrame(np.transpose(mTraj[:, :, seedID]))
            df.to_pickle(filename_)
        else:
            print('attempt m-overwrite seed', seed)

    if 'avgR' in to_record:
        # % avgR (iters, 1) --> cycleInset recoverable, i.e., cycleInset=np.transpose(avgRTraj_[-20:, :])
        filename_ = foldername_ + 'avgR{}_{}_{}_{}_{}k_{}_{}.pkl'.format(NID, aID, eID, bsID, windowID, seedJ, seed)
        
        if not glob.glob(filename_):     
            df = pd.DataFrame(np.expand_dims(avgRTraj_[:, seedID], axis=1))
            df.to_pickle(filename_)
        else:
            print('attempt avgR-overwrite seed', seed)

    if 'r' in to_record:
        # % r (iters, 256)
        filename_ = foldername_ + 'r{}_{}_{}_{}_{}k_{}_{}.pkl'.format(NID, aID, eID, bsID, windowID, seedJ, seed)
        
        if not glob.glob(filename_):         
            df = pd.DataFrame(np.transpose(rTraj[:, :, seedID]))
            df.to_pickle(filename_)
        else:
            print('attempt r-overwrite seed', seed)

    if 's' in to_record:
        # % s (iters, 256) --> M recoverable, i.e., M=np.average(sTraj[:,-1,:], axis=0)
        filename_ = foldername_ + 's{}_{}_{}_{}_{}k_{}_{}.pkl'.format(NID, aID, eID, bsID, windowID, seedJ, seed)
        
        if not glob.glob(filename_):
            df = pd.DataFrame(np.transpose(sTraj[:, :, seedID]))            
            df.to_pickle(filename_)
        else:
            print('attempt s-overwrite seed', seed)
            
    # % J (256, 256)
    filename_ = foldername_ + 'J{}_{}_{}.pkl'.format(NID, eID, seedJ)
    
    if not glob.glob(filename_):
        df = pd.DataFrame(world.J)
        df.to_pickle(filename_)
    else:
        print('J N{}-eps{}-seed{} is in!'.format(NID, eID, seedJ))

def read_results(nAgents, alpha, eps, algo_p, episodeLen, seedJ_p=None, seed_p=None, 
                 epLenSearch=False, window=False, windowLen=None, windowStart=1,
                 reads=[True]*7, foldername_ = 'Results/', adaptStr=['a','temp'], truePid=[1], perturbID = None):
    
    # default
    risk = 'mean'
    J0 = 0
    h = 0
    #tMin = 0.
    #tMax = 2.
    
    adaptBS, BSpool, BSdist, lrbs, initType, tempflip, spinflip, withEvent, delta, nrandom, spinOnly, numOfi, tMid = algo_p 
    
    # BSpool
    beta = None
    if adaptBS and BSpool:
        bsID = str(adaptBS)[0] + str(BSdist)[0] + str(lrbs)
    elif adaptBS and not BSpool:
        bsID = str(adaptBS)[0] + 'None' + str(lrbs)
        if delta is not None:
            bsID += 'd' + str(delta)
        else:
            assert withEvent is False
        
        if initType != .1:
            bsID += str(initType)
        if tempflip:
            bsID += str(tempflip)[0]
        if not spinflip:
            bsID += str(spinflip)[0]
        if nrandom:
            bsID += 'nr'
        if spinOnly:
            bsID += 'sOnly'
        if numOfi > 1:
            bsID += 'n{}'.format(numOfi)
        
        if perturbID is not None:
            bsID += perturbID

    else:
        bsID = str(adaptBS)[0] + str(tMid)
        beta = np.round(tMid, 4)
    
    print('===in pkl:', bsID, '===')    
    
    NID = nAgents
    if alpha is not None:
        aID = float(alpha)
    else:
        aID = alpha
    eID = float(eps)
    
    if not window:
        windowLen = episodeLen
        windowStart = 1
    else:
        assert windowLen is not None
    
    startSeed, numSeeds = seed_p
    startSeedJ, numSeedJs = seedJ_p
    
    ##########################
    
    print('alpha', alpha)
    
    for seedJ in range(startSeedJ, startSeedJ + numSeedJs):    
        print('seedJ', seedJ)
        filename_ = foldername_ + 'J{}_{}_{}.pkl'.format(NID, eID, seedJ)
        J= np.array(pd.read_pickle(filename_))
        
        bTraj, qTraj, iMinTraj, mTraj, avgRTraj, rTraj, sTraj = reinit_arr(nAgents, episodeLen-(windowStart-1)*windowLen, numSeeds, read=True)
        numTruePid = len(truePid)
        bTraj = np.zeros((nAgents, episodeLen-(windowStart-1)*windowLen, numSeeds, numTruePid))
        
        seedID = 0
        for seed in range(startSeed, startSeed + numSeeds):
            print('seed', seed)
            
            if episodeLen%windowLen > 0:
                K = episodeLen//windowLen + 1
            else:
                K = episodeLen//windowLen
                
            for k in range(windowStart, K + 1):
                print(k) 
                
                kmin = (k-1)*windowLen
                kmax = k*windowLen
                
                if k*windowLen > episodeLen:
                    #kmin = (k-1)*windowLen
                    kmax = (k-1)*windowLen + episodeLen%windowLen
                
                if window:
                    windowID = 'w' + str(kmax//1000)
                else:
                    windowID = str(kmax//1000)
                    
                kmin_ = kmin-(windowStart-1)*windowLen
                kmax_ = kmax-(windowStart-1)*windowLen
                
                if reads[0]:
                    count = 0
                    for evo_n in truePid:
                        evo_str = adaptStr[evo_n]
                        if evo_str == 'temp':
                            evo_str = 'b'
                        
                        filename_ = foldername_ + '{}{}_{}_{}_{}_{}k_{}_{}.pkl'.format(evo_str, NID, aID, eID, bsID, windowID, seedJ, seed)
                        bseed = np.transpose(np.array(pd.read_pickle(filename_))) #--> recover 'df' --> arr_convert
                        bTraj[:, kmin_:kmax_, seedID, count] = bseed[:, :kmax-kmin]
                        
                        count += 1
                    
                if reads[1]:
                    # \% q Traj
                    filename_ = foldername_ + 'q{}_{}_{}_{}_{}k_{}_{}.pkl'.format(NID, aID, eID, bsID, windowID, seedJ, seed)
                    qseed = np.transpose(np.array(pd.read_pickle(filename_))) #--> recover 'df' --> arr_convert
                    qTraj[:, kmin_:kmax_, seedID] = qseed[:, :kmax-kmin]
                
                if reads[2]:
                    # \% iMin Traj
                    filename_ = foldername_ + 'i{}_{}_{}_{}_{}k_{}_{}.pkl'.format(NID, aID, eID, bsID, windowID, seedJ, seed)
                    iseed = np.transpose(np.array(pd.read_pickle(filename_))) #--> recover 'df' --> arr_convert
                    iMinTraj[:, kmin_:kmax_, seedID] = iseed[:, :kmax-kmin]

                if reads[3]:                
                    # \% m Traj
                    filename_ = foldername_ + 'm{}_{}_{}_{}_{}k_{}_{}.pkl'.format(NID, aID, eID, bsID, windowID, seedJ, seed)
                    mseed = np.transpose(np.array(pd.read_pickle(filename_))) #--> recover 'df' --> arr_convert
                    mTraj[:, kmin_:kmax_, seedID] = mseed[:, :kmax-kmin]
                    #print(mseed)
                
                if reads[4]:  
                    # \% avgR Traj
                    filename_ = foldername_ + 'avgR{}_{}_{}_{}_{}k_{}_{}.pkl'.format(NID, aID, eID, bsID, windowID, seedJ, seed)
                    avgRseed = np.transpose(np.array(pd.read_pickle(filename_))) #--> recover 'df' --> arr_convert
                    #print(avgRseed.shape)
                    avgRTraj[kmin_:kmax_, seedID] = avgRseed[:, :kmax-kmin]
                
                if reads[5]:  
                    # \% r Traj
                    filename_ = foldername_ + 'r{}_{}_{}_{}_{}k_{}_{}.pkl'.format(NID, aID, eID, bsID, windowID, seedJ, seed)
                    rseed = np.transpose(np.array(pd.read_pickle(filename_))) #--> recover 'df' --> arr_convert
                    rTraj[:, kmin_:kmax_, seedID] = rseed[:, :kmax-kmin]
                
                if reads[6]:  
                    # \% s Traj
                    filename_ = foldername_ + 's{}_{}_{}_{}_{}k_{}_{}.pkl'.format(NID, aID, eID, bsID, windowID, seedJ, seed)
                    sseed = np.transpose(np.array(pd.read_pickle(filename_))) #--> recover 'df' --> arr_convert
                    sTraj[:, kmin_:kmax_, seedID] = sseed[:, :kmax-kmin]
            
            seedID += 1
            
        cycleInset = np.transpose(avgRTraj[-20:, :])
        M = np.average(sTraj[:,-1,:], axis=0)
        
        if epLenSearch:
            print('episodeLen:', episodeLen)
            
            return NID, J, avgRTraj, mTraj, rTraj, bTraj, qTraj, iMinTraj
        
        else:
            # plotting here
            figtitle = 'N:{}, Q(alpha):{}, J0:{}, h: {}, risk:{}, \n J-symm(eps, seedJ):({}, {}), beta(b0, adapt, Tmid):({}, {}, {}), \n BS(numOfi, period, lrbs, pool-dist):({}, {}, {}, {}-{}) \n \n'.format(nAgents, alpha, 
                                                                                                                                                                                                               J0, h, risk,
                                                                                                                                                                                                               eps, seedJ, 
                                                                                                                                                                                                               beta, adaptBS, tMid, 
                                                                                                                                                                                                               1, 1, lrbs, True, BSdist)
            
            filename = 'Figures/'
            filename += '{}_N{}_a{}_r{}-{}_e{}-{}_{}'.format(risk, nAgents, alpha,
                                                                       J0, h, eps, seedJ, 
                                                                       bsID) #, runID)
        
            plotting.general(NID, J, numSeeds, startSeed, episodeLen, avgRTraj, mTraj, rTraj, cycleInset, M, 
                             figtitle, filename)
            
            if adaptBS:        
                plotting.Tevolution(NID, J, numSeeds, startSeed, episodeLen, tMid, 
                                    bTraj, qTraj, iMinTraj, figtitle, filename)