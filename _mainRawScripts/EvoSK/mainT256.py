# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:52:20 2024

@author: nixie
"""

import sys
import os

cwd = os.getcwd()
sys.path.append(cwd) # nb: append cwd or "../" not enough, need to add __init__ to the cwd;
if "../" not in sys.path:
    sys.path.append("../")

import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt    

from SKGame import *

import plotting
import pkl_ #csv_

nAgents = 256 
episodeLen = 150000 #50000 #720000  # . !!!!!!!!!!!!!!!!!!!!!!!
numSeeds = 1 # 1 !!!!!!!!!!!!!!!!!!!!! cannot be > 1! mTraj[.., seed=1] w dumping out of index
startSeed = 9 # 6 7 8 9
numSeedJs = 1
startSeedJ = 0

meanField = False #True #True #False
if not meanField:
    seedJRange = range(startSeedJ, startSeedJ+numSeedJs)
    mu = 0.
else:
    seedJRange = [None] # READ BS RANDOM NEIGHBOR
    mu = 1.

J0 = 0. # -2 to 2
h = 0. # -2 to 2, VARY SITE-TO-SITE: random-field Ising (~maximal flow in graph [?])
eps = 0 # 1.5, .85, 1.05

print('N:', nAgents, '(default)')
print('eps, mu:', eps, mu, '(default)')

########################### ADAPT=F or T ######################################
adaptBS = True #False
tMid = 0. #0.4

#initType, tempflip, spinflip = 0., True, False
initType, tempflip, spinflip = 0., True, True # RUNNING NOW
#initType, tempflip, spinflip = .1, False, True # DEFAULT
#initType, tempflip = 0., False

print('with Evo:', adaptBS)
print('INIT TYPE, TEMP FLIP, SPIN FLIP:', initType, tempflip, spinflip, '(default)')

numOfi = 1
withEvent = False #True # *False !!!!!!!!!!!!!!!!!!!!!!!
deltaInv = 15 # (.5, 1, 2, ..) : expect LARGER deltaInv -> smaller delta -> everything triggers ava - f0 constant -> not critical
delta = 1/deltaInv #.8
nrandom = False # *False !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
spinOnly = False # *False !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

epLenSearch = True #True #True #False
noPlotting = True #True #True #False
dumping = True
windowMaxSize = 100000
if dumping:
    windowLen = int(128*windowMaxSize/nAgents) #int(256*windowMaxSize/nAgents)
    windowLen = min(episodeLen, 1000 * (windowLen // 1000))
    print('epLen, windowLen:', episodeLen, windowLen)
    assert windowLen <= episodeLen
else:
    windowLen = episodeLen

risk = 'mean'
paramDict = {}

printIters = np.linspace(2, episodeLen - 1, 4).astype(int)

# Find phase=F: q >> 0, m >> 0; theoretically (Fig16), T near 0 & eps near 0
# See m when eps=0.1, T=1/beta \in [1/2, 0.]
###CYCLE
for seedJ in seedJRange:#[.1, .2, .3, .4, .5]: #[0., .6, .85, 1.05, 1.5]:
    #J0, h [-2., 0., 1., 1.5, 2.]:  
    #eps [.1, .85, 1.05, 1.5]: 
    #**beta [1000., .25, .5, 1.]: # .5, .25 isP?
    
    lmbd = 0
    for alpha in [.3]: #!!!!!!!!!!!!!!!!!!!!!!!!
        
        foldername_ = 'ResultsT256a{}/'.format(str(alpha)[2:])

        lrbs = alpha
        
        tMid = np.round(tMid, 4)
        
        if tMid > 0:
            beta = np.round(1/tMid, 4)
        else:
            beta = np.inf 
        
        #lmbd in np.linspace(.9967, .9969, 3):
        lmbd = np.round(lmbd, 4)
        paramDict['lmbd'] = lmbd
        
        if not dumping:
            bTraj, qTraj, iMinTraj, mTraj, avgRTraj_, rTraj, sTraj, f0Traj, isEventTraj = pkl_.reinit_arr(nAgents, episodeLen, numSeeds)
            
            #qposTraj = np.copy(qTraj) #### INCLUDE BELOW AFTER DUMPING
            #qnegTraj = np.copy(qTraj)
            
        if not noPlotting:
            M = np.zeros(numSeeds)
            cycleInset = np.zeros((numSeeds, 20))
        
        for seed in range(numSeeds): #[0, 5]:
            
            #if alpha == .3 and startSeed + seed == 5:
            #    continue
            
            np.random.seed(startSeed + seed)
            
            if adaptBS:
                print('T, alpha, lrbs:', None, alpha, lrbs)
                if withEvent:
                    print('delta:', delta)
            else:
                print('T, alpha:', tMid, alpha)
                
            print('seedJ:', seedJ, '(<-- 0-4)')
            print('seed:', startSeed + seed, '(<-- 5-9)')
            print('foldername_:', foldername_)
            
            start = time.time()
            
            world = SKWorld(seedJ=seedJ, N=nAgents, eps=eps, mu=mu, J0=J0, h=h,
                            riskMeasure=risk, paramDict=paramDict, 
                            adaptBS=adaptBS, initType=initType, tempflip=tempflip, spinflip=spinflip,
                            numOfi=numOfi, withEvent=withEvent, nrandom=nrandom, spinOnly=spinOnly)
            
            if dumping:
                bTraj, qTraj, iMinTraj, mTraj, avgRTraj_, rTraj, sTraj, f0Traj, isEventTraj = pkl_.reinit_arr(nAgents, windowLen, 1)
                
                #qposTraj = np.copy(qTraj) #### INCLUDE BELOW AFTER DUMPING
                #qnegTraj = np.copy(qTraj)
                
            for agent in world.agents:
                agent.alpha = alpha
                agent.lrBs = lrbs
                agent.tempMid = tMid
                agent.beta = agent.initBeta(beta=beta, adaptBS=adaptBS, spinOnly=spinOnly)
            
            if withEvent:
                world.delta = delta
                print('set delta:', world.delta)
            
            #for t in range(episodeLen): 
            for t_ in range(episodeLen):   
                t = t_ % windowLen
                
                world.step()
                
                if (t+1) % 5000 == 0:
                    print('t=', t, t_, ', ', int(time.time()-start), 's')
                
                mTraj[:, t, seed] = world.intentions # for generating acf(m_t, m_t+tau)
                rTraj[:, t, seed] = world.rewards
                
                iMinTraj[:, t, seed] = world.iMin # undefined at init -> no one is killed
                bTraj[:, t, seed] = world.betas
                qTraj[:, t, seed] = world.Qs # after iMin killed
                #qposTraj[:, t, seed] = world.Qpos # after iMin killed
                #qnegTraj[:, t, seed] = world.Qneg # after iMin killed
                
                sTraj[:, t, seed] = world.spins
                
                avgRTraj_[t, seed] = world.negEnergy
                
                f0Traj[t, seed] = world.f0
                isEventTraj[t, seed] = world.is_event # -> for eventDuration
                
                #if epLenSearch and t_+1 >= 5000 and (t_+1) % 4000 == 0:
                if epLenSearch and t_+1 >= episodeLen+5:
                    assert numSeeds == 1 #and adaptBS == True
                
                    #for t in range(19999, 73999, 4000):
                    print('t:', t)
                    
                    figtitle = ''
                    
                    filename = 'Figures/N{}J{}s{}a{}eve{}_{}'.format(nAgents, startSeedJ, startSeed, alpha, str(withEvent)[0], t_)
                    plotting.general(world.nAgents, world.J, 1, startSeed, t+1, avgRTraj_[:t+1, :], mTraj[:, :t+1, :], rTraj[:, :t+1, :], None, None, figtitle, filename)
                    
                    if adaptBS:
                    #for t in range(19999, 73999, 4000):
                        #print('t:', t)
                        # t, t_ = 47999, 97999
                        filename = 'Figures/N{}J{}s{}a{}eve{}__{}'.format(nAgents, startSeedJ, startSeed, alpha, str(withEvent)[0], t_)
                        bTraj_ = np.expand_dims(bTraj, axis=3)
                        plotting.Tevolution(world.nAgents, world.J, 1, startSeed, t+1, tMid, 
                                            bTraj_[:, :t+1, :], qTraj[:, :t+1, :], iMinTraj[:, :t+1, :], rTraj[:, :t+1, :],
                                            figtitle, filename, steadyIters=20000, fcIters_=20000, 
                                            noAva=True) #, 
                                            #deltaRange=(.01, .007)) #, deltaRange=(.03,.005))
                
                if (t_+1) % windowLen == 0 or t_+1 == episodeLen:
                    NID = world.nAgents
                    aID = float(world.agents[0].alpha)
                    eID = float(world.eps)
                    seedID = seed
                    actualSeed = startSeed + seed
                    
                    if adaptBS and world.BSpool:
                        bsID = str(adaptBS)[0] + str(world.BSdist)[0] + str(world.agents[0].lrBs)
                    elif adaptBS and not world.BSpool:
                        bsID = str(adaptBS)[0] + 'None' + str(world.agents[0].lrBs)
                        if world.delta is not None: # withEvent=True
                            bsID += 'd' + str(deltaInv) #world.delta)
                        else:
                            assert withEvent is False
                        
                        if initType != .1:
                            bsID += str(float(initType))
                            
                        if tempflip:
                            bsID += str(tempflip)[0]
                        
                        if not spinflip:
                            bsID += str(spinflip)[0]
                        
                        if world.nrandom:
                            bsID += 'nr'
                            
                        if world.spinOnly:
                            bsID += 'sOnly'
                            
                        if world.numOfi > 1:
                            bsID += 'n{}'.format(world.numOfi)
                    else:
                        bsID = str(adaptBS)[0] + str(tMid)
                    
                    #bTraj = np.squeeze(bTraj, axis=3)
                    pkl_.record_results(world, NID, aID, eID, bsID, seedID, actualSeed, t_+1, seedJ,
                                        bTraj, qTraj, iMinTraj, mTraj, avgRTraj_, rTraj, sTraj, window=dumping, foldername_=foldername_)
                    
                    if dumping:
                        bTraj, qTraj, iMinTraj, mTraj, avgRTraj_, rTraj, sTraj, f0Traj, isEventTraj = pkl_.reinit_arr(nAgents, windowLen, 1)
                        #qposTraj = np.copy(qTraj) #### INCLUDE BELOW AFTER DUMPING
                        #qnegTraj = np.copy(qTraj)
            
            #raise ValueError()
            if epLenSearch:
                continue
            
            if not noPlotting:
                ###### skippable: if one seed or many seeds+dumping #####
                M[seed] = world.avgIntention
                cycleInset[seed, :] = avgRTraj_[-20:, seed]
                print("N:", world.nAgents)
                print("epsilon (J-symm) ~[0, 2]:", world.eps)
                print("alpha (Q-memory) ~[0, 1]:", world.agents[0].alpha)
                print("noise (1/temperature):", world.agents[0].beta) ## affecting!!
                print("--- %s seconds ---" % (time.time() - start))
                #######################
            
        if noPlotting:
            print('skip plotting!')
            continue
    
        figtitle = 'N:{}, Q(alpha, lrE, lrV):{}-{}-{}, J0:{}, h: {}, risk:{}, \n J-symm(eps, seedJ):({}, {}), beta(b0, adapt, Tmid):({}, {}, {}), \n BS(numOfi, period, lrbs, pool-dist, delta):({}, {}, {}, {}-{}, {}) \n \n'.format(world.nAgents, world.agents[0].alpha, 
                                                                                                                                                        world.agents[0].lrE, world.agents[0].lrV,
                                                                                                                                                        world.J0, world.h, risk+world.agents[0].Qparams,
                                                                                                                                                        world.eps, seedJ, 
                                                                                                                                                        beta, world.adaptBS, world.agents[0].tempMid, 
                                                                                                                                                        world.numOfi, world.BSperiod, world.agents[0].lrBs, world.BSpool, world.BSdist, world.delta)
        
        
        ## simple version @read_results()
        filename = 'Figures/'
        filename += '{}_N{}_a{}_r{}-{}_e{}-{}_{}'.format(risk, nAgents, alpha,
                                                            J0, h, eps, seedJ, 
                                                            bsID) #, runID)
        # riskID = risk + str(lmbd)
        
        plotting.general(world.nAgents, world.J, numSeeds, startSeed, episodeLen, avgRTraj_, mTraj, rTraj, cycleInset, M, figtitle, filename)
        
        if not adaptBS:
            continue
        
        #import plotting
        plotting.Tevolution(world.nAgents, world.J, numSeeds, startSeed, episodeLen, tMid, 
                            bTraj, qTraj, iMinTraj, figtitle, filename)

        
