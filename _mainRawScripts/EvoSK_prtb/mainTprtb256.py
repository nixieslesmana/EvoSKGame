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
import itertools
import time
#from datetime import datetime
#import matplotlib.pyplot as plt    

from SKGame_aPoolA import *

import plotting#, plot2dhist
import pkl_ #csv_

nAgents = 256 #!!!!!!!!!!!!!!!!!
episodeLen = 15000 #!!!!!!!!!!!!!!!!!!!!!!! N256 usual: 15000; init: 10000 # N1024 ratio: (48000, 60000, 156000)
numSeeds = 1 
startSeed = 5
numSeedJs = 5
startSeedJ = 0

figfoldername = 'Figures_/' 

meanField = False #True #True #False
if not meanField:
    seedJRange = range(startSeedJ, startSeedJ+numSeedJs)
    mu = 0.
else:
    seedJRange = [None] # READ BS RANDOM NEIGHBOR
    mu = 1.

J0 = 0. # -2 to 2
h = 0. # -2 to 2, VARY SITE-TO-SITE: random-field Ising (~maximal flow in graph [?])
eps = 0. # 1.5, .85, 1.05

print('N:', nAgents, '(default)')
print('eps, mu:', eps, mu, '(default)')

########################### ADAPT=F or T ######################################
adaptBS = True 
tMid = 0. #1/1000 #.05

#initType, tempflip, spinflip = 0., True, False
initType, tempflip, spinflip = 0., True, True # RUNNING NOW
#initType, tempflip, spinflip = .1, False, True # DEFAULT
#initType, tempflip = 0., False

print('with Evo:', adaptBS)
print('INIT TYPE, TEMP FLIP, SPIN FLIP:', initType, tempflip, spinflip, '(default)')

######################### ADAPT = T (hyperparams) ################################
numOfi = 1
withEvent = False
deltaInv = 15 # (.5, 1, 2, ..) : expect LARGER deltaInv -> smaller delta -> everything triggers ava - f0 constant -> not critical
delta = 1/deltaInv #.8
nrandom = False
spinOnly = False

risk = 'mean' #'mean-noise'
print('Rwd:', risk, '(default)')
paramDict = {}

numEvo_p = 2
adaptStr = ['a', 'temp']
truepId = [1] #[0, 1]
############### REMINDER!!! ###################
# whatever Id in truepId
# make aNone or bNone
###############################################

adapt_p = [False]*numEvo_p
if adaptBS:
    for pId in truepId:
        adapt_p[pId] = True
else:
    print('----')
    print('WARNING: use singleEvo script')
    #raise NotImplementedError()
    print('----')

pool_p = [True, False] # to activate, set world.BSpool=True first!
#pool_p = [True, True]
#pool_p = [False, False] # default (equiv to world.BSpool=False)

########################## UTIL param #######################################
epLenSearch = True #True #True #False
noPlotting = True #default: True
dumping = True #noPlotting #True
windowMaxSize = 100000
if dumping:
    windowLen = int(128*windowMaxSize/nAgents) #int(256*windowMaxSize/nAgents)
    windowLen = min(episodeLen, 1000 * (windowLen // 1000))
    #windowLen = 1000 ###### Figures delete
    print('epLen:', episodeLen, '(<-- 10000 for _init), windowLen:', windowLen)
    assert windowLen <= episodeLen
else:
    windowLen = episodeLen
printIters = np.linspace(2, episodeLen - 1, 4).astype(int)

for seedJ in seedJRange:
    lmbd = 0
    
    '''
    evoStop_ = np.inf # without freeze (default: np.inf)
    withPerturb = False
    perturb_pList = [('', '', '')] 
    '''
    evoStop_ = 4000+1 #N256: 4000+1 #init: 0 +1  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #N128: 2000 + 1 #N1024: 48000 + 1
    withPerturb = True
    perturb_pList = itertools.product([nAgents], [0.], range(5, 10)) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print('evoStop:', evoStop_, '(<-- 1 for _init)') # [0., 1.] [-1.5, 1.5, 3.5] [2.5]
    
    for spinScatterNum, tempScatterStrength, perturbSeed in perturb_pList: #for evoConst in [0, 50]:
        print('perturb_p (spinScatter, rawTempScaler, seed):', spinScatterNum, '(default), ', tempScatterStrength, '(default), ', perturbSeed, '(<-- 5-9)')
    
        for alpha in [.7]: # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   -> .3 -> .9
            foldername_ = 'Results_/' #'ResultsT256a{}_prtbXinit/'.format(str(alpha)[2:])
            
            evoConst = 0
            evoStop = evoStop_ + evoConst
            if evoStop < np.inf:
                perturbStart = 6000 +1 #N256: 6000 +1 #init: 0+1 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #N128: 3000 +1 #N1024: 60000+1
            else:
                perturbStart = None
                
            print('perturbStart:', perturbStart, '(<-- 1 for _init)')
            
            tDrawList = [] #[evoStop_-1, evoStop_-1+windowLen, perturbStart-1+windowLen, episodeLen]
            fixedLrbs, lrbs = False, alpha #True, .3 
            
            tMid = np.round(tMid, 4)
            
            if tMid > 0:
                beta = np.round(1/tMid, 4)
            else:
                beta = np.inf 
            
            lmbd = np.round(lmbd, 4)
            paramDict['lmbd'] = lmbd
            
            if not dumping:
                bTraj, qTraj, iMinTraj, mTraj, avgRTraj_, rTraj, sTraj, f0Traj, isEventTraj = pkl_.reinit_arr(nAgents, episodeLen, numSeeds)
                try:
                    bTraj = np.zeros((nAgents, episodeLen, numSeeds, len(truepId)))
                except:
                    print('single var adapt, ori bTraj')
                    pass
                
            if not noPlotting:
                M = np.zeros(numSeeds)
                cycleInset = np.zeros((numSeeds, 20))
            
            for seed in range(numSeeds): #[0, 5]:
                
                np.random.seed(startSeed + seed)
                
                if adaptBS:
                    print('T, alpha, lrbs:', None, alpha, lrbs)
                else:
                    print('alpha:', alpha)
                    print('T:', tMid)
                    
                if withEvent:
                    print('delta:', delta)
                
                print('seedJ:', seedJ, '(<-- 0-4)')
                print('seed:', startSeed + seed, '(<-- 5-9)')
                print('foldername_:', foldername_)
                
                start = time.time()
                
                world = SKWorld(seedJ=seedJ, N=nAgents, eps=eps, mu=mu, J0=J0, h=h,
                                riskMeasure=risk, paramDict=paramDict, 
                                adaptBS=adaptBS, initType=initType, tempflip=tempflip, spinflip=spinflip,
                                numOfi=numOfi, withEvent=withEvent, nrandom=nrandom, spinOnly=spinOnly)
                
                world.initPerturb_p(evoStop, withPerturb, spinScatterNum, tempScatterStrength, perturbStart, perturbSeed)
                
                if dumping:
                    bTraj, qTraj, iMinTraj, mTraj, avgRTraj_, rTraj, sTraj, f0Traj, isEventTraj = pkl_.reinit_arr(nAgents, windowLen, 1)
                    try:
                        bTraj = np.zeros((nAgents, windowLen, numSeeds, len(truepId)))
                    except:
                        print('single var adapt, ori bTraj')
                        pass
                
                print('Evo_p:', truepId)
                print('EvoPool_p:', pool_p)
                print('Evo_TempRange:', world.agents[0].tempMin, world.agents[0].tempMax)
                print('Evo_MemoryRange:', world.agents[0].alphaMin, world.agents[0].alphaMax)
                print('Prtb_p (TempPrtbType, TempMultiplier):', world.tempScatterType, np.round(2**world.tempScatterStrength, 2))
                
                for agent in world.agents:
                    agent.lrBs = lrbs
                    agent.fixedLrbs = fixedLrbs
                    agent.tempMid = tMid
                    agent.alpha_init = alpha
                    
                    agent.adapt_p = adapt_p
                    agent.pool_p = pool_p
                    agent.adapt_p_t = adapt_p
                    agent.initEvoArr(evo_p=[alpha, tMid])
                    
                if withEvent:
                    world.delta = delta
                    print('set delta:', world.delta)
                
                #for t in range(episodeLen): 
                for t_ in range(episodeLen):   
                    t = t_ % windowLen
                    
                    world.step()
                    
                    '''###### aRescale (CK)
                    print('BF a(t-1):', sorted([round(world.agents[i].alpha, 4) for i in range(nAgents)]))
                    world.step(Ti=True)
                    print('AFT a(t-1):', sorted([round(world.agents[i].alpha, 4) for i in range(nAgents)]))
                    '''
                    
                    '''##### aRescale (q,m)
                    if t>np.inf and (t-1) % 50 == 0:
                    #if t>50 and (t-1) % 50 == 0:
                        print(t)
                        mArr = mTraj[:, t-50:t, :]
                        print('a(t-1):', [round(world.agents[i].alpha, 4) for i in range(nAgents)])
                        print('m:', list(np.round(mArr[:, -1, 0], 4)))
                        world.step(mArr = mArr) # till previous timestep?
                        
                        ### input TArr!
                        
                        
                        print('a(t-1):', [round(world.agents[i].alpha, 4) for i in range(nAgents)])
                        #raise ValueError()
                    else:
                        world.step()'''
                    
                    if (t+1) % 5000 == 0:
                        print('t=', t, t_, ', ', int(time.time()-start), 's')
                    
                    mTraj[:, t, seed] = world.intentions # for generating acf(m_t, m_t+tau)
                    rTraj[:, t, seed] = world.rewards
                    
                    iMinTraj[:, t, seed] = world.iMin # undefined at init -> no one is killed
                    qTraj[:, t, seed] = world.Qs # after iMin killed
                    
                    if adaptBS:
                        bTraj[:, t, seed, :] = 1/world.trueEvo_ps # 1/alpha, 1/temp range >> plotting: alpha, temp
                    else:
                        bTraj[:, t, seed, :] = np.expand_dims(np.array([agent.beta for agent in world.agents]), axis=1) #world.betas
                    
                    sTraj[:, t, seed] = world.spins
                    avgRTraj_[t, seed] = world.negEnergy
                    
                    f0Traj[t, seed] = world.f0
                    isEventTraj[t, seed] = world.is_event # -> for eventDuration
                    
                    #if t_ == 2:
                    #    raise ValueError()
                    
                    if epLenSearch and t_+1 in tDrawList: #(t_+1) % windowLen == 0: #t_ == episodeLen-1:
                        print('HERE!!!!')
                        assert numSeeds == 1 #and adaptBS == True
                    
                        #for t in range(19999, 73999, 4000):
                        print('t:', t)
                        
                        figtitle = ''
                        
                        if adaptBS and 1 in truepId:
                            if evoStop < np.inf:
                                filename = figfoldername + 'N{}J{}s{}_a{}f{}-{}_T{}-{}_pool-{}_{}_{}({})({})'.format(nAgents, seedJ, startSeed, 
                                                                                                             alpha, lrbs, str(pool_p[0])[0], 
                                                                                                             None, str(pool_p[1])[0], str(world.BSdist)[0], t_,
                                                                                                             world.spinScatterNum, world.tempScatterStrength, world.perturbSeed)
                                if evoConst > 0:
                                    filename += str(evoConst)
                            else:
                                filename = figfoldername + 'N{}J{}s{}_a{}f{}-{}_T{}-{}_pool-{}_{}'.format(nAgents, seedJ, startSeed, 
                                                                                                          alpha, lrbs, str(pool_p[0])[0], 
                                                                                                          None, str(pool_p[1])[0], str(world.BSdist)[0], t_)
                        else:
                            filename = figfoldername + 'N{}J{}s{}_a{}f{}_T{}_{}'.format(nAgents, seedJ, startSeed, alpha, lrbs, tMid, t_)
                            
                        plotting.general(world.nAgents, world.J, 1, startSeed, t+1, avgRTraj_[:t+1, :], mTraj[:, :t+1, :], rTraj[:, :t+1, :], None, None, figtitle, filename)
                        
                        if adaptBS:
                        
                            ##### if len(bTraj.shape) == 3:
                            #bTraj_ = np.expand_dims(bTraj, axis=3)
                            
                            agent = world.agents[0]
                            tRange = np.array([agent.evo_pMin, agent.evo_pMax]).transpose()
                            #adaptStr_ = np.array(adaptStr)[truepId]
                            
                            t0 = 0 #101 # DEFAULT 0
                            plotting.Tevolution(world.nAgents, world.J, 1, startSeed, t+1-t0, tMid, 
                                                bTraj[:, t0:t+1, :], qTraj[:, t0:t+1, :], iMinTraj[:, t0:t+1, :], rTraj[:, t0:t+1, :], 
                                                figtitle, filename, steadyIters=20000, fcIters_=20000, 
                                                tRange=tRange, tOptRange=(.5, 1.), noAva=True,
                                                adaptStr=adaptStr, truePid=truepId) #, deltaRange=(.13, .05)) #, noAva=True)
                            
                            #continue
                            '''
                            #raise ValueError()
                            
                            if len(truepId) == 1 and truepId[0]==1:
                                #aTraj =  # 128, epLen, 1, 1
                                bTraj_ = np.concatenate(((1/alpha)*np.ones(bTraj.shape), bTraj), axis=3)
                                binNum = 20
                            elif len(truepId) == 1 and truepId[0]==0:
                                assert tMid > 0.
                                bTraj_ = np.concatenate((bTraj, 1/tMid*np.ones(bTraj.shape)), axis=3)
                                binNum = 20
                            else:
                                bTraj_ = bTraj
                                binNum = 10
                                
                            stepBounds = (0, windowLen-1) #(1489, 1514) #(1489, 1514) #(0, 200) #(0, episodeLen-1)
                            stepNum = 25
                            
                            plot2dhist.plot2Dhist_(bTraj_, qTraj, mTraj, iMinTraj, stepBounds, filename, stepNum=stepNum, 
                                                   binNum=binNum, tRange=tRange) 
                            '''
                            # 2dhist
                            
                    if (t_+1) % windowLen == 0 or t_+1 == episodeLen:
                        NID = world.nAgents
                        aID = alpha #float(world.agents[0].alpha)
                        eID = float(world.eps)
                        seedID = seed
                        actualSeed = startSeed + seed
                        
                        if adaptBS: # and world.BSpool:
                            #bsID = str(adaptBS)[0] + str(world.BSdist)[0] + str(lrbs)
                            #elif adaptBS and not world.BSpool:
                            
                            bsID = str(adaptBS)[0] + 'None' + str(lrbs) #str(world.agents[0].lrBs)
                            
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
                                
                            if evoStop < np.inf and evoConst > 0:
                                # add wPerturb
                                bsID += '({},{},{},{})'.format(evoConst, spinScatterNum, tempScatterStrength, perturbSeed)
                            elif evoConst == 0:
                                bsID += '({},{},{})'.format(spinScatterNum, tempScatterStrength, perturbSeed)
                                
                            if world.nrandom:
                                bsID += 'nr'
                                
                            if world.spinOnly:
                                bsID += 'sOnly'
                                
                            if world.numOfi > 1:
                                bsID += 'n{}'.format(world.numOfi)
                                
                            if world.BSpool:
                                bsID += '-' + str(world.BSdist)[0] + str(pool_p[0])[0] + str(pool_p[1])[0]
                                
                        else:
                            bsID = str(adaptBS)[0] + str(tMid)
                        
                        #bTraj = np.squeeze(bTraj, axis=3)
                        
                        #print('avgR[:10]', avgRTraj_[:10])
                        #print('avgR[-10:]', avgRTraj_[-10:])
                        #raise ValueError()
                        adaptStr_ = np.array(adaptStr)[truepId]
                        
                        if not withPerturb:
                            pkl_.record_results(world, NID, aID, eID, bsID, seedID, actualSeed, t_+1, seedJ,
                                                bTraj, qTraj, iMinTraj, mTraj, avgRTraj_, rTraj, sTraj, 
                                                window=dumping, foldername_=foldername_, adaptStr_=adaptStr_)
                        
                        elif t_ > perturbStart or (perturbSeed==5 and t_ > evoStop): #t_ > evoStop:
                            pkl_.record_results(world, NID, aID, eID, bsID, seedID, actualSeed, t_+1, seedJ,
                                                bTraj, qTraj, iMinTraj, mTraj, avgRTraj_, rTraj, sTraj, 
                                                window=dumping, foldername_=foldername_, to_record=['m', 'avgR'],
                                                adaptStr_=adaptStr_)
                        
                        if dumping:
                            bTraj, qTraj, iMinTraj, mTraj, avgRTraj_, rTraj, sTraj, f0Traj, isEventTraj = pkl_.reinit_arr(nAgents, windowLen, 1)
                            try:
                                bTraj = np.zeros((nAgents, windowLen, numSeeds, len(truepId)))
                            except:
                                pass
                            
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
            '''
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
            '''