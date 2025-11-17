# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:06:41 2024

@author: nixie
"""

import numpy as np
import itertools
import math
import csv


# System Variations

ismax = False #True, False (default)
nAgents = 256

'''
# Measurement I
to_write, lastiters = 'TVD-gap', None

adaptBS, withPerturb, tMid = True, True, 0.
for alpha in [0.3, 0.5, 0.7, 0.75, 0.8, 0.9]:
    for (isInit, tPrtbRange) in [(True, [0.]), (False, np.arange(-6., 7., .5))]:
        for tempScatterStrength in tPrtbRange:
            tempScatterStrength = np.round(tempScatterStrength, 2)
'''

'''
# Measurement II    
to_write, lastiters = 'q,R', 500 #None #100

isInit, tempScatterStrength = False, ''
for alpha in [0.3, 0.5, 0.7, 0.75, 0.8, 0.9]:
    for (adaptBS, withPerturb, tMidRange) in [(False, False, np.arange(0., 1., .05)), (True, False, [0.]), (True, True, [0.])]:    
        for tMid in tMidRange:
            tMid = np.round(tMid, 2) 
'''            

'''
nAgents = 256
alpha = .3

adaptBS, tMid = True, 0.
#adaptBS, tMid = False, .25
#withPerturb, isInit, tempScatterStrength = False, False, ''
withPerturb, isInit, tempScatterStrength = True, False, 6.
#withPerturb, isInit, tempScatterStrength = True, True, 0.
'''


# Measurement I
# Measurement II    
to_write, lastiters = 'q,R', 500 #None #100

isInit, tempScatterStrength = False, 0.
for alpha in [.3, .9]: #[0.3, 0.5, 0.7, 0.75, 0.8, 0.9]:
    for (adaptBS, withPerturb, tMidRange) in [(False, False, np.arange(0., 1., .05)), (True, False, [0.]), (True, True, [0.])]:    
        for tMid in tMidRange:
            tMid = np.round(tMid, 2) 
            
            # File Names
            
            if adaptBS and withPerturb and isInit: 
                
                tsEnd = 10000 # init: 10000 # warm: 12000
                tsStart = 4000 # init: 4000 # warm: 6000
                aLagMax = 5000 # init: 5000 # warm: 5000
                
                datafoldername = 'compressed/chaos_N{}TpInit/'.format(nAgents)
                
                datafilename = 'N{}a{}T{}{}_ts{}k,{}k_alag{}'.format(nAgents, alpha, 'adapt', tempScatterStrength,
                                                              tsStart//1000, tsEnd//1000, aLagMax)
            
                param_list = [alpha, datafoldername[-6:-1], tempScatterStrength]
            
            elif adaptBS and withPerturb: #_prtb
                tsEnd = 15000 
                tsStart = 10000 
                aLagMax = 4000 
                
                datafoldername = 'compressed/chaos_N{}a{}TprtbX/'.format(nAgents, str(alpha)[2:])
                
                datafilename = 'N{}a{}T{}{}_ts{}k,{}k_alag{}'.format(nAgents, alpha, 'adapt', tempScatterStrength,
                                                              tsStart//1000, tsEnd//1000, aLagMax)
                
                param_list = [alpha, datafoldername[-6:-1], tempScatterStrength]
                
            elif adaptBS: #_adapt
                tsStart = 30000
                tsEnd = 150000
                aLagMax = 96000 
                datafoldername = 'compressed/chaos_N{}a{}Tadapt/'.format(nAgents, str(alpha)[2:])
                
                datafilename = 'N{}a{}T{}_ts{}k,{}k_alag{}'.format(nAgents, alpha, 'adapt',
                                                            tsStart//1000, tsEnd//1000, aLagMax)
                
                param_list = [alpha, datafoldername[-6:-1], tempScatterStrength]
                
            else: #_const
                tsStart = 10000 #1024: 60000 # _F, _T: 60000, _prtb: 72*, 96, 108000
                tsEnd = 100000 #1024: 156000 # _F: 156000, _T(a.7): 252000, _T: 300000, _prtb: 120000
                aLagMax = 72000 #1024: 77000
                datafoldername = 'compressed/chaos_N{}a{}Tconst/'.format(nAgents, str(alpha)[2:])
                
                datafilename = 'N{}a{}T{}_ts{}k,{}k_alag{}'.format(nAgents, alpha, tMid,
                                                            tsStart//1000, tsEnd//1000, aLagMax)
            
                param_list = [alpha, datafoldername[-6:-1], tMid]
            
            print('param_list:', param_list)
            
            ###############################################################################
            
            # Read files
            
            try:
                with open(datafoldername + datafilename + '.npy', 'rb') as f:
                    hyperparams = np.load(f, allow_pickle=True)
                    t0s = np.load(f)
                    lags = np.load(f)
                    acvfCumt0 = np.load(f)
                    acvfCum = np.load(f)
                    mTrajAll = np.load(f)
                    avgRTrajPlot = np.load(f)
                    
                nAgents, adaptBS, alpha, tMid, JRange, sRange, \
                withPerturb, record_evoStop, tempScatterStrength_, perturbSeedRange, \
                startSteady, tEnd, steadyIters0, steadyIters1, steadyIters = hyperparams
            except:
                print('file missing, skip')
                
                continue
            
            if to_write == 'q,R':
                
                #avgRTrajPlot.shape
                #avgRTrajPlot = avgRTrajPlot[-10000:, :, :]
                #if ismax:
                #    meanRTraj = np.max(avgRTrajPlot, axis=0).flatten()
                #if lastiters is None:
                #    meanRTraj = avgRTrajPlot.mean(axis=0).flatten()
                    
                if withPerturb:
                    meanRTraj = avgRTrajPlot[-lastiters:, :].mean(axis=0).mean(axis=1).flatten() # 1, 25, 5
                else:
                    meanRTraj = avgRTrajPlot[-lastiters:,:].mean(axis=0).flatten()
                        
                with open('compressed/R_avg_1-25.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    '''if adaptBS and withPerturb:
                        tMid = tempScatterStrength
                    elif adaptBS:
                        tMid = 'adapt'
                    '''
                    
                    # [alpha, tMid, q=acvfCum[0]
                    writer.writerows([param_list + list(meanRTraj)])
    
                    
            elif to_write == 'TVD-gap':
            
                if lastiters is not None:
                    avgRTrajPlot = avgRTrajPlot[-lastiters:, :, :]
                
                xmin = avgRTrajPlot.flatten().min() #min(dist1.min(), dist2.min())
                xmax = avgRTrajPlot.flatten().max() #max(dist1.max(), dist2.max())
                
                if withPerturb:
                    k_range = avgRTrajPlot.shape[1]
                    ij_range = avgRTrajPlot.shape[2]
                else:
                    k_range = 1
                    ij_range = avgRTrajPlot.shape[1]
                    
                pairwise_dist = np.zeros((k_range, math.comb(ij_range, 2)))
                
                for k in range(k_range):
                    ij = 0
                    
                    for i,j in itertools.combinations(range(ij_range), 2):
                    
                        if not withPerturb:
                            dist1 = avgRTrajPlot[:, i, k]
                            dist2 = avgRTrajPlot[:, j, k]
                        else:
                            dist1 = avgRTrajPlot[:, k, i]
                            dist2 = avgRTrajPlot[:, k, j]
                            
                        # _Wasserstein
                        #pairwise_dist += [stat.wasserstein_distance(dist1, dist2)]
                        #p = 1 # p>1: care more abt outliers? left tails transport assigned more weights?
                        #pairwise_dist += [1/len(dist1) * ((np.abs(np.sort(dist1)-np.sort(dist2))**p).sum())**(1/p)]
                        
                        # _TV
                        binnum = 20
                        hist1, binEdges1 = np.histogram(dist1, bins=np.linspace(xmin, xmax, binnum), density=True)
                        hist2, binEdges2 = np.histogram(dist2, bins=np.linspace(xmin, xmax, binnum), density=True)
                    
                        p1 = hist1*np.diff(binEdges1) # pmf; p(dx) = f(x)*dx
                        p2 = hist2*np.diff(binEdges2)
                        
                        #pairwise_dist += [np.abs(p1 - p2).sum()] #np.abs(p1 - p2).max()
                        pairwise_dist[k, ij] = np.abs(p1 - p2).sum()
                        
                        ij += 1
                
                if withPerturb:
                    pairwise_dist = pairwise_dist.T
                #pairwise_dist = pairwise_dist.T
                
                #np.savetxt('compressed/meanR_F.csv', pairwise_dist, delimiter=',')
                
                f=open('compressed/tvd_raw_10-25.csv','a')
                if withPerturb:
                    f.write("a:{}T:{}\n".format(alpha, tempScatterStrength))     
                else:
                    f.write("a:{}T:{}\n".format(alpha, tMid))
                
                np.savetxt(f, pairwise_dist, delimiter=',')
                
                #f.write("\n")
                f.close()
                
                # add median, write DF
                '''
                df_write = np.append(param_list, np.median(pairwise_dist, axis=0)).reshape((1, -1))
                
                f=open('compressed/tvd_med_1-25.csv', 'a')
                np.savetxt(f, df_write, delimiter=',')
                f.close()
                '''
                
                with open('compressed/tvd_med_1-25.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows([param_list + list(np.median(pairwise_dist, axis=0))])
                
                '''
                # read DF
                df_ = pd.read_csv(, col_header = 2)
                df_[head_a] # one subplot, (T, 25)
                np.stats(df_[head_a]) # (T, 5)
                
                for stat_id in range(5):
                    plt.plot(Tarr, np.stats(df_[head_a])[:, stat_id])
                '''
                
            else:
                raise NotImplementedError()