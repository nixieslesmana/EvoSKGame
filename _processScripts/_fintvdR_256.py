# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:06:41 2024

@author: nixie
"""

import numpy as np

nAgents = 256
alpha = .3

#adaptBS, tMid = True, 0.
adaptBS, tMid = False, .25
withPerturb, tempScatterStrength = False, ''
#withPerturb, tempScatterStrength = True, 0.

#to_write, lastiters = 'TVD-gap', None
to_write, lastiters = 'q,R', 500 #None #100

ismax = False #True, False (default)

if adaptBS and withPerturb: #_prtb
    tsEnd = 15000 # init: 10000 # warm: 12000
    tsStart = 10000 # init: 4000 # warm: 6000
    aLagMax = 4000 # init: 5000 # warm: 5000
    
    datafoldername = 'compressed/chaos_N{}a{}TprtbX/'.format(nAgents, str(alpha)[2:])
    #datafoldername = 'compressed/chaos_N{}TprtbXwarm/'.format(nAgents)
    
elif adaptBS: #_adapt
    tsStart = 30000
    tsEnd = 150000
    aLagMax = 96000 
    datafoldername = 'compressed/chaos_N{}a{}Tadapt/'.format(nAgents, str(alpha)[2:])
    
else: #_const
    tsStart = 10000 #1024: 60000 # _F, _T: 60000, _prtb: 72*, 96, 108000
    tsEnd = 100000 #1024: 156000 # _F: 156000, _T(a.7): 252000, _T: 300000, _prtb: 120000
    aLagMax = 72000 #1024: 77000
    datafoldername = 'compressed/chaos_N{}a{}Tconst/'.format(nAgents, str(alpha)[2:])

###############################################################################
if adaptBS and withPerturb:
    datafilename = 'N{}a{}T{}{}_ts{}k,{}k_alag{}'.format(nAgents, alpha, 'adapt', tempScatterStrength,
                                                  tsStart//1000, tsEnd//1000, aLagMax)
elif adaptBS and not withPerturb:
    datafilename = 'N{}a{}T{}_ts{}k,{}k_alag{}'.format(nAgents, alpha, 'adapt',
                                                tsStart//1000, tsEnd//1000, aLagMax)
else:
    datafilename = 'N{}a{}T{}_ts{}k,{}k_alag{}'.format(nAgents, alpha, tMid,
                                                tsStart//1000, tsEnd//1000, aLagMax)

with open(datafoldername + datafilename + '.npy', 'rb') as f:
    hyperparams = np.load(f, allow_pickle=True)
    t0s = np.load(f)
    lags = np.load(f)
    acvfCumt0 = np.load(f)
    acvfCum = np.load(f)
    mTrajAll = np.load(f)
    avgRTrajPlot = np.load(f)
    
nAgents, adaptBS, alpha, tMid, JRange, sRange, \
withPerturb, record_evoStop, tempScatterStrength, perturbSeedRange, \
startSteady, tEnd, steadyIters0, steadyIters1, steadyIters = hyperparams

if to_write == 'q,R':
    
    import csv
    #avgRTrajPlot.shape
    #avgRTrajPlot = avgRTrajPlot[-10000:, :, :]
    if ismax:
        meanRTraj = np.max(avgRTrajPlot, axis=0).flatten()
    elif lastiters is not None:
        meanRTraj = avgRTrajPlot[-lastiters:,:].mean(axis=0).flatten()
    else:
        meanRTraj = avgRTrajPlot.mean(axis=0).flatten()
        
    with open('compressed/meanR_F.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if adaptBS and withPerturb:
            tMid = tempScatterStrength
        elif adaptBS:
            tMid = 'adapt'
        writer.writerows([[alpha, tMid, acvfCum[0]] + list(meanRTraj)])
        
elif to_write == 'TVD-gap':

    import itertools
    import math
    
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
    
    f=open('compressed/meanR_F.csv','a')
    if withPerturb:
        f.write("a:{}T:{}\n".format(alpha, tempScatterStrength))     
    else:
        f.write("a:{}T:{}\n".format(alpha, tMid))
    np.savetxt(f, pairwise_dist, delimiter=',')
    #f.write("\n")
    f.close()
    
else:
    raise NotImplementedError()