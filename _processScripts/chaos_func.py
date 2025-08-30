# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:12:18 2024

@author: nixie
"""

import numpy as np
import matplotlib.pyplot as plt
from plotting import add_subplot_axes

def computeAcf_t0_Js(mTraj, lagList, t0List, steadyIters):
    
    ts = mTraj[:, -steadyIters:, :] # (128, 200, 1); 200=300:500
    
    numLags = len(lagList)
    numt0 = len(t0List)

    acvf_temp = np.zeros((numt0, numLags, 3)) # acvf.shape (15, 38)
        
    for t0ID in range(numt0): #range(len(t0List)):
        t0 = t0List[t0ID]
        t1 = t0 # the t_w
        
        for tauID in range(len(lagList)): #range(numLags):
            
            tau = lagList[tauID]
            
            t2 = t0 + tau
            
            #print(t1, t2, ts.shape[1])
            if t2 > ts.shape[1]-1:
                break
            
            X1 = ts[:, t1, :] # 128, 25
            X2 = ts[:, t2, :]
            
            acvf_temp[t0ID, tauID, 0] = np.sum(np.multiply(X1,X2), axis=0).item()
            acvf_temp[t0ID, tauID, 1] = np.sum(X1, axis=0).item()
            acvf_temp[t0ID, tauID, 2] = np.sum(X2, axis=0).item()
           
    return acvf_temp        # [numt0, numLags, 3]

def aggregateCorrelation(acvf_temp, divisor):
    
    if len(acvf_temp.shape) == 3: ## non stationary
        acvfCum_ = acvf_temp[:, :, 0] / divisor
        acvfCum_norm = acvf_temp[:, :, 0] / divisor - (acvf_temp[:, :, 1] / divisor)*(acvf_temp[:, :, 2] / divisor)
        
    elif len(acvf_temp.shape) == 2: ## 2: stationary
        acvfCum_ = acvf_temp[:, 0] / divisor
        acvfCum_norm = acvf_temp[:, 0] / divisor - (acvf_temp[:, 1] / divisor)*(acvf_temp[:, 2] / divisor)
    
    else:
        raise AssertionError()
        
    return acvfCum_, acvfCum_norm

# Dimension of Correl data: (len(t0s), len(lags))
#corr_p = (lags, t0s, corr_norm, with_ylim, Corr_ylim)
#window_p = (startSteady, tEnd, steadyIters0, steadyIters1, steadyIters) --> steadyIters0,1=steadyIters

def plotResults(figfoldername, acvfCumt0, acvfCum, avgRTrajPlot, mTrajAll,
                nAgents, adaptBS, alpha, tMid, JRange, sRange,
                withPerturb, record_evoStop, tempScatterStrength, perturbSeedRange,
                startSteady, tEnd, steadyIters0, steadyIters1, steadyIters,
                lags, t0s, corr_norm=True, with_ylim=False, Corr_ylim={}):
    
    fig = plt.figure(figsize=(5*2, 4*2))
    ax = fig.add_subplot(2, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2) 
    
    for t0 in np.linspace(0, acvfCumt0.shape[0]-1, 10).astype(int):
        to_plot = acvfCumt0[t0, :]
        
        if t0 == 0:
            
            ax.plot(alpha*np.array(lags)[to_plot > 0.], to_plot[to_plot > 0.], color='red', alpha=.5, label='t0:{}'.format(t0s[t0]))
            ax1.plot(alpha*np.array(lags)[to_plot > 0.], to_plot[to_plot > 0.], color='red', alpha=.5, label='t0:{}'.format(t0s[t0]))
            
        elif t0 == (acvfCumt0.shape[0]-1)//2:
        
            ax.plot(alpha*np.array(lags)[to_plot > 0.], to_plot[to_plot > 0.], color='orange', alpha=.5, label='t0:{}'.format(t0s[t0]))
            ax1.plot(alpha*np.array(lags)[to_plot > 0.], to_plot[to_plot > 0.], color='orange', alpha=.5, label='t0:{}'.format(t0s[t0]))
        
        elif t0 == acvfCumt0.shape[0]-1:
            ax.plot(alpha*np.array(lags)[to_plot > 0.], to_plot[to_plot > 0.], color='green', alpha=.5, label='t0:{}'.format(t0s[t0]))
            ax1.plot(alpha*np.array(lags)[to_plot > 0.], to_plot[to_plot > 0.], color='green', alpha=.5, label='t0:{}'.format(t0s[t0]))
        
        else:
            ax.plot(alpha*np.array(lags)[to_plot > 0.], to_plot[to_plot > 0.], color='grey', alpha=1-(t0+1)/(2*acvfCumt0.shape[0]))
            ax1.plot(alpha*np.array(lags)[to_plot > 0.], to_plot[to_plot > 0.], color='grey', alpha=1-(t0+1)/(2*acvfCumt0.shape[0]))
        
    # t0AGGR: larger taus maybe inaccurate, different num of datas
    ax.plot(alpha*np.array(lags)[to_plot > 0.], acvfCum[to_plot > 0.], color='black', linestyle='--', label='stationary, aggr t0')
    ax1.plot(alpha*np.array(lags)[to_plot > 0.], acvfCum[to_plot > 0.], color='black', linestyle='--', label='stationary, aggr t0')
    
    ax.legend()
    ax1.legend()
    
    ax.set_xlabel('alpha*tau')
    ax1.set_xlabel('alpha*tau')
    secax1 = ax1.secondary_xaxis('top', functions=(lambda x: x/alpha, lambda x: x/alpha))
    secax1.set_xlabel('tau')
    
    if corr_norm:
        ax.set_ylabel('C(t0, t0+tau) normalized')                
    else:
        ax.set_ylabel('C(t0, t0+tau)')
    
    if with_ylim:
        ax.set_ylim(Corr_ylim[tMid])
        ax1.set_ylim(Corr_ylim[tMid])
    
    ax.set_xscale('log')
    ax.set_title('ts=mTraj[{}:{}]'.format(startSteady+tEnd-steadyIters0, startSteady+tEnd))
    
    ax = fig.add_subplot(2, 2, 3)
    for js in range(avgRTrajPlot.shape[1]):
        for test in range(avgRTrajPlot.shape[2]):
            ax.plot(range(startSteady+tEnd-steadyIters1, startSteady+tEnd), avgRTrajPlot[:, js, test], alpha=.5)
        
    ax.plot(range(startSteady+tEnd-steadyIters1, startSteady+tEnd), np.average(avgRTrajPlot, axis=(1,2)), 
            color='black', linestyle='--')
    ax.set_xlabel('iter')
    ax.set_ylabel('<R>(Js)')
    
    gapR_all = avgRTrajPlot.mean(axis=0).flatten().max() - avgRTrajPlot.mean(axis=0).flatten().min()
    
    gapR_Js = avgRTrajPlot.mean(axis=0).max(axis=1) - avgRTrajPlot.mean(axis=0).min(axis=1)
    ax.set_title('aggr Js avgR:{}, \n gapR_prtb:{}, gapR_Jsprtb:{}'.format(np.round(avgRTrajPlot.mean(), 4), 
                                                                        np.round(gapR_Js.mean(), 4),
                                                                        np.round(gapR_all, 4)))
    
    ax = fig.add_subplot(2, 2, 4)
    mSteadyDraw = mTrajAll.shape[1]
    for i in range(mTrajAll.shape[0]):
        ax.plot(range(startSteady+tEnd-mSteadyDraw, startSteady+tEnd), mTrajAll[i, :, :], alpha=.2)
        
    ax.set_ylabel('m_i(t)')
    ax.set_xlabel('iter')
    ax.set_ylim((-1.05, 1.05))
    ax.set_xlim([startSteady+tEnd-mSteadyDraw, startSteady+tEnd-mSteadyDraw+mSteadyDraw*1.5])
    ax.set_title('m, {} samples i, aggr Js'.format(mTrajAll.shape[0]))
    
    plt.tight_layout()
    
    subpos = [0.7, 0., 0.3, .95]
    inset_ax = add_subplot_axes(fig.get_axes()[-1], subpos)
    data = mTrajAll[:, -mSteadyDraw:, :].flatten()
    inset_ax.hist(data, bins=50, orientation='horizontal')
    inset_ax.set_xticklabels([])
    inset_ax.set_yticklabels([])
    inset_ax.set_ylim((-1.05, 1.05))
    
    '''### m distrib. fixed Js. histogram
    nrows, ncols = 5, 5
    fig = plt.figure(figsize=(5*nrows, 4*ncols))
    for JsId in range(mTrajAll.shape[-1]):
        ax = fig.add_subplot(nrows, ncols, JsId+1)
        data = mTrajAll[agentSamples, -steadyIters:, JsId].flatten()
        ax.hist(data, bins=50)
        ax.set_xlim((-1., 1.))
        ax.set_title('JsId={}'.format(JsId))
    
    plt.suptitle('N{}, a{},T{}: PHASE (Age,noAge) X (Break,noBreak)'.format(nAgents, alpha, tMid))
    plt.tight_layout()
    '''
    
    if not adaptBS:
        plt.suptitle('N{}, a{}, J{}, s{}, T{}: PHASE (Age,noAge) X (Break,noBreak)'.format(nAgents, alpha, 
                                                                                           JRange, sRange, tMid))
        plt.tight_layout()
    
        figname = 'N{}a{}T{}_lagMax{}_iter{}k,{}k'.format(nAgents, alpha, tMid, lags[-1], 
                                                              (startSteady+tEnd-steadyIters)//1000, (startSteady+tEnd)//1000)
        
        
    else:
        plt.suptitle('N{}, a{}, J{}, s{}, T{}'.format(nAgents, alpha, JRange, sRange, 'adapt'))
        plt.tight_layout()
        
        figname = 'N{}a{}T{}_lagMax{}_iter{}k,{}k'.format(nAgents, alpha, 'adapt', lags[-1], 
                                                          (startSteady+tEnd-steadyIters)//1000, (startSteady+tEnd)//1000)
        
    if withPerturb and record_evoStop:
        figname += '_stop'
    elif withPerturb and not record_evoStop:
        figname += '_test({},{}-{})'.format(tempScatterStrength, perturbSeedRange[0], perturbSeedRange[-1])
    else:
        figname += '_train'
        
    plt.savefig(figfoldername + figname + '.png')
    plt.close()
    
def computeAcf(mTraj, isStationary=True, numLags=7, steadyIters=1000):
    
    ts = mTraj[:, -steadyIters:, :]
    
    acvf = np.zeros(numLags)
    acf = np.zeros(numLags)
    
    for tau in range(numLags):
        
        #print('tau:', tau)
        
        t1 = 0
        t2 = t1 + tau
        
        while t2 < ts.shape[1]:
            
            #print('t1,t2:', t1, t2)
            
            if t1 == 0:
                X1 = ts[:, t1, :]
                X2 = ts[:, t2, :]
            else:
                X1 = np.concatenate((X1, ts[:, t1, :]), axis=1)
                X2 = np.concatenate((X2, ts[:, t2, :]), axis=1)
            
            if not isStationary:
                break
            
            t1 += 1
            t2 += 1
        
        if tau==0:
            EX1 = np.average(X1, axis=1) # mX1 = mX2 (256, )
            m = np.average(EX1)
            
        EX1X2 = np.average(np.multiply(X1,X2), axis=1) # (256, )
        
        cX1 = np.average(np.multiply(X1,X1), axis=1) # (256, )
        cX2 = np.average(np.multiply(X2,X2), axis=1) # (256, )
        
        acvf[tau] = np.average(EX1X2, axis=0)
        acf[tau] = np.average(np.divide(EX1X2, np.sqrt(np.multiply(cX1, cX2))), axis=0)
        
    return acf, acvf, m