# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:25:45 2024

@author: nixie
"""

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import NullFormatter
import numpy as np
#from scipy import stats as st
#from scipy.optimize import curve_fit
from scipy import stats
import powerlaw
import math

import pandas as pd

from SKGame_aPoolA import computeAcf_ # (= SKGame.computeAcf)

'''Power Law is scale invariance
--------------------------------
scaling by a constant simply multiplies the original power-law relation by 
another constant, i.e. f(cx) = Kf(x)

Suppose f(x) = a*x**-k.
log f(x) = -k*log a - k*log x

f(x) = a*exp(-x)
log f(x) = log a - x = log a - exp(log x)

# hist data -> produce exp decreasing line
X = np.linspace(0, .6, 11)
X = (X[:-1] + X[1:])/2
Y = [700, 420, 500, 300, 200, 130, 100, 80, 30, 10]

# dummy data -> produce straight line
b = 2
k = 1.5
X = range(1, 11)
Y = [b*x**-k for x in X]

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)

ax.scatter(X, Y, color='blue')

ax.set_yscale('log')
ax.set_xscale('log')

x = np.arange(20)
y1 = 2*x**3
y2 = 2**x

plt.plot(y1)
plt.plot(y2)
'''

def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power)) #np.exp(start) #np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power)) #np.exp(stop) #np.power(stop, 1/float(power))
    return np.power( np.linspace(start, stop, num=num), power) 

class SqueezedNorm(matplotlib.colors.Normalize):
    
    #vmin=Tmin, vmax=Tmax, mid=scaler_*Tmax, s1=2, s2=2
    
    def __init__(self, vmin=None, vmax=None, mid=0, s1=2, s2=2, clip=False):
        self._vmin = vmin # minimum value
        self.mid  = mid  # middle value
        self._vmax = vmax # maximum value
        self.s1=s1; self.s2=s2
        f = lambda x, zero,vmax,s: np.abs((x-zero)/(vmax-zero))**(1./s)*0.5
        self.g = lambda x, zero,vmin,vmax, s1,s2: f(x,zero,vmax,s1)*(x>=zero) - \
                                             f(x,zero,vmin,s2)*(x<zero)+0.5
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        #print(self.vmax, self.mid)
        #print((1./self.s1)*.5, (1./self.s2)*.5)
        r = self.g(value, self.mid,self.vmin,self.vmax, self.s1,self.s2)
        
        return np.ma.masked_array(r)

class MidpointNorm(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)
        
        # 0 is in the middle but is not whitened

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# inset plotting func    
def add_subplot_axes(ax, rect, axisbg='w'):
    
    fig_ = plt.gcf()
    
    box = ax.get_position()
    width = box.width
    height = box.height
    
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig_.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    
    x = infig_position[0]
    y = infig_position[1]
    
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    
    subax = fig_.add_axes([x,y,width,height])#,axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    
    return subax

def smape(actual, predicted) -> float: 
  
    # Convert actual and predicted to numpy 
    # array data type if not already 
    if not all([isinstance(actual, np.ndarray),  
                isinstance(predicted, np.ndarray)]): 
        actual, predicted = np.array(actual), 
        np.array(predicted) 
  
    return round( 
        np.mean( 
            np.abs(predicted - actual) / 
            ((np.abs(predicted) + np.abs(actual))) 
        ), 2) 

def mase(Actual, Predicted):
    values = []
    for i in range(1, len(Actual)):
        values.append(abs(Actual[i] - Predicted[i]) / (abs(Actual[i] - Actual[i - 1]) / len(Actual) - 1))
    return np.round(np.mean(values), 2)

def criticalExponent(data, ax1=None, name=None, dataType='discrete', ccdf=True, divisor=5,
                     loggedPdf=False, pmf=True, maxSize=50, xmin=2, fit_method='KS', fit_discrete=False):
    
    assert dataType == 'discrete'
    
    if fit_method == 'MLE':
        fit_method = 'Likelihood'
    
    if maxSize is not None:
        data = data[data<=maxSize]
    else:
        maxSize = max(data)
        data = data[data<=maxSize]
    data_ = data[:]
    data = data[data>=xmin]
    
    slopeArr = np.zeros(divisor)
    pArr = np.zeros(divisor)
    seArr = np.zeros(divisor)
    rsqArr = np.zeros(divisor)
    ydArr = []
    tplArr = []
    
    if len(data) < 5:
        return (slopeArr, pArr, seArr, rsqArr, ydArr, tplArr)
    
    d = np.diff(np.unique(data)).min()
    
    left_of_first_bin = data.min() - float(d)/2
    right_of_last_bin = data.max() + float(d)/2
    
    P, binEdges = np.histogram(data, density=True, bins=np.arange(left_of_first_bin, right_of_last_bin + d, d))
    if pmf:
        P /= 1/np.diff(binEdges)
    
    C = np.cumsum(P)
    X = (binEdges[:-1] + binEdges[1:])/2
    X = X.astype('int')
    
    if loggedPdf:
        ### pdf log ####
        binEdges = np.logspace(np.log10(xmin), np.log10(maxSize), 20) #np.logspace(0, 4, 50)
        Plog, _ = np.histogram(data, density=True, bins=binEdges)
        if pmf:
            widths = (binEdges[1:] - binEdges[:-1])
            Plog /= 1/widths
        
        Xlog = (binEdges[:-1] + binEdges[1:])/2
        Xlog = Xlog.astype('int')
    else:
        Xlog, Plog = X, P
    
    if ccdf:
        Y = 1-C
    else:
        Y = Plog
        X = Xlog
    
    #bool_ = np.prod(np.vstack([X>=np.min(X), X<=np.max(X) , Y>=1e-6]), axis=0).astype('bool') 
    bool_ = np.prod(np.vstack([X>=np.min(X), X<=np.max(X), Y>0]), axis=0).astype('bool')
    X_ = X[bool_]
    Y_ = Y[bool_]
     
    posStart = 0 #.1
    xlogStart = (1-posStart)*(np.log(X_)[0]) + posStart*(np.log(maxSize)) #(X_)[-1])
    idStart = np.argmin(np.abs(np.log(X_) - xlogStart))
    
    count = -1
    idEnd = idStart
    
    idScatter = None
    
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    
    for pos in np.linspace(posStart, 1, divisor): #idEnd in rangeId: #range(len(X_)//divisor, len(X_), len(X_)//divisor):
        count += 1
        
        pos = np.round(pos, 1)
        xlogEnd = (1-pos)*(np.log(X_)[idStart]) + pos*(np.log(maxSize)) #(np.log(X_)[-1])
        
        previd = idEnd
        idEnd = np.arange(0, len(X_))[np.log(X_) <= xlogEnd][-1] #np.argmin(np.abs(np.log(X_) - xlogEnd))
        #if ax1 is not None:
        #    print(xlogEnd, idEnd)
        if idEnd <= idStart:
            continue
        
        #idScatter = np.arange(idStart, idEnd)
        if idEnd - previd > 100:
            replace = False
        elif idEnd - previd > 0:
            replace = True
        else:
            continue
            
        if idScatter is None:
            idScatter = np.random.choice(np.arange(previd, idEnd), replace=replace, size=100)
            idScatter_ = idScatter
        else:
            idScatter = np.random.choice(np.arange(previd, idEnd), replace=replace, size=100)
            idScatter_ = np.concatenate((idScatter_, idScatter))
            
        try:
            #slope, intercept, r, p, se = stats.linregress(np.log(X_[idScatter]), np.log(Y_[idScatter]))
            slope, intercept, r, p, se = stats.linregress(np.log(X_[idScatter_]), np.log(Y_[idScatter_]))
            
            Yfitted = slope * np.log(X_) + intercept
            
            if p < .05:
                slopeArr[count] = slope
                pArr[count] = p
                seArr[count] = se
                rsqArr[count] = r**2
                
            if ax1 is not None:
                ax1.axvline(x=X_[idEnd], color='grey', alpha=.5, ls='--')
    
                if p < .05:
                    #ax1.plot(X_[previd:idEnd], np.exp(Yfitted)[previd:idEnd], alpha=.5, color=colors[count//math.ceil(divisor/len(colors))], 
                    #         label=(np.round(slope, 2), np.round(r**2, 2))) #np.round(idEnd/len(X_), 2)))
                    ax1.plot(X_[idStart:idEnd], np.exp(Yfitted)[idStart:idEnd], alpha=.5, color=colors[count//math.ceil(divisor/len(colors))], 
                             label=(np.round(slope, 2))) #, np.round(r**2, 2))) #np.round(idEnd/len(X_), 2)))
                    
        except:
            continue
        
    if ax1 is not None:
        #print('in scatter..')
        ax1.scatter(X_, Y_, color='grey', alpha=.1, s=15)
        ax1.scatter(X_[idScatter_], Y_[idScatter_], color='black', alpha=.1, s=15)
    
    # Cutoff manual fit (fix slope)
    if ccdf:
        slope = .5
    else:
        slope = 1.5
        
    ydArr = np.round(-np.log(Y_/(X_**(-slope)))/X_, 4)
    #ydArr = np.round(-np.log(Y_/(X_**(-slope))), 4)
    
    ydArr = np.unique(ydArr[idStart:])
    
    try:
        results = powerlaw.Fit(data_, discrete=fit_discrete, xmin=xmin, fit_method=fit_method) #X_[idStart:], discrete=True, xmin=2)
        #results.truncated_power_law
        fittedDistr = results.truncated_power_law
        slope_d = fittedDistr.parameter1
        beta_d = fittedDistr.parameter2
        KSscore = fittedDistr.KS(data_[data_>=xmin])
        MLEscore = fittedDistr.loglikelihoods(data_[data_>=xmin]).mean()
        tplR, tplp = results.distribution_compare('truncated_power_law', 'exponential', normalized_ratio=True)
        
        if ax1 is not None:
            print('TPL vs Exp R (>0?), p, KS:', tplR, tplp, KSscore, '| slope, cutoff:', slope_d, beta_d)
            
            plR, plp = results.distribution_compare('truncated_power_law', 'power_law', normalized_ratio=True)
            
            fittedPL = results.power_law
            slopePL = fittedPL.alpha
            betaPL = 0
            #KSPL = fittedPL.KS(data[data>=xmin])
            KSPL = fittedPL.KS(data_[data_>=xmin])
            plexp_R, plexp_p = results.distribution_compare('power_law', 'exponential', normalized_ratio=True)
            
            print('TPL vs PL R (>0?), p:', plR, tplp, '| PLslope, PLcutoff:', slopePL, betaPL)
            
            if ccdf and tplR > 0 and tplp < .01:     
                #ccdf_d = 1-fittedDistr.cdf(data)
                ccdf_d = 1-fittedDistr.cdf(data_)
                #ax1.scatter(data[data>=xmin], ccdf_d, s=3, color='blue', alpha=.1)
                ax1.scatter(data_[data_>=xmin], ccdf_d, s=3, color='blue', alpha=.1)
                
                if plR < 0 and plp < .01:
                    #ccdf_d = 1-fittedPL.cdf(data)
                    ccdf_d = 1-fittedPL.cdf(data_)
                    #ax1.scatter(data[data>=xmin], ccdf_d, s=3, color='red', alpha=.1)
                    ax1.scatter(data_[data_>=xmin], ccdf_d, s=3, color='red', alpha=.1)
                
            elif tplR > 0 and tplp < .01:
                #pdf_d = fittedDistr.pdf(data)
                pdf_d = fittedDistr.pdf(data_)
                #ax1.scatter(data[data>=xmin], pdf_d, s=3, color='blue', alpha=.2, label=(np.round(slope_d,2), np.round(beta_d,4), np.round(KSscore, 4)))
                ax1.scatter(data_[data_>=xmin], pdf_d, s=3, color='blue', alpha=.1, label=(np.round(slope_d,2), np.round(beta_d,4), np.round(KSscore, 4)))
                
                if plR < 0 and plp < .01:
                    
                    #pdf_d = fittedPL.pdf(data)
                    pdf_d = fittedPL.pdf(data_)
                    #ax1.scatter(data[data>=xmin], pdf_d, s=3, color='red', alpha=.2, label=(np.round(slopePL,2), np.round(betaPL,4), np.round(KSPL, 4)))
                    ax1.scatter(data_[data_>=xmin], pdf_d, s=3, color='red', alpha=.1, label=(np.round(slopePL,2), np.round(betaPL,4), np.round(KSPL, 4)))
                    
        tplArr = [-slope_d, beta_d, tplR, tplp, MLEscore, KSscore]
        
    except:
        pass
        
    if ax1 is not None:
        #ax1.set_xlim((min(X_), maxSize))
        ax1.set_ylim((Y_.min(), 1))
        print('removed xlim,ylim')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.legend(prop={'size': 7}, framealpha=.5, loc='lower left', ncols=2)
        if ccdf:
            ax1.set_title(name + '_' + 'log-ccdf')
        else:
            ax1.set_title(name + '_' + 'log-pdf')    
    
    return (slopeArr, pArr, seArr, rsqArr, ydArr, tplArr)
    
        

def getCriticality(fig, ax_, rowID, colID, ncol, data, dataType, name, 
                   insetDict, insetDel=0, qtmin=0., qtmax=1., loggedPdf=True):
    
    if len(np.unique(data)) <= 1:
        dataType = 'dirac'
    
    if dataType == 'discrete':
        
        d = np.diff(np.unique(data)).min()
        #d=1
        left_of_first_bin = data.min() - float(d)/2
        right_of_last_bin = data.max() + float(d)/2
        
        P, binEdges = np.histogram(data, density=True, bins=np.arange(left_of_first_bin, right_of_last_bin + d, d))
        P /= 1/np.diff(binEdges)
        
        C = np.cumsum(P)
        X = (binEdges[:-1] + binEdges[1:])/2
        X = X.astype('int')
        
        if loggedPdf:
            ### pdf log ####
            binEdges = np.logspace(0, 4, 50)
            widths = (binEdges[1:] - binEdges[:-1])
            Plog, _ = np.histogram(data, density=True, bins=binEdges)
            Plog /= 1/widths
            
            Xlog = (binEdges[:-1] + binEdges[1:])/2
            Xlog = Xlog.astype('int')
        else:
            Xlog, Plog = X, P
        
    elif dataType == 'continuous':
        P, binEdges = np.histogram(data, density=True, bins=30)
        P /= 1/np.diff(binEdges) #np.sum(P) 
        C = np.cumsum(P)
        X = (binEdges[:-1] + binEdges[1:])/2
    
    else: # all zeros
        
        P, binEdges = np.histogram(data, density=True, bins=1)
        C = np.cumsum(P)
        X = (binEdges[:-1] + binEdges[1:])/2
        #X = X.astype('int')
        
        Xlog, Plog = X, P
    
    for type_ in ['ccdf', 'pdf']:
        
        #print(type_)
        if type_ == 'pdf':
            #Y = P
            Y = Plog
            X = Xlog
            
        elif type_ == 'ccdf':
            Y = 1-C
            
        else:
            raise NotImplementedError()
        
        xmax = np.quantile(X, qtmax) #max(X)
        xmin = np.quantile(X, qtmin) # detect if one-tailed or two-tailed...
        
        bool_ = np.prod(np.vstack([X>=xmin, X<=xmax, Y>=1e-6]), axis=0).astype('bool') #bool_ = np.multiply(X>=xmin, X<=xmax)
        X_ = X[bool_]
        Y_ = Y[bool_]
        
        if len(X_) == 0:            
            colID += 2
            continue
        
        axID = ncol * rowID + colID - insetDel
        if axID in insetDict.keys():
            raise ValueError()
        insetDict[axID] = (X, Y, xmin, xmax)
        
        ax0 = ax_[rowID, colID]
        ax1 = ax_[rowID, colID+1]
        
        # fig, ax = plt.subplots((1,1))
        if type_ == 'pdf':
            ax0.scatter(X_, Y_, s=10, color='black')
            ax1.scatter(X_, Y_, s=10, color='black')
            m = -3. # -3.
            
        elif type_ == 'ccdf':
            ax0.scatter(X_, Y_, s=5, color='black')
            ax1.scatter(X_, Y_, s=5, color='black')    
            m = -2. # -1
        
        posStart = .2
        xlogStart = (1-posStart)*(np.log(X_)[0]) + posStart*(np.log(X_)[-1])
        idStart = np.argmin(np.abs(np.log(X_) - xlogStart))
        #idStart = 10
        #print(len(X_), idStart)

        divisor = 5
        assert divisor < 10
        
        for pos in np.linspace(posStart, 1, divisor): #idEnd in rangeId: #range(len(X_)//divisor, len(X_), len(X_)//divisor):
            pos = np.round(pos, 1)
            
            xlogEnd = (1-pos)*(np.log(X_)[idStart]) + pos*(np.log(X_)[-1])
            idEnd = np.argmin(np.abs(np.log(X_) - xlogEnd))
            
            if idEnd < idStart:
                continue
            
            try:
                slope, intercept, r, p, se = stats.linregress(np.log(X_[idStart:idEnd]), np.log(Y_[idStart:idEnd]))
                Yfitted = slope * np.log(X_) + intercept
                #print(p)
                if p < .05:
                    ax1.plot(X_[idStart:idEnd], np.exp(Yfitted)[idStart:idEnd], alpha=.5, label=(np.round(slope, 2)))
                                                                                                 #np.round(pos*(1-posStart), 2))) #np.round(idEnd/len(X_), 2)))
                    ax1.axvline(x=X_[idEnd], color='grey', alpha=.5, ls='--')
            #idStart = idEnd
            except:
                continue
        
        # np.round(min(X_), 4)
        ax0.axvline(x=min(X_), label='xmin=q{}'.format(qtmin), color='green', ls='--')
        # np.round(max(X_), 4)
        ax0.axvline(x=max(X_), label='xmax=q{}'.format(qtmax), color='orange', ls='--')
        #ax0.set_ylim(0., 1.)
        ax0.legend(prop={'size': 10}, loc='upper right', framealpha=.5)
        ax0.set_title(name + '_' + type_)
        
        ax1.axvline(x=min(X_), color='green', ls='--')
        ax1.axvline(x=max(X_), color='orange', ls='--')
        #ax1.set_ylim(0., 1.)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend(prop={'size': 10}, loc='upper right', framealpha=.5)
        ax1.set_title(name + '_' + 'log-'+ type_)
        
        colID += 2
        
    return insetDict

def insetCritical(fig, insetDict):
    # insets
    for subaxId in insetDict.keys():
        #print('subax index', subaxId)
        
        X, Y, xmin, xmax = insetDict[subaxId]
        # no bool (incl Y>1e-6)
        
        subpos = [0, 0, 0.4, 0.4]
        
        inset_ax = add_subplot_axes(fig.get_axes()[subaxId], subpos)
        inset_ax.scatter(X, Y, color='grey')
        inset_ax.axvline(x=xmin, color='green', ls='--')
        inset_ax.axvline(x=xmax, color='orange', ls='--')
        inset_ax.xaxis.set_major_formatter(NullFormatter())
        inset_ax.xaxis.set_minor_formatter(NullFormatter())
        inset_ax.yaxis.set_major_formatter(NullFormatter())
        inset_ax.yaxis.set_minor_formatter(NullFormatter())
        inset_ax.patch.set_alpha(0.5)
        
        inset_ax = add_subplot_axes(fig.get_axes()[subaxId+1], subpos)
        inset_ax.scatter(X, Y, color='grey')
        inset_ax.axvline(x=xmin, color='green', ls='--')
        inset_ax.axvline(x=xmax, color='orange', ls='--')
        inset_ax.set_xscale('log')
        inset_ax.set_yscale('log')
        inset_ax.xaxis.set_major_formatter(NullFormatter())
        inset_ax.xaxis.set_minor_formatter(NullFormatter())
        inset_ax.yaxis.set_major_formatter(NullFormatter())
        inset_ax.yaxis.set_minor_formatter(NullFormatter())
        #inset_ax.set_yticklabels([])
        #inset_ax.set_xticklabels([])
        inset_ax.patch.set_alpha(0.5)

def get_fdelta(fmin, delta):
    
    fdelta = np.zeros(len(fmin))
    
    f0 = fmin[0]
    fdelta[0] = f0
    for t in range(1, len(fmin)):
        if fmin[t] >= f0 - delta:
            f0 = fmin[t]
            
        fdelta[t] = f0
    
    return fdelta

def get_dArray(fmin_, q_l=0., q_u=1., numDeltas=None, d_l=None, d_u=None):
    
    diffs = np.diff(fmin_)
    diffs = diffs[diffs < 0]
    #d_u = np.abs(np.quantile(diffs, .05, method='closest_observation')) # diffs -> Exponential dist left tail!; too large -> v small avaSize
    #d_l = np.abs(np.quantile(diffs, .95, method='closest_observation')) # too small -> noAva
    
    # diffs = 1 - fmin_[1:] / fmin_[:-1] # problematic if fmin < 0
    
    if numDeltas is None:
        if d_l is not None and q_l == 0.:
            lb = d_l
            
        elif d_l is not None:
            assert q_l > 0.
            lb = min(d_l, np.quantile(diffs, q_l, method='closest_observation'))
        else:
            lb = np.quantile(diffs, q_l, method='closest_observation')
            
        if d_u is not None and q_u == 1.:
            ub = d_u
        elif d_u is not None:
            assert q_u < 1.
            ub = max(d_u, np.quantile(diffs, q_u, method='closest_observation'))
        else:
            ub = np.quantile(diffs, q_u, method='closest_observation')
            
        print('q_l, q_u, lb, ub:', q_l, q_u, lb, ub)
        
        diffs = diffs[diffs >= lb]
        diffs = diffs[diffs <= ub]
        
        return np.abs(diffs)
    
    #if d_l is not None: # d_l = -.003, 0 | -.05, -.003 
    #    diffs = diffs[diffs>d_l]
    #    diffs = diffs[diffs<d_u]
        
    #    return np.abs(diffs)
    
    return np.array([np.abs(np.quantile(diffs, q, method='closest_observation')) for q in np.linspace(q_l, q_u, numDeltas)])

def getAva(qTraj_, fdelta):

    imins_ = qTraj_ < np.expand_dims(fdelta, axis=0)
    sizeLow_ = np.sum(imins_, axis=0) # (100k, )
    isEvent_ = (sizeLow_ > 0).astype('bool') 
    
    startTimes = []
    endTimes = []
    sizes = []
    
    size = 0
    if len(np.where(isEvent_ == False)[0])>1: #0:
        FId = np.where(isEvent_ == False)[0][0]
        FId_ = np.where(isEvent_ == False)[0][-1]
    else:
        return sizes, startTimes, endTimes
    
    for t in range(FId+1, FId_+1): #len(isEvent_)):
        if isEvent_[t]:
            
            if size == 0:
                startTimes += [t-1]
                
            size += sizeLow_[t]
        else:
            if size > 0:
                sizes += [size]
                endTimes += [t]
            
            size = 0
    '''
    if size > 0:
        sizes += [size]
        endTimes += [t]
        
    if isEvent_[-1]: # last avalanche is not closed #endTimes[-1] == len(isEvent_) - 1:
        #print(isEvent_[-10:])
        #print(startTimes[-10:])
        #print(endTimes[-10:])
        sizes = sizes[:-1]
        startTimes = startTimes[:-1]
        endTimes = endTimes[:-1]
        #print('last ava not closed.')
        #raise ValueError()
    '''
    if isEvent_[-1] and endTimes[-1]==49999: # make sure endTimes
        print(FId_, np.where(isEvent_ == False)[0])
        print(startTimes[-10:])
        print(endTimes[-10:])
        print(sizes[-10:])
        raise ValueError()
        
    # find isEvent_ first False index
    # for t in range(index+1, len(isEvent_))
    if startTimes[0]==-1: #isEvent_[0]:
        print(np.where(isEvent_ == False)[0])
        print(isEvent_[:np.where(isEvent_ == False)[0][0]+1])
        print('start', startTimes[:10])
        print('end:', endTimes[:10])
        print('size:', sizes[:10])
        raise ValueError()
        
    return sizes, startTimes, endTimes
    
def AvalancheSize(nAgents, seed_, startSeed, episodeLen, qTraj, iMinTraj, figtitle, filename, 
                  steadyIters=5000, qtmin=0., ava_p=None): 
                  #, midLine_c=None, midLine_g=None):
    
    assert ava_p is not None

    fmin, d_u, d_l, q_u, q_l, numDeltas, loggedPdf = ava_p
    fgap = np.maximum.accumulate(fmin) # fdelta(delta=0)
    
    if d_u is None or d_l is None:
        fmin_ = fmin[-steadyIters:]
        dArr = get_dArray(fmin_, q_l, q_u, numDeltas)
        print('dArr', dArr)
            
    else:
        dArr = np.linspace(d_u, d_l, 5)
        
    nrow = len(dArr) #numt = 3
    ncol = 2 + 4
    
    fig, ax = plt.subplots(nrow, ncol, figsize=(5*ncol, 3*nrow))
    
    insetDict = dict()
    insetDel = 0
    
    rowID = 0
    
    qTraj_ = qTraj[:, :, seed_]
    ikilled = iMinTraj[:, 1:, seed_]
    print(ikilled.shape)
    append_ = np.zeros((ikilled.shape[0], 1))
    append_[np.argmin(qTraj_[:, -1])] = 1
    
    ikilled = np.concatenate((ikilled, append_), axis=1)
    print(ikilled.shape)
    
    for delta in dArr:
        
        fdelta = get_fdelta(fmin, delta)
        
        # % Visualize agentIDs in avalanche (below fdelta)
        imins = qTraj_ < np.expand_dims(fdelta, axis=0) #imins_ = qTraj_ < f_c
        
        # % Visualize avalanches given f_c
        imins_ = imins[:, -steadyIters:]
        print(imins_.shape)
        sizeLow_ = np.sum(imins_, axis=0) # (100k, )
        isEvent_ = (sizeLow_ > 0).astype('bool') 
        
        #print('isEvent:', np.sum(isEvent_))
        
        startTimes = []
        endTimes = []
        sizes = []
        
        size = 0
        for t in range(len(isEvent_)):
            if isEvent_[t]:
                
                if size == 0:
                    startTimes += [t-1]
                    
                size += sizeLow_[t]
            else:
                if size > 0:
                    sizes += [size]
                    endTimes += [t]
                
                size = 0
        
        if size > 0:
            sizes += [size]
            endTimes += [t]
        
        sizes = np.array(sizes) # every seq of ones compute sizes
        
        startTimes = (episodeLen - steadyIters) + np.array(startTimes)
        endTimes = (episodeLen - steadyIters) + np.array(endTimes)
        
        maxLen = min(len(startTimes), 80)
        if len(startTimes) > 0:
            printIters = min(episodeLen - startTimes[-maxLen], steadyIters)
        else:
            printIters = steadyIters
            
        ax_ = ax[rowID, 1]
        
        [ax_.axvline(x, linewidth=.5, color='black') for x in startTimes[-maxLen:]]
        [ax_.axvline(x, linewidth=.5, linestyle='--', color='red') for x in endTimes[-maxLen:]]
        
        ax_.scatter(range(episodeLen-printIters, episodeLen), fmin[-printIters:], \
                    color='grey', s=1, alpha=.1)
        ax_.scatter(range(episodeLen-printIters, episodeLen), fdelta[-printIters:], \
                    color='green', s=1, alpha=.1)
        ax_.scatter(range(episodeLen-printIters, episodeLen), fgap[-printIters:], \
                    color='blue', s=1, alpha=.1)
        
        ax_.set_xlim([episodeLen-printIters - printIters//50, episodeLen + printIters//50])
        
        if len(startTimes)> 0:
            tmax = np.argmax(sizes)
            #ax_.set_title('max avaSize at t={}:{}'.format(startTimes[tmax], endTimes[tmax]))
            
            try:
                ikilled_ = ikilled[:, startTimes[tmax]:endTimes[tmax+1]].astype(int) * 4
                #imins__ = ikilled_ + imins[:, startTimes[tmax]:endTimes[tmax+1]] #np.concatenate((imins[:, startTimes[tmax]:endTimes[tmax]], imins[:, startTimes[tmax+1]:endTimes[tmax+1]]), axis=1)
                #print(ikilled[:20])
                #print(imins[:, startTimes[tmax]:endTimes[tmax+1]][:20])
                imins__ = imins[:, startTimes[tmax]:endTimes[tmax+1]].astype(int) + ikilled_
                print('here1', startTimes[tmax], endTimes[tmax+1])
                print(ikilled.shape, ikilled_.shape, imins[:, startTimes[tmax]:endTimes[tmax+1]].astype(int).shape)
                '''
                print(np.argwhere(imins__[:, 1] == np.amax(imins__[:, 1]))) #list(imins__[:, 0]))
                print(np.argmax(ikilled_[:, 1], axis=0))
                print(np.max(imins__[:, 1]))
                print(np.argwhere(imins__[:, 0] == np.amax(imins__[:, 0]))) #list(imins__[:, 0]))
                print(np.argmax(ikilled_[:, 0], axis=0))
                print(np.max(imins__[:, 0]))
                '''
                
                ax_.set_title('max ava at t={}:{}; +1 {}:{}'.format(startTimes[tmax], endTimes[tmax], startTimes[tmax+1], endTimes[tmax+1]))
            except:
                ikilled_ = ikilled[:, startTimes[tmax-1]:endTimes[tmax]].astype(int) * 4
                #ikilled = iMinTraj[:, startTimes[tmax-1]:endTimes[tmax], seed_]
                #imins__ = ikilled + imins[:, startTimes[tmax-1]:endTimes[tmax]] #np.concatenate((imins[:, startTimes[tmax-1]:endTimes[tmax-1]], imins[:, startTimes[tmax]:endTimes[tmax]]), axis=1)
                imins__ = imins[:, startTimes[tmax-1]:endTimes[tmax]].astype(int) + ikilled_
                print('here2', startTimes[tmax-1], endTimes[tmax])
                
                #imins[:, startTimes[tmax]:endTimes[tmax]]
                ax_.set_title('max ava at t={}:{}; -1 {}:{}'.format(startTimes[tmax], endTimes[tmax], startTimes[tmax-1], endTimes[tmax-1]))
        else:
            ax_.set_title('no ava')
            imins__ = imins
        
        #raise ValueError()
        
        fig.delaxes(ax[rowID, 0]) #(ax[1, 2])
        insetDel += 1
        
        ax_ = fig.add_subplot(nrow, ncol, ncol*rowID + 1) #(236)
        h_, w_ = imins__.shape
        print(imins__.shape)
        idMin = np.min(imins__) #0.
        idMax = 4 #np.max(imins__) #1.
        heatmapI = plt.imshow(imins__, cmap='gist_heat_r', #'binary', 
                              clim=(idMin, idMax))
        plt.colorbar(heatmapI)
        
        ax_.set_aspect(w_/h_)
        ax_.set_title('ilow Evo(delta={})'.format(np.round(delta, 4)))    
        
        # % Get sizes distribution: is power law?
        colID = 2 #0 # start col
        data = sizes # discrete distr
        dataType = 'discrete' #'continuous' #'discrete'
        name = 'size'
        
        #print(dataType)
        if len(startTimes)> 0:
            insetDict = getCriticality(fig, ax, rowID, colID, ncol, data, dataType, name, 
                                       insetDict, qtmin = qtmin, insetDel=insetDel, loggedPdf=loggedPdf)
        
        rowID += 1
    
    fig.suptitle(figtitle)
    plt.tight_layout()
    
    insetCritical(fig, insetDict)
    
    plt.savefig(filename+'_Size_s{}_{}.png'.format(startSeed+seed_, q_u)) #'_bEvo{}.png'.format(seed_, k))
    plt.close()

################################### PLOTTING ##################################
#numSeeds, episodeLen, risk, seedJ, beta, tMid, paramDict, 
def general(nAgents, J, numSeeds, startSeed, episodeLen, avgRTraj_, mTraj, rTraj,
            cycleInset=None, M=None, figtitle=None, filename=None):
    # record spins
    nrow = 3 #numSeeds
    ncol = 1 + numSeeds #+ len(printIters)#(episodeLen//1000+1)
    
    fig, ax = plt.subplots(nrow, ncol, figsize=(5*ncol, 3*nrow))
    
    #for seed in range(numSeeds):
    ax[0, 0].plot(avgRTraj_) #, label='{}: M={}'.format(seed, np.round(world.avgIntention, 2)))
    
    #temps = 1/np.array([agent.beta for agent in world.agents])
    #countLowHigh += ['lowT-count:{}, highT-count:{}'.format(sum(temps<=world.agents[0].tempMid), sum(temps > world.agents[0].tempMid))]
    
    ax[0, 0].set_ylim([-.3, 2. + .1])
    ax[0, 0].set_xlabel('iteration')
    ax[0, 0].set_ylabel('Avg Reward (R_N)')
    #ax[0, 0].set_title('Final M (mean, std)={}'.format((np.mean(M), np.std(M))))
    
    # \% Log J (fixed across seeds)
    Jmax = np.max(np.abs([np.max(J), np.min(J)]))
    Jmin = -Jmax
    heatmapJ = ax[1, 0].imshow(J, cmap='bwr', interpolation='nearest', vmin=Jmin, vmax=Jmax)
    plt.colorbar(heatmapJ, ax = ax[1, 0])
    
    # \% Compute C(tau) or C(t, t+tau) ~ final 1000 iters
    isStationary = True
    numLags = 30 #7
    steadyIters = min(1000, episodeLen)
    
    acf, acvf, m = computeAcf_(mTraj, numLags=numLags, steadyIters=steadyIters)
    
    ax[2, 0].plot(acvf)
    ax[2, 0].set_ylabel('C(lag)')
    ax[2, 0].set_xlabel('lag')
    ax[2, 0].set_ylim([-1-.1, 1+.1])
    ax[2, 0].set_xlim([0, numLags*1.5])
    
    acvf_min = np.round(np.min(acvf), 4)
    acvf_max = np.round(np.max(acvf), 4)
    ax[2, 0].set_title('SKphase: q={} | Ctau min={}, max={}'.format(np.round(acvf[0], 4), acvf_min, acvf_max))
    
    # \% Log Reward & M evolutions of all agents
    steadyIters = min(100, episodeLen)
    
    for seed in range(numSeeds):
        for i in range(nAgents):
            # ax[3,0]
            ax[0, 1+seed].plot(mTraj[i, -steadyIters:, seed])
            # ax[4,0]
            ax[1, 1+seed].plot(rTraj[i, -steadyIters:, seed])
            
        ax[0, 1+seed].set_xlabel('iteration')
        ax[0, 1+seed].set_ylabel('intentions m_i (seed={})'.format(seed))
        ax[0, 1+seed].set_ylim([-1-.1, 1+.1])
        ax[0, 1+seed].set_xlim([0, steadyIters*1.5])
        
        ax[1, 1+seed].set_xlabel('iteration')
        #ax[4,0].set_ylabel('BS Qs Q_i (seed={})'.format(seed))
        ax[1, 1+seed].set_ylabel('rewards r_i (seed={})'.format(seed))
        ax[1, 1+seed].set_xlim([0, steadyIters*1.5])
    
    ax[0, 1+0].set_title('Spins phase by m_i; m-aggr (i,seed)={}'.format(np.round(m, 4)))

    fig.suptitle(figtitle)
    plt.tight_layout()
    
    if cycleInset is not None:
        # \% Visualize Cycle (last 20 iterations)
        subpos = [0.6, 0.5, 0.4, 0.4]
        subaxId = 0
        inset_ax = add_subplot_axes(fig.get_axes()[subaxId], subpos)
        
        for seed in range(numSeeds):
            inset_ax.plot(cycleInset[seed, :]) # enlarge last avgR -> detect cycle
        inset_ax.patch.set_alpha(0.5)  
        
        # set title ax[0, 0]: min cycleInset, max 
        avgR_min = np.round(np.min(cycleInset.flatten()), 4)
        avgR_max = np.round(np.max(cycleInset.flatten()), 4)
        avgR_mean = np.round(np.average(cycleInset.flatten()), 4)
        #avgR_med = np.round(np.median(cycleInset.flatten()), 4)
        
    else:
        pastIterNum = min(avgRTraj_.shape[0], 100)
        avgR_min = np.round(np.min(avgRTraj_[-pastIterNum:, :]), 4)
        avgR_max = np.round(np.max(avgRTraj_[-pastIterNum:, :]), 4)
        avgR_mean = np.round(np.average(avgRTraj_[-pastIterNum:, :]), 4)
        
    ax[0, 0].set_title('avgR mean:{} | min:{}, max:{}'.format(avgR_mean, avgR_min, avgR_max))
    
    if M is not None:
        # \% Plot avgIntention M
        subpos = [0.7, 0.1, 0.3, 0.3]
        subaxId = 0
        inset_ax = add_subplot_axes(fig.get_axes()[subaxId], subpos)
        inset_ax.hist(M)
        inset_ax.set_xlim([-1.-.1, 1.+.1])
        inset_ax.set_xlabel('M')
        inset_ax.patch.set_alpha(0.5)
    
    # \% Plot acf
    subpos = [0.75, 0.6, 0.25, 0.3]
    subaxId = ncol*2        
    inset_ax = add_subplot_axes(fig.get_axes()[subaxId], subpos)
    inset_ax.plot(acvf)
    inset_ax.set_xticklabels([])
    #inset_ax.set_title('acvf')
    inset_ax.patch.set_alpha(0.5)
    
    subpos = [0.75, 0.2, 0.25, 0.3]
    inset_ax = add_subplot_axes(fig.get_axes()[subaxId], subpos)
    inset_ax.plot(acf)
    #inset_ax.set_title('acf')
    inset_ax.patch.set_alpha(0.5)
    
    for seed in range(numSeeds):
        # \% Plot m_i histogram (detect phase m=0 if hist is unimodal at 0)
        subpos = [0.75, 0., 0.25, 1.]
        subaxId = ncol*0 + seed + 1 # ncol*3 # 
        inset_ax = add_subplot_axes(fig.get_axes()[subaxId], subpos)
        data = mTraj[:, -100:, seed].flatten()
        inset_ax.hist(data, bins=50, orientation='horizontal')
        inset_ax.set_xticklabels([])
        inset_ax.set_yticklabels([])
        
        subaxId = ncol*1 + seed + 1 #ncol*4        
        inset_ax = add_subplot_axes(fig.get_axes()[subaxId], subpos)
        data = rTraj[:, -100:, seed].flatten()
        inset_ax.hist(data, bins=50, orientation='horizontal')
        ##### !! PLOT QUANTILE LINES ON HISTOGRAM
        inset_ax.set_xticklabels([])
        inset_ax.set_yticklabels([])
    
    plt.savefig(filename+'.png')
    plt.close()

######################### BS Temp, Q Trajectories ########################
# Selection rule
# Period (k=15)
# Fitness (MA, E, lrbs=.5, .1)
# Killed population (n=1, 5, 20)
# Check Source of Fluctuation @beta-Evo L2
#bTraj = bTraj_
#qTraj = qTraj_
#iMinTraj = iMinTraj_
#episodeLen

def visualizePT(world, nAgents, truepId, adapt_p, adaptStr, filename):
    
    nrow = 1
    ncol = 1
    fig, ax_ = plt.subplots(nrow, ncol, figsize=(5*1.5*ncol, 3*1.5*nrow))
    
    X = np.linspace(-3, 3, 100) #(-100, 100, 100) #(-3, 3, 100)
    ax = ax_
    
    ax.plot(X, X, color='black')
    ax.plot(X, np.zeros(len(X)), color='grey', linestyle='--')
    for id_ in np.linspace(0, nAgents-1, 10).astype(int):
        agent = world.agents[id_]
        
        #agent.a1 = 2.5
        #agent.lmbd1 = .525
        '''
        agent.rScaler = 1.
        #agent.lmbd2 = .88
        agent.lmbd1 = agent.lmbd2
        #agent.a2 = 2.25
        agent.a1 = 1.
        '''
        
        legend_ = ''
        for pId in truepId:
            legend_ += adaptStr[pId] + ':{}'.format(np.round(agent.evo_p[pId], 2))
            if pId != truepId[-1]:
                legend_ += ', '
        
        agent.visualPT(X, ax, legend_)
    
    #lgd = ax.legend(loc=9, bbox_to_anchor=(0.5,0))
    lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #(ncol=2) #, loc='outside lower right')
    plt.ylim((-5, 5))
    
    #REPLOT AT TEvo (last row, last col)
    figtitle = ''
    #notpId = 
    for pId in range(len(adapt_p)):
        if not adapt_p[pId] and pId < len(adapt_p)-1:
            figtitle += adaptStr[pId] + ':{}, '.format(np.round(agent.evo_p[pId], 2))
            
        elif not adapt_p[pId] and pId == len(adapt_p) -1:
            figtitle += adaptStr[pId] + ':{}'.format(np.round(agent.evo_p[pId], 2))
            
    #plt.title(figtitle)
    ax.set_title(figtitle)
    #raise ValueError()
    #continue
    
    fig.savefig(filename + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()
    
    toScatter = world.trueEvo_ps#[:, 0]
    #df = pd.DataFrame(toScatter, columns = ['a'])
    df = pd.DataFrame(toScatter, columns = np.array(adaptStr)[adapt_p])
    axl = pd.plotting.scatter_matrix(df, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')
    
    plt.suptitle(figtitle)
    plt.savefig(filename + '_.png')
    plt.close()

def get_fmin(nAgents, episodeLen, seed_, qTraj, iMinTraj):
    
    qs_ = np.zeros((nAgents, episodeLen))
    for i in  range(nAgents):
        qs_[i, :] = qTraj[i, :, seed_]
        
    imins_ = iMinTraj[:, :, seed_]
    fminsBf_ = np.multiply(imins_[:, 1:], qs_[:, :-1])

    # works when numOfi = 1
    numOfis_ = np.sum(imins_[:, 1:], axis=0)
    fminsBf_ = np.sum(fminsBf_, axis=0) / numOfis_
    fmin = np.zeros(episodeLen)
    fmin[:-1] = fminsBf_
    fmin[-1] = qs_[:, -1].min()
    
    return fmin

def Tevolution(nAgents, J, numSeeds, startSeed, episodeLen, tMid, 
               bTraj, qTraj, iMinTraj, rTraj,
               figtitle, filename, steadyIters=5000, fcIters_=10000,
               tRange = (0., 2.), tOptRange = (.1, .2), fcRange = (.05, .95), 
               deltaRange = None, noAva = False, nrandom = False, numOfi = 1,
               adaptStr = ['temp'], truePid = [1]): #(.03, .005)):
    
    #steadyIters = 5000  #5000 #episodeLen - min(500, episodeLen)
    qtmin = 0.
    #fcIters_ = 10000 #len(fmin)
    
    print('epLen:', episodeLen)
    print('steadyIters, qtmin:', steadyIters, qtmin)
    print('f_cIters (to compute median for f_c):', fcIters_)
    
    steadyIters = min(steadyIters, episodeLen-1)
    fcIters_ = min(fcIters_, episodeLen-1) 
    
    if len(np.array(tRange).shape) == 1:
        tRange = np.expand_dims(np.array(tRange), axis=0)
    
    tOptMin, tOptMax = tOptRange

    for seed_ in range(numSeeds):
        print(len(bTraj.shape))
        if len(bTraj.shape) == 4:
            numTruepId = bTraj.shape[3]
            temps = np.zeros((nAgents, episodeLen, numTruepId))
        else:
            temps = np.zeros((nAgents, episodeLen))
            numTruepId = 1
            
        qs_ = np.zeros((nAgents, episodeLen))
        rs_ = np.zeros((nAgents, episodeLen))
        for i in  range(nAgents):
            #temps_[i, :] = 1/bTraj[i, :, seed_]
            temps[i, :, :] = 1/bTraj[i, :, seed_]
            qs_[i, :] = qTraj[i, :, seed_]
            rs_[i, :] = rTraj[i, :, seed_]
        
        imins_ = iMinTraj[:, :, seed_]
        
        nrow = 3 + numTruepId
        ncol = 1 + 1 + 2 #+ 4
        
        fig, ax = plt.subplots(nrow, ncol, figsize = (5*ncol, 4*nrow)) #(5*3, 4*2))
        
        rowID = 0
        
        for pId in range(numTruepId):
            temps_ = temps[:, :, pId]
            
            tMin, tMax = tRange[truePid[pId], :]
            #print(pId, tMin, tMax, temps_.shape)
            # % Plot temp Evo
            #rowID = 0
            fig.delaxes(ax[rowID, 0]) #[1, 0])
            ax_ = fig.add_subplot(nrow, ncol, ncol*rowID + 1) #(234) 
            h_, w_ = temps_.shape # (N, epLen)
            Tmin = tMin 
            Tmax = tMax 
            
            if adaptStr[truePid[pId]] == 'a':
                scaler_ = Tmin + .7*(Tmax - Tmin) # assert mid > vmin
            else:
                scaler_ = Tmin + .15*(Tmax - Tmin) #.3
                
            norm=SqueezedNorm(vmin=Tmin, vmax=Tmax, mid=scaler_*Tmax, 
                              s1=2, s2=2)
            heatmapT = plt.imshow(temps_, cmap='inferno_r', #interpolation='nearest', 
                                  norm = norm, aspect = 'auto')
                                  #vmin=Tmin, vmax=Tmax)
            
            plt.colorbar(heatmapT) #, ax = ax[1, 0]) #, ax = ax[1, 0])
            ax_.set_aspect(w_/h_)
            
            ax_.set_title('{} Evo'.format(adaptStr[truePid[pId]])) #'temp Evo')#countLowHigh[seed_])
            
            ############################ iMin, FITNESS BEFORE START HERE! ###############################
            
            windowSize = 20 #500 #steadyIters #20
            K = episodeLen // windowSize #(windowSize // 2) #10
            rem = episodeLen % windowSize
            
            numk = min(50, K)
            step = K // numk
            kList = list(range(1, K, step)) + [K]
            
            kID = 0
            for k in kList: #kList: #np.linspace(1, K, numk): #[1, 100, 200, 300, 400, 500]:
                
                kID = 1
                if k == K:
                    color_ = 'blue'
                    transp_ = 1
                    startWindow = (k-1)*windowSize+rem
                    endWindow = k*windowSize+rem
                elif k == 0:
                    color_ = 'grey'
                    transp_ = 1
                    startWindow = 0
                    endWindow = windowSize
                else:
                    color_ = 'tab:gray'
                    transp_ = .1
                    startWindow = (k-1)*windowSize+rem
                    endWindow = k*windowSize+rem
                
                # % Plot temp Evo
                data = temps_[:, startWindow:endWindow].flatten() 
                #bTraj[:, startWindow:endWindow, seed_].flatten()
                #data = 1/data
                
                if max(data) - min(data) < 1e-2:
                    # ax[0, 2] 
                    x = ax[rowID, kID].hist(data, bins=1, orientation='horizontal', color=color_, #'tab:blue',
                                        histtype = 'step', alpha = transp_)
                    # color='black', alpha=1/(episodeLen//15), histtype='step'
                    ax[rowID, kID].plot([x[1][0] for _ in range(int(x[0][0]))], color = color_, #'tab:blue',
                                    alpha =transp_)
                else:
                    x = ax[rowID, kID].hist(data, bins=30, orientation='horizontal', color = color_,
                                        histtype = 'step', alpha = transp_)
                
                ax[rowID, kID].set_title('{} Dist Evo (window size={})'.format(adaptStr[truePid[pId]], windowSize))
                
                
                if pId == 0: # only one fitness for all temps
                    # % Plot Q (fitness) Evo
                    data = qTraj[:, startWindow:endWindow, seed_].flatten()
                    
                    if max(data) - min(data) < 1e-2:
                        # ax[0, 2] 
                        x = ax[numTruepId+1, kID].hist(data, bins=1, orientation='horizontal', color= color_, #'tab:blue',
                                            histtype = 'step', alpha = transp_)
                        ax[numTruepId+1, kID].plot([x[1][0] for _ in range(int(x[0][0]))], color = color_, #'tab:blue',
                                        alpha = transp_)
                    else:
                        x = ax[numTruepId+1, kID].hist(data, bins=30, orientation='horizontal', color = color_,
                                            histtype = 'step', alpha = transp_)
                    
                    ax[numTruepId+1, kID].set_title('fitness/q Dist Evo (window size={})'.format(windowSize))
                    
                    # % Plot R (fitness) Evo
                    data = rTraj[:, startWindow:endWindow, seed_].flatten()
                    
                    if max(data) - min(data) < 1e-2:
                        # ax[0, 2] 
                        x = ax[numTruepId, kID].hist(data, bins=1, orientation='horizontal', color= color_, #'tab:blue',
                                            histtype = 'step', alpha = transp_)
                        ax[numTruepId, kID].plot([x[1][0] for _ in range(int(x[0][0]))], color = color_, #'tab:blue',
                                        alpha = transp_)
                    else:
                        x = ax[numTruepId, kID].hist(data, bins=30, orientation='horizontal', color = color_,
                                            histtype = 'step', alpha = transp_)
                    
                    ax[numTruepId, kID].set_title('fitness/q Dist Evo (window size={})'.format(windowSize))
            
            _, hist_max = ax[0, kID].get_xlim()
            ax[rowID, kID].plot([tOptMin for _ in range(int(hist_max)+1)], linestyle='--', color = 'r', label='t={}'.format(np.round(tOptMin, 4)))
            ax[rowID, kID].plot([tOptMax for _ in range(int(hist_max)+1)], linestyle='--', color = 'g', label='t={}'.format(np.round(tOptMax, 4)))
            ax[rowID, kID].legend(loc='lower right')
                
            # kID+1
            # % Temp's
            Tavg = np.average(temps_, axis=0)        
            ax[rowID, kID+1].plot(Tavg, label = 'mean', color = 'black')
            
            #Tmode = st.mode(temps_, axis=0, keepdims=True).mode.flatten()
            Tmedian = np.median(temps_, axis=0)
            Tmin = np.min(temps_, axis=0)
            Tq1 = np.quantile(temps_, .25, axis=0)
            Tq3 = np.quantile(temps_, .75, axis=0)
            Tmax = np.max(temps_, axis=0)
            
            #5-summary
            ax[rowID, kID+1].plot(Tmin, label = 'min:'+str(np.round(Tmin[-1], 4)))
            ax[rowID, kID+1].plot(Tq1, label = 'q1:'+str(np.round(Tq1[-1], 4)))
            ax[rowID, kID+1].plot(Tmedian, label = 'median:'+str(np.round(Tmedian[-1], 4)))
            ax[rowID, kID+1].plot(Tq3, label = 'q3:'+str(np.round(Tq3[-1], 4)))
            ax[rowID, kID+1].plot(Tmax, label = 'max:'+str(np.round(Tmax[-1], 4)))
            #ax[0, kID+1].plot(Tmode)
            ax[rowID, kID+1].set_title('temp mean-5summary')
            ax[rowID, kID+1].legend(loc='lower right')
            
            Tstdev = np.std(temps_, axis=0) 
            ax[rowID, kID+2].plot(Tstdev, label='stdev')
            ax[rowID, kID+2].set_title('temp stdev')
            ax[rowID, kID+2].legend(loc='lower right')
            
            rowID += 1
            
        ################################# FITNESS, iMin moved here after ################################
        # % Fitness(Q)'s
        # Add key observables for B-S
        # Minimum fitness: f_min(t) # increasing in t
        
        # % Plot R Evo
        fig.delaxes(ax[rowID, 0]) #(ax[1, 2])
        ax_ = fig.add_subplot(nrow, ncol, ncol*rowID + 1) #(236)
        #print('getaxes try:', fig.get_axes())
        h_, w_ = rs_.shape
        Qmin = np.min(rs_)
        Qmax = np.max(rs_)
        #norm_ = MidpointNorm(vmin=Qmin, vmax=Qmax, midpoint=0.) #mid=0., s1=0, s2=0)
        norm_ = matplotlib.colors.TwoSlopeNorm(vmin=Qmin, vcenter=0, vmax = Qmax)
        heatmapR = plt.imshow(rs_, cmap='RdBu', clim=(Qmin, Qmax), #interpolation='nearest', 
                              norm = norm_)#, aspect = 'auto')
                              #vmin=Qmin, vmax=Qmax)
        plt.colorbar(heatmapR) #, ax = ax[1, 0]) #, ax = ax[1, 0])
        ax_.set_aspect(w_/h_)
        ax_.set_title('r Evo')
        
        # % Plot Q (fitness) Evo
        rowID += 1
        fig.delaxes(ax[rowID, 0]) #(ax[1, 2])
        ax_ = fig.add_subplot(nrow, ncol, ncol*rowID + 1) #(236)
        #print('getaxes try:', fig.get_axes())
        h_, w_ = qs_.shape
        Qmin = np.min(qs_)
        Qmax = np.max(qs_)
        #norm_ = MidpointNorm(vmin=Qmin, vmax=Qmax, midpoint=0.) #mid=0., s1=0, s2=0)
        norm_ = matplotlib.colors.TwoSlopeNorm(vmin=Qmin, vcenter=0, vmax = Qmax)
        heatmapQ = plt.imshow(qs_, cmap='RdBu', clim=(Qmin, Qmax), #interpolation='nearest', 
                              norm = norm_)#, aspect = 'auto')
                              #vmin=Qmin, vmax=Qmax)
        plt.colorbar(heatmapQ) #, ax = ax[1, 0]) #, ax = ax[1, 0])
        ax_.set_aspect(w_/h_)
        ax_.set_title('q Evo')
        
        # % Plot iMin (activity) Evo
        rowID += 1
        fig.delaxes(ax[rowID, 0]) #(ax[1, 2])
        ax_ = fig.add_subplot(nrow, ncol, ncol*rowID + 1) #(236)
        h_, w_ = imins_.shape
        idMin = np.min(imins_) #0.
        idMax = np.max(imins_) #1.
        heatmapI = plt.imshow(imins_, cmap='binary', clim=(idMin, idMax))
        plt.colorbar(heatmapI)
        ax_.set_aspect(w_/h_)
        ax_.set_title('imin Evo')
        
        # fitness-temp Dirac checks
        numOfis_ = np.sum(imins_[:, 1:], axis=0)
        #print(numOfis_)
        isAvged = np.max(numOfis_)
        
        if nrandom or numOfi > 1 or isAvged > 1:
            imins_temp = np.argmin(qs_[:, :-1], axis=0)
            
            imins__ = np.zeros(imins_[:, 1:].shape)
            for count, i in enumerate(imins_temp):
                #print(count, i)
                imins__[i, count] = 1
                
            fminsBf_ = np.multiply(imins__, qs_[:, :-1]) # 0:T-1
            fminsAft_ = np.multiply(imins__, qs_[:, 1:]) # 1:T
        else:
            fminsBf_ = np.multiply(imins_[:, 1:], qs_[:, :-1])
            fminsAft_ = np.multiply(imins_[:, 1:], qs_[:, 1:])

        # works when numOfi = 1
        fminsBf_ = np.sum(fminsBf_, axis=0) / numOfis_#np.min(fminsBf_, axis=0) #
        fminsAft_ = np.sum(fminsAft_, axis=0) / numOfis_ #np.min(fminsAft_, axis=0) #
        
        ax[rowID, 1].scatter(range(len(fminsBf_)), fminsBf_, color='red', s=5, 
                             alpha=.1, label='bf')
        ax[rowID, 1].scatter(range(len(fminsAft_)), fminsAft_, color='blue', s=5, 
                             alpha=.1, label='aft')
        ax[rowID, 1].legend(loc='lower right')
        if isAvged > 1:
            ax[rowID, 1].set_title('MIN({}) f[imin] bf-aft Update'.format(isAvged))
        else:
            ax[rowID, 1].set_title('f[imin] bf-aft Update')
        
        print('steady', steadyIters)
        print('len fminsBf_', len(fminsBf_[-steadyIters:]))
        print('epLen', episodeLen)
        
        ax[rowID, 2].scatter(range(len(fminsBf_)-steadyIters, len(fminsBf_)), 
                             fminsBf_[-steadyIters:], color='red', s=5, alpha=.2, label='bf')
        ax[rowID, 2].scatter(range(len(fminsBf_)-steadyIters, len(fminsBf_)), 
                             fminsAft_[-steadyIters:], color='blue', s=5, alpha=.2, label='aft')
        ax[rowID, 2].legend(loc='lower right')
        ax[rowID, 2].set_title('f[imin] bf-aft (steadyIters={})'.format(steadyIters))
        
        '''# Check fitness[lmbd1, lmbd2, ...]
        tminsBf_ = np.multiply(imins_[:, 1:], temps_[:, :-1])
        tminsAft_ = np.multiply(imins_[:, 1:], temps_[:, 1:])
        tminsBf_ = np.sum(tminsBf_, axis=0) / numOfis_
        tminsAft_ = np.sum(tminsAft_, axis=0) / numOfis_
        
        ax[rowID, 3].scatter(range(len(tminsBf_)-steadyIters, len(tminsBf_)), tminsBf_[-steadyIters:], color='red', s=5, alpha=.2, label='bf')
        ax[rowID, 3].scatter(range(len(tminsBf_)-steadyIters, len(tminsBf_)), tminsAft_[-steadyIters:], color='blue', s=5, alpha=.2, label='aft')
        ax[rowID, 3].plot(range(len(tminsBf_)-steadyIters, len(tminsBf_)), tminsAft_[-steadyIters:], color='grey', alpha=.5)
        ax[rowID, 3].legend()
        ax[rowID, 3].set_title('Temp[imin] bf-aft (steadyIters={})'.format(steadyIters))
        '''
        
        # % Plot iMin (activity) Critical
        episodeLen_ = episodeLen #10000
        startSteady = episodeLen_ - min(steadyIters, episodeLen_)
        assert startSteady > 0 # avoid i,j < 0
        
        '''distances = np.zeros(len(range(startSteady, episodeLen_-1)))
        agentIDs = np.arange(0, nAgents, dtype=int)
        for t in range(startSteady, episodeLen_-1):
            is_ = agentIDs[iMinTraj[:, t, seed_].astype('bool')] #[0] # cz numofi=1
            js_ = agentIDs[iMinTraj[:, t+1, seed_].astype('bool')] #[0]
            
            #print(is_, js_)
            assert len(is_) > 0 and len(js_) > 0

            distances[t-startSteady] = np.array([np.abs(J[i, j]) for i in is_ for j in js_]).mean()
            #np.abs(J[i, j]) # /maxJ; := np.max(J)
        
        
        data = distances
        name = 'abs J(i_t,i_t+1)'
        dataType = 'continuous'
        '''
        
        print('Bf fitness..') #fmin = fminsBf_ 
        fmin = np.zeros(episodeLen)
        fmin[:-1] = fminsBf_
        fmin[-1] = qs_[:, -1].min()
        
        fgap = np.maximum.accumulate(fmin) # fdelta(delta=0)
        
        #print(np.unique(fgap))
        #raise ValueError()
        
        fmin_ = fmin[-fcIters_:]
        qc, qg = fcRange
        f_c = np.quantile(fmin_, qc) #f_c = np.median(fmin[-fcIters_:])
        f_g = np.quantile(fmin_, qg) #f_g = np.max(fgap[-steadyIters:])
        
        # ax[0]
        ax[numTruepId+1, kID+1].scatter(range(len(fgap)), fgap, 
                    color='blue', alpha=.1, s=5, label='fgap={}'.format(np.round(fgap[-1], 4)))
        ax[numTruepId+1, kID+1].scatter(range(len(fmin)), fmin, 
                    color='grey', alpha=.1, s=5, label='fmin')
        # scatter(range.., fds, ..)
        ax[numTruepId+1, kID+1].axhline(y=f_c, color='g', linestyle='--', label='f_c={}'.format(np.round(f_c, 4)))
        ax[numTruepId+1, kID+1].axhline(y=f_g, color='tab:orange', linestyle='--', label='f_g={}'.format(np.round(f_g, 4)))
        ax[numTruepId+1, kID+1].legend(loc='lower right')
        ax[numTruepId+1, kID+1].set_title('fmin-Gap')
        
        # ax[1]
        ax[numTruepId+1, kID+2].scatter(range(len(fgap)-steadyIters, len(fgap)), fgap[-steadyIters:], 
                    color='blue', alpha=.1, s=5, label='fgap={}'.format(np.round(fgap[-1], 4)))
        ax[numTruepId+1, kID+2].scatter(range(len(fmin)-steadyIters, len(fmin)), fmin[-steadyIters:], 
                    color='grey', alpha=.1, s=5, label='fmin')
        ax[numTruepId+1, kID+2].axhline(y=f_c, color='g', linestyle='--', label='f_c={}'.format(np.round(f_c, 4)))
        ax[numTruepId+1, kID+2].axhline(y=f_g, color='tab:orange', linestyle='--', label='f_g={}'.format(np.round(f_g, 4)))
        maxLine = np.max(fgap)
        
        ax[numTruepId+1, kID+2].set_ylim([f_c - (maxLine-f_c) - .01, maxLine + .01])
        ax[numTruepId+1, kID+2].legend(loc='lower right')
        ax[numTruepId+1, kID+2].set_title('fmin-Gap (steadyIters={})'.format(steadyIters))
        
        fig.suptitle(figtitle)
        plt.tight_layout()
        
        plt.savefig(filename+'_s{}.png'.format(startSeed+seed_)) #'_bEvo{}.png'.format(seed_, k))
        plt.close()
        
        if not noAva:
            #d_u, d_l = deltaRange
            if deltaRange is not None:
                d_u, d_l = deltaRange
            else:
                d_u, d_l = None, None
            
            q_l, q_u = .05, .6 #.05, .5 # .15, .6
            numDeltas = 12
            AvalancheSize(nAgents, seed_, startSeed, episodeLen, 
                          qTraj, figtitle, filename, 
                          steadyIters=steadyIters, qtmin=qtmin, 
                          #midLine_c=f_c, midLine_g=f_g,
                          ava_p=(fmin, d_u, d_l, q_u, q_l, numDeltas))
        else:
            return fmin
            
        
