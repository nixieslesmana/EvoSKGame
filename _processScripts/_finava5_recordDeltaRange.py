# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:36:59 2024

@author: nixie
"""

import pandas as pd
import numpy as np
#import plotting
#import math
#import matplotlib.pyplot as plt
import csv
import powerlaw

NRange = [256] # or [512] -> break down MLE_c, MLE_d, _init_params, _param_range
JRange = [0, 1, 2, 3, 4]
sRange = [5, 6, 7, 8, 9]

truncated=True

fit_discrete = True # False # constrained or not
if fit_discrete:
    str_fitdiscr = 'dTflip' #'d_2389' # i=22, delta=dArr[i]
else:
    str_fitdiscr = 'c'

usebound = True # want to cover lognorm range. Or True, but only set d_min (near 0)
dbound = {(128, .7): (-.076, -.004), #(-.076, -.023),#
          (256, .3): (-.035, -.003), #(-.065, -.035), #(-.095, -.065),
          (256, .5): (-.035, -.003), #(-.065, -.035), #(-.095, -.065),
          (256, .7): (-.035, -.003), #(-.065, -.035), #(-.095, -.065),
          (256, .75): (-.035, -.003), #(-.065, -.035),#(-.095, -.065),
          (256, .77): (-.035, -.003), #(-.065, -.035),#(-.095, -.065), 
          (256, .8): (-.035, -.003), #(-.065, -.035),#(-.095, -.065),
          (256, .9): (-.035, -.003), #(-.065, -.035), #(-.095, -.065),
          (512, .7): (-.038, -.003), #(-.03, -.012), #
          (1024, .3): (-.05, -.027),#(-.027, -.008), #(-.008, -.003),
          (1024, .5): (-.05, -.027), #(-.027, -.008), #(-.008, -.003),
          (1024, .7): (-.008, -.003), #(-.05, -.027), #(-.027, -.008), #'sub, small size dyn': (-.05, -.025)
          (1024, .75): (-.008, -.003),#(-.0161, -.0149), #(-.05, -.027), #(-.027, -.008), #(-.008, -.003),
          (1024, .77): (-.0333, -.0327), #(-.008, -.003), #(-.05, -.027), #(-.027, -.008),
          (1024, .8): (-.1, -.05), #(-.027, -.008), #(-.05, -.027), #(-.2, -.1), 
          (1024, .9): (-.05, -.027) #(-.008, -.003)
          }

#(-.15, -.1), #(-.1, -.05), #(-.05, -.027) #(-.027, -.008), #(-.008, -.003),


dataProp = False #True # default: False (record tpl fitting)
'''
dbound = {(1024, .3): (-.008, -.003),
          (1024, .5): (-.008, -.003),
          (1024, .7): (-.008, -.003),
          (1024, .75): (-.008, -.003), #(-.027, -.008), #(-.008, -.003),
          (1024, .77): (-.008, -.003),
          (1024, .8): (-.027, -.008),
          (1024, .9): (-.05, -.027)
          }
'''

numlog = 20 # closer to cutoff estimate from sizes directly?
# To expect finite-size-scaling: (cutoff*N**b, delta*N**a)

xmin_def=2

for nAgents in NRange:    
    for alpha in [.9]:  #.3,.5,.7,.75,.77,.8,.9
        print('alpha:', alpha)
        if nAgents==128:
            t_str = 100
        elif nAgents==256:
            t_str = 150
        elif nAgents==512:
            t_str=200
        #elif alpha == .7:
        #    t_str=252
        else:
            t_str=300
            
        for xmin in [2]: #[20, 22, 24, 28, 30]:
            fit_method_name = 'MLE'
                
            np.random.seed(0)
            
            ### Delta binning
            deltaSampleNum = 50 # 20 
            maxSize = None #2000 (small d) #500
            print('maxSize:', maxSize)
            divisor = 10 #5
            
            # default size setting
            pmf = False
            loggedPdf = True
            
            foldername = 'compressed/ava_N{}a{}Tadapt'.format(nAgents, str(alpha)[2:]) #'dResults_{}/'.format(str_fitdiscr)
            #if alpha in [.75]:
            #    foldername += '_full/'
            #elif alpha in [.7]:
            #    foldername += '_300k/'
            #elif alpha in [.8, .9]:
            #    foldername += '_dLarge/'
            #elif alpha in [.3, .5]:
            #    foldername += '_win60k/'
            #else:
            foldername += '/'
                
            print('correct folder?', foldername)
            
            filename = foldername + 'N{}a{}_xmin{}_fit{}'.format(nAgents, alpha, xmin_def, fit_method_name)
            
            # Get ALL SAMPLED deltas
            slopeTarget = -1.5
            slopeLow = slopeTarget + .2 
            slopeHigh = slopeTarget - .2 
            slopeOut = -1.
            #if alpha == .7:
            #    slopeOut = -1#.1
            #else:
            #    slopeOut = -1.1
            dSeed = 0
            alldelta = pd.read_csv(filename+'.csv', header=None, index_col=[0, 1, 2, 3])
            alldelta = alldelta.loc[(slopeLow, slopeHigh, slopeOut, dSeed)].to_numpy(dtype=float)
            if len(alldelta.shape) > 1:
                alldelta = alldelta[0, :]
            
            idx = pd.IndexSlice
            
            ############################## DELTA SEARCH START #############################
            if usebound:
                d_l, d_u = dbound[(nAgents, alpha)]
            else:
                d_l, d_u = -alldelta[0], -alldelta[-1]
            
            slope_l = -np.inf
            slope_u = np.inf 
            
            tplAll = []
                
            dArr = alldelta[(alldelta >= -d_u)*(alldelta <= -d_l)] #dArr = plotting.get_dArray(fmin_, q_l=q_l, q_u=q_u)
            if len(dArr) == 0:
                break
            
            print('all d len', len(dArr))
            
            # Resample delta init - 10
            dArr_ = []
            dAll = np.linspace(dArr[0], dArr[-1], deltaSampleNum + 1) #min(deltaSampleNum, len(dArr))+1)
            for i in range(len(dAll)-1):
                dRange = dArr[(dArr <= dAll[i]) * (dArr > dAll[i+1])]
                if len(dRange) >= 1:
                    dArr_ += [np.random.choice(dRange, 1)[0]]
            
            dArr = np.unique(dArr_)
            dArr[::-1].sort()
            #dArr = np.array(dArr_)
            
            if len(dArr) == 0:
                continue
            
            print('dArr:', dArr, ', len:', len(dArr))
            #raise ValueError()
            
            d_pdfs = []
            
            for i in range(len(dArr)):
                delta = dArr[i]
                print('===', i, delta, '===')
                #if i <= 34:
                #    continue
                sizes = []
                
                for startSeedJ in JRange: 
                    for startSeed in sRange: 
                        if truncated:
                            filename_ =  filename+'_l{}h{}_{}_J{}s{}_{}k.csv'.format(-slopeLow, -slopeHigh, dSeed, startSeedJ, startSeed, t_str)                        
                        else:
                            filename_ =  filename+'_l{}h{}_{}_J{}s{}.csv'.format(-slopeLow, -slopeHigh, dSeed, startSeedJ, startSeed)

                        df = pd.read_csv(filename_, header=0)
                                
                        try:
                            sizeJs = df.loc[idx[:], idx[str(delta)]].fillna(0).to_numpy(dtype=int)
                        except:
                            sizeJs = df.loc[idx[:], idx[str(np.round(delta, 9))]].fillna(0).to_numpy(dtype=int)
                        sizeJs = sizeJs[sizeJs > 0]
                        
                        sizes += list(sizeJs)
                
                ################ NEW FITTING
                # Sizes < maxSize for all delta + Exponents
                sizes = np.array(sizes)
                data = sizes[:] #[sizes<maxSize]
                maxSize = None
                
                if len(np.unique(data)) < 5:
                    print('len data < 5')
                    continue
                
                if fit_method_name == 'MLE':
                    fit_method = 'Likelihood'
                
                if maxSize is not None:
                    data = data[data<=maxSize]
                else:
                    maxSize = max(data)
                    data = data[data<=maxSize]
                data_ = data[:]
                
                data = data[data>=xmin]
                #print('data max:', max(data))
                print('bf powerlaw fitting, size min, max', min(data), max(data))
                if dataProp:
                    # delta, min(data)
                    tplAll += [[-delta, min(data), max(data)]]
                    continue
                
                print('--')
                
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
                    binEdges = np.logspace(np.log10(xmin), np.log10(maxSize), numlog) #np.logspace(0, 4, 50)
                    Plog, _ = np.histogram(data, density=True, bins=binEdges)
                    if pmf:
                        widths = (binEdges[1:] - binEdges[:-1])
                        Plog /= 1/widths
                    
                    Xlog = (binEdges[:-1] + binEdges[1:])/2
                    Xlog = Xlog.astype('int')
                else:
                    Xlog, Plog = X, P
                
                Xbool_ = Plog > 0
                
                ################################### start fitting #############################
                
                
                ### to record
                d_pdfs += [[delta] + ['size', '', ''] + list(np.log10(Xlog[Xbool_]))] #### ??? Use Xbool here???
                d_pdfs += [[delta] + ['P', '', ''] + list(np.log10(Plog[Xbool_]))]
                
                results = powerlaw.Fit(data, discrete=True, xmin=xmin, fit_method=fit_method)
                
                names = ['pl', 'tpl', 'ln']
                count = 0
                tplArrP = []
                
                for fittedDistr in [results.power_law, results.truncated_power_law, results.lognormal]:
                    if count==0:
                        slope=fittedDistr.alpha
                        cutoff=0.
                    else:
                        slope=fittedDistr.parameter1
                        cutoff=fittedDistr.parameter2
                    pdf_d = fittedDistr.pdf(Xlog)
                    
                    if count in [0, 1]:
                        d_pdfs += [[delta] + [names[count]] + [-slope, cutoff] + list(np.log10(pdf_d[Xbool_]))]
                    else:
                        d_pdfs += [[delta] + [names[count]] + [slope, cutoff] + list(np.log10(pdf_d[Xbool_]))]
                        
                    KSscore = fittedDistr.KS(data_[data_>=xmin])
                    MLEscore = fittedDistr.loglikelihoods(data_[data_>=xmin]).mean()
                    #tplR, tplp = results.distribution_compare('truncated_power_law', 'exponential', normalized_ratio=True)
                    mse = np.average((np.log10(pdf_d[Xbool_]) - np.log10(Plog[Xbool_]))**2)
                    
                    if count == 2:
                        score1, score2 = results.distribution_compare('truncated_power_law', 'lognormal')
                        tplArrP += [slope, cutoff, MLEscore, KSscore, mse, score1, score2]
                        # actual cutoff = coeff_sq = -1/(2*sig^2)
                    elif count == 1:
                        tplArrP += [-slope, cutoff, MLEscore, KSscore, mse]
                    else:
                        tplArrP += [-slope, MLEscore, KSscore, mse]
                    
                    if count == 1:
                        dataPlogged = [np.log10(P[0]), np.log10(Plog[0])] 
                        
                    count += 1
                
                tplAll += [[-delta, len(data)] + dataPlogged + tplArrP]
                #raise ValueError()
            
            tplAll = np.array(tplAll)
            
            if dataProp:
                f = open(foldername + 'delta_aggrJs_sizeDat{}.csv'.format(nAgents), 'a', newline = '')
                writer = csv.writer(f)
                rownames = ['d', 'minSizeDat', 'maxSizeDat']
                for r in range(tplAll.shape[1]): #l in range(len(tplAll)):
                    to_append = [[nAgents, alpha, xmin, fit_method_name+str(usebound)[0], rownames[r]] + list(tplAll[:, r])]
                    writer.writerows(to_append)
                
                f.close()
            
            f = open(foldername + 'delta_aggrJs_new{}.csv'.format(nAgents), 'a', newline = '')
            writer = csv.writer(f)
            rownames = ['d', 'lenSizeDat', 'datLogP-o', 'datLogP-x', 
                        'pl-slope', 'pl-MLE', 'pl-KS', 'pl-mse{}'.format(numlog),
                        'tpl-slope', 'tpl-cutoff', 'tpl-MLE', 'tpl-KS', 'tpl-mse{}'.format(numlog),
                        'ln-mu', 'ln-sig', 'ln-MLE', 'ln-KS', 'ln-mse{}'.format(numlog), 'tplvsLn1', 'tplvsLn2']
            for r in range(tplAll.shape[1]): #l in range(len(tplAll)):
                to_append = [[nAgents, alpha, xmin, fit_method_name+str(usebound)[0], rownames[r]] + list(tplAll[:, r])]
                writer.writerows(to_append)
            
            f.close()
            
            f = open(foldername + 'delta_aggrJs_pdf{}.csv'.format(nAgents), 'a', newline = '')
            writer = csv.writer(f)
            for r in range(len(d_pdfs)):
                to_append = [[nAgents, alpha, xmin, fit_method_name+str(usebound)[0]] + d_pdfs[r]]
                writer.writerows(to_append)
                
            f.close()

