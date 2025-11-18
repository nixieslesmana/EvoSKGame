# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TVD

# read DF
tvd_df = pd.read_csv('compressed/tvd_med_1-25.csv', header = 0, index_col = [0, 1, 2]) #col_header = 3)
#tvd_df.head()

tvd_df.columns = range(1, tvd_df.shape[1]+1)
tvd_df *= 0.5 # total variation formula

aRange = np.array([[0.3, 0.5, 0.7],
                   [0.75, 0.8, 0.9]]).T
#tvd_df.index.levels[0] # [0.3, 0.5, 0.7, 0.75, 0.8, 0.9]

fig, axs = plt.subplots(aRange.shape[0], aRange.shape[1], sharex = True, sharey = True, 
                        figsize = (6*aRange.shape[1], 4*aRange.shape[0]))
for j in range(aRange.shape[1]): #range(len(aRange)):
    for i in range(aRange.shape[0]):
        #print(i, j)
        alpha = aRange[i, j]
        
        if alpha is None:
            continue
            
        #tvd_df.loc[alpha, :, :].head()
        #tvd_df.loc[alpha, :, :].shape
        
        tvd_summ = tvd_df.loc[alpha, :, :].T.describe().T
        
        ax = axs[i, j]
        
        x_names = tvd_summ.index
        x_ = [8.] + list(tvd_summ.loc['prtbX'].index) # 7. proxy for (pInit, 0.)
        colIDs = list(range(-5, 0, 1))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        for cId in colIDs: #5-summary stats col indexes
            col_name = tvd_summ.columns[cId]
            
            if col_name in ['min', 'max']:
                continue
            
            y_ = tvd_summ[col_name]
            ax.scatter(x_, y_, color=colors[cId+5], label = col_name)
            
            ax.hlines(y_[0], xmin=0, xmax=1, transform=ax.get_yaxis_transform(), color="0.8", linestyle='--')
                
            if col_name == '25%': #'min':
                ymin = y_[0]
            elif col_name == '75%': #'max':
                ymax = y_[0]
            
            ax.plot(x_[1:], y_[1:], color=colors[cId+5])
            
            if col_name == '75%':
                xBreakId = np.argmax(y_[1:] < 1. - 0.05) - 1 #1st true from above - 1
            elif col_name == '25%':
                xErgoId = -1*np.argmax(y_[1:][::-1] > 0. + .05) #1st true from below + 1
            
        #ax_ylim = list(ax.get_ylim())
        ax.vlines(0, ymin=0, ymax=1, transform=ax.get_xaxis_transform(), color = 'green', linestyle = '--',
                  alpha = 0.3, label = 'T_a (unperturbed)')
        ax.axhspan(ymin, ymax, facecolor = "0.8", alpha = 0.2, label = 'T_a init (random)')
        
        #x_min, x_max = list(ax.get_xlim())
        ax.axvspan(-6.2, x_[1:][xBreakId], facecolor = 'white', hatch = '..', 
                   edgecolor = 'maroon', alpha = 0.2, zorder = 1, label = 'Phase: Breaking')
        ax.axvspan(x_[1:][xErgoId], 8.2, facecolor = 'white', hatch = '//', 
                   edgecolor = 'darkgoldenrod', alpha = 0.3, zorder = 1, label = 'Phase: Ergodic')
        
        if alpha == .9:
            ax.legend()    
        
        ax.set_title('alpha:{}'.format(alpha))
        
plt.xlim((-6.2, 8.2))
plt.subplots_adjust(wspace=0.05, hspace=0.15)
                    
fig.text(0.5, 0.09, 'Perturbation Strength (T_a Rescale)' , ha = 'center')
fig.text(0.08, 0.5, 'Total Variation (Ergodicity Level)', va = 'center', rotation = 'vertical')



'''
fig.delaxes(axs[0, 1])
fig.delaxes(axs[3, 1])

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.7, .7), ncol=3)'''
#plt.tight_layout(rect=[0, 0, 1, .8]) # Adjust layout to make space for the legend

# x-axis: remove tick OR draw horizontal line to replace INIT

# R
R_df = pd.read_csv('compressed/R_avg_1-25.csv', header = 0, index_col = [0, 1, 2])
#R_df.head()

R_df.columns = range(1, R_df.shape[1]+1)



fig, axs = plt.subplots(2, 1, figsize = (6, 4*2)) #sharey = True, 

ax = axs[0]

R75_summ = R_df.loc[0.75, :, :].T.describe().T
x_ = list(R75_summ.loc['const', :].index) + [.9, 1.0]

for cId in colIDs:
    print(cId)
    col_name = R75_summ.columns[cId]
    y_ = R75_summ[col_name]
    
    #ax.scatter(x_[:-2], y_[:-2], label=col_name, color='black')

    if col_name == '50%':
        ax.plot(x_[:-2], y_[:-2], color='black')
    elif col_name in ['25%', '75%']:
        ax.scatter(x_[:-2], y_[:-2], label=col_name, color='black', marker = '_', s=50)
        ax.scatter(x_[-1], y_[-1], color='green', marker = '_', s=50)
        ax.scatter(x_[-2], y_[-2], color='purple', marker = '_', s=50)
        
    else:
        ax.scatter(x_[:-2], y_[:-2], label=col_name, color='black', 
                   marker = 'o', facecolors='none', s=20)
        ax.scatter(x_[-1], y_[-1], color='green', marker = 'o', facecolors='none', s=20)
        ax.scatter(x_[-2], y_[-2], color='purple', marker = 'o', facecolors='none', s=20)
        
    lower_error = R75_summ['50%'] - R75_summ['25%']
    upper_error = R75_summ['75%'] - R75_summ['50%']
    
    asymm_error = [lower_error[:-2], upper_error[:-2]]
    ax.errorbar(x_[:-2], R75_summ['50%'][:-2], yerr=asymm_error, 
                 fmt='none', color='black')
    
    asymm_error = [[lower_error[-2]], [upper_error[-2]]]
    ax.errorbar([x_[-2]], [R75_summ['50%'][-2]], yerr=asymm_error, 
                 fmt='none', color='purple')
    
    asymm_error = [[lower_error[-1]], [upper_error[-1]]]
    ax.errorbar([x_[-1]], [R75_summ['50%'][-1]], yerr=asymm_error, 
                 fmt='none', color='green')
       
    ax.hlines(y_[-2], xmin=0, xmax=1, transform=ax.get_yaxis_transform(), 
              color='tab:purple', alpha=0.3, linestyle='--', zorder=1) # adapt
    ax.hlines(y_[-1], xmin=0, xmax=1, transform=ax.get_yaxis_transform(), 
              color='lawngreen', alpha=0.3, linestyle='--', zorder=1) # adapt
    
    if col_name == 'min':
        yEvomin, ySKmin = y_[-2:]
    if col_name == 'max':
        yEvomax, ySKmax = y_[-2:]
        
    # x,y[-1] -- SK (T_a)
    # x,y[-2] -- EvoSK (T_init to T_a)
    
ax.axhspan(yEvomin, yEvomax, facecolor = 'tab:purple', alpha = 0.2, label = 'EvoSK', zorder=1)
ax.axhspan(ySKmin, ySKmax, facecolor = 'lawngreen', alpha = 0.2, label = 'SK (T: T_a)', zorder=1)

ax.legend()
ax.set_ylim((.9, 1.52))
ax.set_xlabel('temperature')
ax.set_title('Optimality of SK (T constant) at alpha = 0.75')
#ax.set_ylim(axs[1].get_ylim())


ax = axs[1]

R_evoSK = R_df.loc[:, 'adapt', 0.0]
R_evoSK_summ = R_evoSK.T.describe().T

x_ = R_evoSK_summ.index
colIDs = list(range(-5, 0, 1))
aRange = [0.3, 0.5, 0.7, 0.75, 0.8, 0.9]

for cId in colIDs:
    col_name = R_evoSK_summ.columns[cId]
    
    y_ = R_evoSK_summ[col_name]
    
    if col_name == '50%':
        ax.plot(x_, y_, color='purple')
    elif col_name in ['25%', '75%']:
        ax.scatter(x_, y_, label=col_name, color='purple', marker='_', s=50)
    else:
        ax.scatter(x_, y_, label=col_name, color='purple', marker='o',
                   facecolors='none', s=20)

lower_error = R_evoSK_summ['50%'] - R_evoSK_summ['25%']
upper_error = R_evoSK_summ['75%'] - R_evoSK_summ['50%']
asymm_error = [lower_error, upper_error]
ax.errorbar(x_, R_evoSK_summ['50%'], yerr=asymm_error, fmt='none', color='purple')

ax.axhspan(yEvomin, yEvomax, facecolor = 'tab:purple', alpha = 0.2, label = 'EvoSK (a: .75)')

ax.legend()
ax.set_xlabel('alpha')
ax.set_ylim((1.27, 1.52))
ax.set_title('Optimality of EvoSK across alpha')


plt.tight_layout()

'''
ax = axs[1]

y0 = R_evoSK_summ['mean']
y_ub = y0 + R_evoSK_summ['std']
y_lb = y0 - R_evoSK_summ['std']

ax.scatter(x_, y0)
ax.plot(x_, y0)

ax.scatter(x_, y_ub, label='ub')
ax.plot(x_, y_ub, label='ub')

ax.scatter(x_, y_lb, label='lb')
ax.plot(x_, y_lb, label='lb')
'''

