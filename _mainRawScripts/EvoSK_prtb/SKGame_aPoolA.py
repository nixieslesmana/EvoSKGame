# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:53:49 2024

@author: nixie
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def softmax(Q, beta=1000., stable = True):
    betaQ = beta*Q
    
    if stable:
        
        return np.exp(betaQ - max(betaQ))/np.exp(betaQ - max(betaQ)).sum()
    else:
        return np.exp(betaQ)/np.exp(betaQ).sum()

def T_c(eps):
    
    v = 1-eps+1/2*eps**2 # Eqn 9, assume sigma=1
    
    # By Fig16, Tc(eps) shd depend on alpha. Larger alpha -> smaller Tc(eps) (see the 'blur')
    
    # alpha=.5, Tc(eps=0, alpha) = 1
    # alpha=.9, Tc(eps=0, alpha) = .5?
    
    return 1/2*((2-eps)**2/np.sqrt(v))

def R_N(eps, N=None, A=1.5):
    
    R_inf = .7631*(2-eps)
    
    if N is None:
        return R_inf
    else:
        return R_inf - A*N**(-2/3)
'''
R_N(0, 256)
R_N(.6, 256)
R_N(.85, 256)
R_N(1.05, 256)
R_N(1.5, 256)'''

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

def computeAcf(mTraj, isStationary=True, numLags=7, steadyIters=1000, out_norm=False):
    
    ts = mTraj[:, -steadyIters:, :] # (128, 200, 25); 200=300:500
     
    if type(numLags) is not int:
        lagList = numLags
        numLags = len(lagList)
    else:
        lagList = range(numLags)
    
    if isStationary:
        numt0 = 1
        t0List = [1]
    else:
        #numt0 = ts.shape[1]-lagList[-1] # actual t0 = iterList[col] + t0 here
    
        numt0 = 10
        t0List = np.logspace(0, np.log10(2/3*ts.shape[1]), numt0).astype(int)
        t0List = sorted(list(set(t0List)))
        numt0 = len(t0List)
        
        print(t0List)
        print(lagList)
    
    acvf = np.zeros((numt0, numLags)) # acvf.shape (15, 38)
    acf = np.zeros((numt0, numLags))
    if out_norm:
        acvf_norm = np.zeros((numt0, numLags)) # acvf.shape (15, 38)
    
    #for t0 in range(numt0): 
    for t0ID in range(numt0): #range(len(t0List)):
        t0 = t0List[t0ID]
        
        for tauID in range(len(lagList)): #range(numLags):
            
            tau = lagList[tauID]
            
            #print(t0, tau)
            #raise ValueError()
            
            t1 = t0 # the t_w
            t2 = t0 + tau
            
            #print(t1, t2)
            if t2 > ts.shape[1]:
                break
            
            #while t2 < ts.shape[1]: # steadyIters
            for _ in range(ts.shape[1]-lagList[-1]-1):    ## 
                #print('t1,t2:', t1, t2)
                
                if t1 == t0: #tauID == 0:
                    X1 = ts[:, t1, :] # 128, 25
                    X2 = ts[:, t2, :]
                else:
                    X1 = np.concatenate((X1, ts[:, t1, :]), axis=1) # 128, 2*25?? up to 128, 25*|T|
                    X2 = np.concatenate((X2, ts[:, t2, :]), axis=1)
                
                if not isStationary:
                    break
                
                t1 += 1
                t2 += 1
            
            if tau==0:
                EX1 = np.average(X1, axis=1) # mX1 = mX2 (256, )
                m = np.average(EX1)
            
            ############ EDIT HERE ##############
            
            EX1X2 = np.average(np.multiply(X1,X2), axis=1) # (256, )
            m1 = np.average(X1, axis=1)
            m2 = np.average(X2, axis=1)
            
            cX1 = np.average(np.multiply(X1,X1), axis=1) # (256, )
            cX2 = np.average(np.multiply(X2,X2), axis=1) # (256, )
            
            acvf[t0ID, tauID] = np.average(EX1X2, axis=0)
            acf[t0ID, tauID] = np.average(np.divide(EX1X2, np.sqrt(np.multiply(cX1, cX2))), axis=0)
            
            if out_norm:
                #acvf_norm[t0, tauID] = acvf[t0, tauID]-np.average(np.multiply(m1, m2),axis=0) # E_i [Em1m2 - Em1*Em2]
                #print('here', np.average(np.multiply(m1, m2),axis=0), m1.mean()*m2.mean())
                acvf_norm[t0ID, tauID] = acvf[t0ID, tauID]-m1.mean()*m2.mean() # E_i[Em1m2] - E_iEm1 * E_i Em2
                # i is on the same level as Js
    # plt.plot(acvfCumt0[11, :])
    if 0 not in lagList:
        m = None
        
    if out_norm:
        return acf, acvf, m, acvf_norm
    
    return acf, acvf, m

def computeAcf_(mTraj, isStationary=True, numLags=7, steadyIters=1000):
    
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

################################# SK CLASSES ##################################

class SKEntityState(object):
  def __init__(self):
    self.id = None
    self.p_pos = None # pos/index @ state-vector

class SKAgentState(SKEntityState):
  def __init__(self):
    super(SKAgentState, self).__init__()
    # up or down
    self.spin = None

class SKAction(object):
  def __init__(self):
    # action
    self.a = None

# properties and state of physical world entity
class SKEntity(object):
  def __init__(self):
    # name
    self.name = ''
    # properties:
    self.size = 0.050
    # entity can move / be pushed
    self.movable = False
    # color
    self.color = None
    # state: position and spin
    self.state = SKEntityState()

class SKAgent(): #(SKEntity):
    def __init__(self, i, view_sight=None, riskMeasure='mean', paramDict={}):
        # super(SKAgent, self).__init__() ##-> inherits attr of SKEntity
        
        ###########
        self.id = i
        self.n_actions = 2
        self.actSpace = np.array([1, -1])
        
        self.a = None
        self.updateSpin(init=True)
        self.pActs = None
        
        self.alpha_init = None
        self.reward = None 
        self.Q = np.array([0, 0])
        
        self.riskMeasure = riskMeasure
        self.paramDict = paramDict
        self.Qparams = self.initRiskVariables()
        
        # m: intention (magnetism)
        self.m = None
        #self.updateIntention()
        
        # state: position and spin
        self.state = None 
        # agents are movable by default
        self.movable = None
        # -1: observe the whole state, 0: itself, 1: neighbour of 1 unit
        self.view_sight = view_sight
        self.spin_mask = None  # the mask for who is neighbours
    
    def assignEvo_p(self):       
        
        if self.riskMeasure == 'pt':
            self.lmbd1, self.lmbd2, self.a1, self.a2, self.rScaler, self.b = self.evo_p
        elif self.riskMeasure == 'tanh':
            self.lmbd1, self.lmbd2, self.b = self.evo_p
        elif self.riskMeasure in ['mean', 'mean-noise']:
            self.alpha, invbeta = self.evo_p
            
            if invbeta > 0:
                self.beta = 1/invbeta
            else:
                self.beta = np.inf
                
            if not self.fixedLrbs:
                self.lrBs = self.alpha
            
            #print('lrbs:', self.id, self.lrBs)
            #print('in assign', self.evo_p) #, self.alpha, invbeta, self.beta)
        else:
            raise NotImplementedError()
    
    def initEvo_p(self, pId=None, pVal=None, adaptBS=False, distribution='uniform', pool=None, spinOnly=False):
        
        if not adaptBS:
            assert pVal is not None
            return pVal
        
        else:
            if spinOnly:
                raise NotImplementedError()
                
            if pool is None:
                return np.random.uniform(self.evo_pMin[pId], self.evo_pMax[pId], 1)[0]
            
            elif distribution=='uniform':
                print('here, pool:', pool)
                # pool = beta array; if more precise, input J-matrix to get nearest-neighbors
                return np.random.choice(pool, 1)[0] #1/np.random.choice(pool, 1)[0]
            
            elif distribution =='uniform-noise':
                return np.clip(np.random.choice(pool, 1)[0]* (np.random.uniform(.5, 1.5, 1)[0]),
                               self.evo_pMin[pId], self.evo_pMax[pId])
            
            elif distribution=='crossAdd':
                return np.clip(np.average(np.random.choice(pool, 2)), self.evo_pMin[pId], self.evo_pMax[pId])
            
            elif distribution=='crossAdd-noise':
                return np.clip(np.average(np.random.choice(pool, 2)) * (np.random.uniform(.9, 1.1, 1)[0]), 
                               self.evo_pMin[pId], self.evo_pMax[pId])
            
            elif distribution=='avg':
                return np.average(pool) #1/np.average(pool)
            
            elif distribution=='avg-noise':
                return np.clip(np.average(pool) * (np.random.uniform(.5, 1.5, 1)[0]), 
                               self.evo_pMin[pId], self.evo_pMax[pId])
                
            elif distribution=='median':
                raise NotImplementedError()
                return np.median(pool)
            elif distribution=='max':
                raise NotImplementedError()
                return pool[0]
            else:
                raise NotImplementedError()        
    
    def initEvoArr(self, evo_p, distribution='uniform', pool=None):
        assert self.adapt_p_t is not None #print(self.adapt_p)
        
        self.evo_p = np.empty(len(self.adapt_p_t))
        
        for pId in range(len(self.adapt_p_t)):     
            #print(pId)
            if self.adapt_p_t[pId]: # specific evo_p needs adaptation
                
                if pool is not None and self.pool_p[pId]:
                    pool_pid = pool[:, pId]
                else:
                    pool_pid = None
                
                self.evo_p[pId] = self.initEvo_p(pId=pId, pVal=None, adaptBS=True, 
                                                 distribution=distribution, pool=pool_pid) # do init for single var
                
            else: # specific evo_p is predetermined, access previous iter's var
                assignedVal = evo_p[pId]
                assert assignedVal is not None
                
                self.evo_p[pId] = self.initEvo_p(pId=pId, pVal=assignedVal, adaptBS=False)
        
        self.assignEvo_p()
        #print(self.adapt_p, self.evo_p, self.beta)
        
            
    def initRiskVariables(self):
        
        if self.riskMeasure in ['mean', 'mean-noise']:
            # self.alpha
            Qparams = ''
            self.lmbd = ''
            self.lrV = ''
            self.lrE = ''
            
            self.QBs = np.array([0, 0]) # for BS, need lrV?
            self.lrBs = None #.5 #1.
            self.fixedLrbs = None
            
            # alpha: memory rate; a=1: forgetful, a->0+: long memory
            self.alpha = None #.01
            self.alphaMin = 0. #.5 #0. 
            self.alphaMax = 1. 
            
            # beta: 1/temperature; **beta->inf: argmax (NOISELESS), beta->0: random-uniform
            self.beta = None #1/.1
            self.rand_ = None
            self.tempMid = None
            self.tempMin = 0. #0.
            self.tempMax = 2. #4. #2.
            
            self.a = None
            self.b = None
            
            self.adapt_p = None
            self.adapt_p_t = None
            self.evo_p = None
            self.evo_pMin =  np.array([self.alphaMin, self.tempMin])
            self.evo_pMax = np.array([self.alphaMax, self.tempMax])
            self.pool_p = None
            
        elif self.riskMeasure == 'mv' or self.riskMeasure == 'mv-no-memory':
            # access from external dict
            self.lmbd = self.paramDict['lmbd']
            self.lrV = self.paramDict['lrV']
            self.lrE = self.paramDict['lrE']
            
            self.E = np.array([0, 0])
            self.Esq = np.array([0, 0])
            self.Qv = np.array([0, 0])
            self.Qe = np.array([0, 0])
            
            self.curQ = np.array([0, 0])
            
            Qparams = '-'+str(self.lmbd)
            
        else:
            raise NotImplementedError()
            
        return Qparams
        
    # method: softmax
    def chooseAction(self, init=False):
        
        if init:
            pActs = np.array([1/2, 1/2])
        elif self.beta == np.inf:
            maxId = np.argmax(self.Q)
            pActs = np.zeros(self.Q.shape)
            pActs[maxId] = 1.
        else:
            betaQ = self.beta*self.Q
            #unstable ver: pActs = (np.exp(betaQ)/np.exp(betaQ).sum()) 
            #alternate formula: 1/2*[1+tanh(betaQ_pos), 1-tanh(betaQ_neg)]
            pActs = (np.exp(betaQ - max(betaQ))/np.exp(betaQ - max(betaQ)).sum())
        
        self.pActs = np.round(pActs, 4)
            
        return np.random.choice(self.actSpace, 1, p=pActs)[0]
    
    def updateSpin(self, init=False):
        
        self.a = self.chooseAction(init)
        self.spin = self.a
        
    def updateMemory(self, Ti=False, correl=None):
        
        '''
        if correl < .95: # Temps too high
            self.alpha -= 5*(.95-correl)
        else: # Temps too low
            self.alpha += 5*(correl-.95)
        '''
        
        if Ti:
            #print(self.evo_p[1], 1/self.beta)
            #raise ValueError()
            
            # 1: f(T) != 1/a0 * g(T). a0 remains the dyn hyperparameter
            #rescaler = np.clip(1/self.evo_p[1], self.alphaMin/self.alpha_init, self.alphaMax/self.alpha_init) # 0 (CK's)
            #rescaler = np.clip(self.tempMax - self.evo_p[1], self.alphaMin/self.alpha_init, self.alphaMax/self.alpha_init) # 1
            
            # 2: a0 linear. a0 cancelled out, not working; UNLESS rescaler transformed to exp(rescaler) or rescaler**2 or rescaler**1/2 or log**1/2
            #grad = -(self.alphaMax/self.alpha_init - self.alphaMin/self.alpha_init)/(self.tempMax - self.tempMin)
            #rescaler = grad * (self.evo_p[1] - self.tempMax)
            
            # 3: sublinear, a0 cancelled out
            
            # 4: combine 1 and tanh, a0 cancelled out.
            #rescaler = np.tanh(1/self.evo_p[1]) / self.alpha_init
            
            ### OTHER RESCALER
            # 1. Decreasing in T_i
            # 2. If possible, smooth transition to min, max (no clipping)
            # 3. Punish large T_i more
            # 4. Shape can be made dependent on a0; pivot at T_i=.5?
            #self.alpha = self.alpha_init * rescaler # !!!!!!!!!!!!
            
            ### tanh, incr in T
            #grad = -(self.alphaMax/self.alpha_init - self.alphaMin/self.alpha_init)/(self.tempMax - self.tempMin)
            #rescaler = grad * (self.evo_p[1] - self.tempMax)
            
            scaler = 2. # steepness of tanh increase
            self.alpha = self.alpha_init + np.tanh(scaler*self.evo_p[1])*(1-self.alpha_init)
            # plt.plot(np.linspace(0, 2., 100), (1-.3)*np.tanh(scaler*np.linspace(0., 2., 100)))
            
            ####
            
            #rescaler = (self.tempMax - self.evo_p[1])/(self.tempMax - self.tempMin) # end up a too close to 1...
            #self.alpha = self.alpha_init + (self.alphaMax - self.alpha_init) * rescaler # rescaler made between 0, 1
        
        if correl is not None:
            
            q_min = .8 #55
            q_max = 1.
            
            assert self.alpha_init is not None
            
            #self.alpha = np.clip(self.alpha_init * (correl - q_init) / (q_tresh - q_init), 0., 1.)
            ratio = (correl - q_min) / (q_max - q_min)
            if ratio < 0.:
                return
            
            rescaler = np.tanh(2*ratio-1)
            #self.alpha = self.alpha_init + (1. - self.alpha_init)*(max(0, rescaler)) + (self.alpha_init - 0.)*(min(0, rescaler))
            self.alpha += (1. - self.alpha)*(max(0, rescaler)) + (self.alpha - 0.)*(min(0, rescaler))
        
            if self.id == 0:
                print('ratio, tanh-rescale:', ratio, rescaler)
        
        self.lrBs = self.alpha
        self.evo_p[0] = self.alpha
        #print('a0:', self.alpha_init)
        
        #raise ValueError()
        
    def updateReward(self, J, spins, J0perN, h): # ~Hamiltonian/energy function
        # assumptions: (1) R = [r, -r], by assump on Q & J
        # (2) J_ii = 0 -> R not affected by S_i(t) thru S_j's entries -> S_i(t) affects R by "sign"
        # (3) H_i = 0 (no idiosyn preference to S_i=+1 or -1), if exists: r_pos + Hi
        # (4) J0 > 0 (avg value of interactions = 0 (self.J0)), if exists: r_pos + self.J0 *1/N*np.sum(self.spins)
        # (5) i has global view of all j's spin/action
        # (6) Jij ~ Gaussian, homogenous
        
        # J0 is not h (external field of SK)
        # If J0: ri(s_i) = s_i * (\sum_j Jij*s_j(t)) + s_i * J0 * (sum_j s_j(t))
        # --> J0 acts more like .mu of P(Jij)
        r_pos = np.matmul(J[self.id, :] + J0perN, spins) + h 
        #print(J[self.id,:])
        #print(J0perN)
        #print(spins)
        
        # If h: ri(s_i) = s_i * (\sum_j Jij*s_j(t)) + s_i * h_i
        # r_pos = np.matmul(J[self.id, :], spins) + h[self.id] 
        
        # using h_i or J0/N results in the same \sum_i E_i;
        # but will drive different system (of individual agents) behaviors.
        self.reward = r_pos * self.actSpace #np.array([r_pos, -r_pos])
        '''all equal: shd not be cause init spins[id] different?
        print(self.spin)
        print('r', self.reward) #[self.spin])'''
        
        
    def updateQ(self): # updateQ > m > spin
        # assumptions: (1) Q = [q, -q]
        
        if self.riskMeasure == 'mean':
            #noise = 1-np.random.normal(0, .01, 1)[0]
            self.Q = (1-self.alpha) * self.Q + self.alpha * self.reward #* noise
            self.QBs = (1-self.lrBs) * self.QBs + self.lrBs * self.reward #* noise
        
        elif self.riskMeasure == 'mean-noise':
            noise = 1-np.random.normal(0, .01, 1)[0]
            self.Q = (1-self.alpha) * self.Q + self.alpha * self.reward * noise
            self.QBs = (1-self.lrBs) * self.QBs + self.lrBs * self.reward * noise
        
        elif self.riskMeasure == 'mv-no-memory':
            # Risk-sensitive Q; memory alpha is wrt .reward
            # estimate of .reward
            self.E = (1-self.alpha) * self.E + self.alpha * self.reward
            # estimate of .reward**2
            self.Esq = (1-self.alpha) * self.Esq + self.alpha * self.reward**2
            
            # estimate of MV
            #self.Var = self.Esq - self.E**2
            self.Qe = self.E
            self.Qv = self.Esq - self.E**2
            self.Q = (1 - self.lmbd) * self.Qe - self.lmbd * (self.Qv)
            
        elif self.riskMeasure == 'mv': 
            #to reproduce 'mv-no-memory', use: .lrV = .lrE = .alpha, .alpha = 1.
            self.Qe = (1-self.lrE) * self.Qe + self.lrE * self.reward
            
            self.E = (1-self.lrV) * self.E + self.lrV * self.reward
            self.Esq = (1-self.lrV) * self.Esq + self.lrV * self.reward**2
            self.Qv = self.Esq - self.E**2 
            
            #allows replacing self.Qe with self.reward, else double memory
            self.curQ = (1 - self.lmbd) * self.Qe - self.lmbd * (self.Qv)
            #Above^ Var with memory model is ok?
            self.Q = (1 - self.alpha) * self.Q + self.alpha * self.curQ
            
        else:
            raise NotImplementedError()
    
    def updateIntention(self):
        
        '''# If mean, can use tanh(beta*Q) formula, else m=softmax(betaQ) * {-1, 1}
        
        betaQ = self.beta*self.Q
        self.m = np.tanh(betaQ[0]) # m_JP
        pJP = np.round([1/2 * (1+np.tanh(betaQ[0])), 1/2 * (1-np.tanh(betaQ[0]))], 2)
        print(self.id, self.pActs, pJP) # Verified: .pActs = pJP (if 'mean', not if 'mv')
        '''
        # If risk, then compute E[.spin_i] = pActs*[+1, -1]
        self.m = np.dot(self.pActs, self.actSpace) # m_pActs
        #print(m_pActs, m_JP)
        
class SKWorld(object):
    ## if needed to map to gym, wrap with Ising-env.py ##    
    def __init__(self, seedJ=None, N=256, eps=0, mu=0, J0=0, h=0, riskMeasure='mean', paramDict={},
                 adaptBS=False, initType=0., tempflip=False, spinflip=True,
                 numOfi=1, withEvent=False, nrandom=False, spinOnly=False):
        # list of agents and entities (can change at execution-time!)
        self.nAgents = N
        self.agent_view_sight = 1
        self.riskMeasure = riskMeasure
        self.paramDict = paramDict
        self.agents = [SKAgent(i, self.agent_view_sight, self.riskMeasure, self.paramDict) for i in range(self.nAgents)]
        
        self.iter = 0
        self.adaptBS = adaptBS
        self.nrandom = nrandom
        self.spinOnly = spinOnly
        
        self.BSpool, self.BSdist = False, 'F'
        #self.BSpool, self.BSdist = True, 'uniform'
        #self.BSpool, self.BSdist = True, 'uniform-noise'
        #self.BSpool, self.BSdist = True, 'crossAdd'
        #self.BSpool, self.BSdist = True, 'crossAdd-noise'
        #self.BSpool, self.BSdist = True, 'avg'
        #self.BSpool, self.BSdist = True, 'avg-noise'
        print('in SKworld >> BSpool, dist:', self.BSpool, self.BSdist)
        
        self.BSperiod = 1
        self.numOfi = numOfi #5
        
        self.initType = initType
        self.tempflip = tempflip
        self.spinflip = spinflip
        
        #self.betas = None
        self.trueEvo_ps = None
        self.iMin = np.zeros(self.nAgents, dtype='int') #-np.ones(self.numOfi, dtype='int')
        
        # event-based BS
        self.withEvent = withEvent
        self.f0 = None
        self.delta = None # set to small, ideally 0 | delta=inf -> becomes BScall()
        self.is_event = None
        #self.eventSize = 0
        
        # ising specific
        self.spins = None # global state, assume: view_sight of agent is all grid
        #self.moment = 1
        self.field = None  # external magnetic field
        #self.temperature = .1  # Temperature (in units of energy)
        #self.interaction = 1  # Interaction (ferro if positive, antiferro if negative)
        self.rewards = None
        self.Qs = None
        self.negEnergy = None # negative energy (-H)
        self.avgIntention = None # M in Fig 8
        self.intentions = None
        self.mflips = None
        
        self.Qpos = None
        self.oldQSign = None
        
        self.order_param = 1.0
        self.order_param_delta = 0.01  # log the change of order parameter for "done"
        self.n_up = 0
        self.n_down = 0
        
        # gauss noise; N(J0/N, stdev=sig/sqrt(N))
        self.J0 = J0 # alternative: np.ones(self.nAgents)*J0
        self.J0perN = self.J0/self.nAgents
        self.h = h
        self.mu = mu
        self.sig = 1
        # degree of symm; eps=0: full JS; eps=1: asymm (J_ij, J_ji independent); eps=2: full JA
        self.eps = eps # range: [0, 2]
        #self.corrJ = self.computeCorrelation()
        
        if seedJ is not None:
            self.seedJ = seedJ
            self.J = self.generateInteractions(seed=seedJ)
        else:
            self.J = self.generateInteractions(constant=self.mu)
        
    def computeCorrelation(self): # paper notation: eta
        
        assert self.mu == 0 and self.sig == 1
        
        v_eps = 1 - self.eps + 1/2 * self.eps**2 # var of mat J (why: cz assuming Gauss0,1?)
        return (1 - self.eps) / v_eps
    
    def generateInteractions(self, seed=None, constant=None, mf=False):
        # Assumptions: J_ii set to 0

        N = self.nAgents

        if constant is not None:
            X = np.ones((N, N)) * constant/N
            
        elif mf:
            # for i
            # Sneppen98: numofNeighbor[i] = Poisson(numNeighbor)
            # Flyv93: numofNeighbor[i] = constant
            
            # J = randomize_ones(numofNeighbor[i]) # no need symmetric??? Jij = 1 but Jji = 0?
            
            # generate Lattice(d) # numNeighbor = fixed funcntion of d
            # J = func(Lattice(d))
            raise NotImplementedError()
            
        else:
            assert seed is not None
            
            np.random.seed(seed)
            
            X = np.random.normal(loc=self.mu/N, scale=self.sig/np.sqrt(N), size=(N, N))
        
        Xl = np.tril(X, k=-1)
        Xu = np.triu(X, k=1) #, X_l = X
        
        Xsymm = Xl + Xl.T
        Xanti = Xu - Xu.T
        
        return (1-self.eps/2)*Xsymm + self.eps/2 *Xanti
    
    
    def getRange(self):
       
        if self.initType == 0:
            return None
       
        elif self.initType == np.inf:
            return (self.Qs.min(), self.Qs.max())
       
        else:
            return (0., self.initType)
    
    def respawnAgents(self, smallestIndices, pool=None, fRange=None):
        
        for i in smallestIndices: #[:self.numOfi]:            
            # Reinit agent[i]
            rand_ = self.agents[i].rand_
            tMid = self.agents[i].tempMid
            #beta = self.agents[i].beta
            #alpha = self.agents[i].alpha
            fixedLrbs = self.agents[i].fixedLrbs
            lrBs = self.agents[i].lrBs
            alpha_init = self.agents[i].alpha_init
            
            evo_p = self.agents[i].evo_p #[:]  # @respawn, get old
            #print('evo_p in respawn', evo_p)
            adapt_p = self.agents[i].adapt_p #[:]
            pool_p = self.agents[i].pool_p
            adapt_p_t = self.agents[i].adapt_p_t
            
            if not self.spinflip:
                oldspin = self.agents[i].spin
            
            self.agents[i] = SKAgent(i, self.agent_view_sight, self.riskMeasure)#, self.paramDict)
            self.agents[i].rand_ = rand_
            self.agents[i].tempMid = tMid
            #self.agents[i].beta = beta
            #self.agents[i].alpha = alpha
            self.agents[i].fixedLrbs = fixedLrbs
            self.agents[i].lrBs = lrBs
            self.agents[i].alpha_init = alpha_init
            
            if not self.spinflip:            
                self.agents[i].spin = oldspin
            
            #print('evo_p in respawn:', evo_p)
            self.agents[i].adapt_p = adapt_p
            self.agents[i].pool_p = pool_p
            self.agents[i].adapt_p_t = adapt_p_t
            self.agents[i].initEvoArr(evo_p, distribution=self.BSdist, pool=pool)
            
            if fRange is not None:
                fmin, fmax = fRange
                fmin = max(fmin, 0.)
                fmax = max(fmax, 0.) #print('fmin, fmax:', fmin, fmax)
                
                fnew = np.random.uniform(low=fmin, high=fmax, size=1)[0]
                self.agents[i].QBs = np.array([fnew, fnew])
            
            self.trueEvo_ps[i, :] = self.agents[i].evo_p[self.agents[i].adapt_p] #self.lmbd2s[i] = self.agents[i].lmbd2 #self.lmbd1s[i] = self.agents[i].lmbd1
            #print('in respawn', self.trueEvo_ps[i, :])
    
    def BScall_(self): # with event
        
        assert not self.BSpool and not self.nrandom
        
        if self.iter > 2: #self.is_event: # if iter > 1
            
            assert self.f0 is not None and self.delta is not None
            
            smallIndices = np.array(range(self.nAgents))[self.Qs < (self.f0 - self.delta)]
            
            if len(smallIndices) == 0:
                self.f0 = self.BScall()
                self.is_event = False
            else:
                
                self.respawnAgents(smallIndices, fRange=self.getRange())
                self.is_event = True
                
                self.iMin[:] = 0
                self.iMin[smallIndices] = 1
            
        else: 
            self.f0 = self.BScall()
            self.is_event = False
    
    def dominationDensity(self, F1, F2): 
        # F1: q, F2: mflip
        # manage density of F1 beyond some value: adds upperbound to the dominating region
        
        F1_ub = .25 #np.inf
        return np.array([np.multiply(np.multiply(F1 > F1[i], F1 < F1_ub), F2 > F2[i]).sum() for i in range(F1.shape[0])])
        
        #return np.array([np.multiply(F1 > F1[i], F2 > F2[i]).sum() for i in range(F1.shape[0])])
        
        
    def nonDominationLevel(self, F1, F2):
        
        #while 
        
        pass
    
    def initPerturb_p(self, evoStop, withPerturb, spinScatterNum, tempScatterStrength, perturbStart, perturbSeed):
        
        self.withPerturb = withPerturb
        self.evoStop = evoStop # np.inf
        self.perturbStart = perturbStart
        
        self.spinScatterNum = spinScatterNum #79 #int(self.nAgents * (1 - 1/2)) #self.nAgents
        #self.spinScatterLoc = None # replace perturb with "respawn"
        
        ###########################
        self.tempScatterType = 'constX' #'constX' #'const' # 'shuffle' # 'constXo'
        self.tempScatterStrength = tempScatterStrength
        ###########################
        
        #self.evoRestart = int(1.5*evoStop)
        self.evoRestart = np.inf
        self.alphaScatterType = 'const'
        self.alphaScatterStrength = 0.
        self.perturbSeed = perturbSeed #1
        
    def perturb(self): # Record: avgRTraj_[2900-3 - 5: 2900-3 + 5]
    
        #'''######### PERTURB ############
        # parameters: stop=2000, perturbStep=3000, 
        # (Type, Strength) = (spinShuffle, num=64), (Tshuffle, num=128), (Tadd, const=.2),
        # Seed = {0, 1, 2, ..}
        if self.iter == self.perturbStart:
            
            np.random.seed(self.perturbSeed)
            init = False
            
            if self.spinScatterNum == 0:
                spinScatterIDs = []
            elif self.spinScatterNum == self.nAgents:
                spinScatterIDs = np.arange(self.nAgents)
            else:    
                spinScatterIDs = list(np.random.choice(range(self.nAgents), self.spinScatterNum, replace=False))
                '''
                smallestIndices = list(np.argpartition(self.Qs, 1)[:1])
                if smallestIndices[0] not in spinScatterIDs:
                    print('NOT IN')
                    spinScatterIDs = spinScatterIDs[:-1] + smallestIndices
                '''
                print('perturbedID, smallestID:', np.sort(spinScatterIDs), np.argpartition(self.Qs, 1)[:1])
                # some combo of Js can release, irrespective of whether imin is included
                
                #np.random.randint(0, self.nAgents, size=self.spinScatterNum) # np.arange(self.nAgents)
            
            # Modify Temperature Distributions
            Tpool = np.array([agent.evo_p[1] for agent in self.agents])
            
            if self.tempScatterType == 'const' and type(self.tempScatterStrength) == str:
                tMid = float(self.tempScatterStrength[1:])
                if tMid == 0.:
                    init = True # Else, not 1/2, 1/2 pActs but always choose argmaxQ=0
                Tpool = np.array([tMid]*len(Tpool))
            
            elif self.tempScatterType == 'const':
                Tpool = np.clip(Tpool + self.tempScatterStrength, max(self.agents[0].tempMin, 1e-3), self.agents[0].tempMax)
            
            elif self.tempScatterType == 'constX':                
                Tpool = np.clip(Tpool*np.round(2**self.tempScatterStrength, 2), max(self.agents[0].tempMin, 1e-3), self.agents[0].tempMax)
                
            elif self.tempScatterType == 'constXo':
                Tpool = np.clip(Tpool*self.tempScatterStrength, max(self.agents[0].tempMin, 1e-3), self.agents[0].tempMax)
            
            elif self.tempScatterType == 'shuffle':
                #np.random.seed(self.perturbSeed + 10000)
                np.random.shuffle(Tpool)
            
            #print(np.sort(Tpool[spinScatterIDs]))
            
            # Modify Alpha (not used yet)
            Apool = np.array([agent.evo_p[0] for agent in self.agents])
            if self.alphaScatterType == 'const':
                Apool = np.clip(Apool + self.alphaScatterStrength, max(self.agents[0].alphaMin, 1e-3), self.agents[0].alphaMax)
            elif self.alphaScatterType == 'shuffle':
                #np.random.seed(self.perturbSeed + 20000)
                np.random.shuffle(Apool)
            
            for agent in self.agents:
                #if self.evoRestart == np.inf:
                ## Do not evolve a, T (at this perturb iter)
                agent.adapt_p_t = [False, False]
                
                #agent.lrBs = lrbs
                #agent.fixedLrbs = fixedLrbs
                #agent.tempMid = tMid
                
                #agent.adapt_p = adapt_p
                #agent.pool_p = pool_p
                
                # Change system "T-states"                
                #print('bf:', agent.alpha, 1/agent.beta, agent.evo_p)
                agent.initEvoArr(evo_p=[Apool[agent.id], Tpool[agent.id]]) # if evoOn, re-init to UNIF
                #print('aft:', agent.alpha, 1/agent.beta, agent.evo_p)
                
                ## Scatter spin (forgetting)
                if agent.id in spinScatterIDs:
                    agent.Q = np.array([0, 0])
                    agent.updateSpin(init) 
                    
                    #spin = updateSpin() #*= -1 # EFFECT ON REWARD WILL CANCEL OUT!!! DO ONLY FOR SOME
                    
            #print([agent.spin for agent in self.agents])
            #raise ValueError()
            
            #for agent in self.agents:
            #    agent.adapt_p_t = [True, True]
        
        elif self.iter == self.evoRestart+1:
            for agent in self.agents:
                agent.adapt_p_t = [True, True]
        
    def BScall(self):
        
        #if self.iter > 1: # DEFAULT!!!!  
        #if self.iter > 500:
        if (self.iter > 1 and self.iter < self.evoStop) or self.iter >= self.evoRestart+1:
        #if (self.iter > 1 and self.iter < 2500): # or self.iter==1500:
            
            if self.nrandom:
                smallestIndices = np.random.choice(range(self.nAgents), size=self.numOfi)
                #print('here')
            else:
                # DEFAULT: 
                smallestIndices = np.argpartition(self.Qs, self.numOfi) # Use self.rewards <- set lrbs=1
                
                '''##################### incorp. mflips + 2-OBJ ###############
                if self.mflips is not None:
                    #self.mflips is not None and np.median(self.mflips) <= .05: 
                    #self.iter % 2 == 1: # PAIRED with TMax = 1.
                    #self.mflips is not None
                    
                    ##### Novelty Seeking (NS) Methods #####
                    #smallestIndices = np.argpartition(self.mflips, self.numOfi) # absolute jump size
                    #smallestIndices = np.argpartition(-np.abs(self.mflips - np.median(self.mflips)), self.numOfi) # relative jump size
                    
                    # q
                    # Entropy (pi*log pi)
                    
                    ##### Balancing f1 (OBJ), f2 (NS): MOEA Methods ######
                    smallestIndices = np.argpartition(-self.dominationDensity(self.Qs, self.mflips), self.numOfi)
                    # ...
                    
                else:
                    smallestIndices = np.argpartition(self.Qs, self.numOfi) # but constrained: does not cause mflips < mmin
                '''
                
                #################### incorp. f2=q for alpha (!= alternate??)
                
                # f2 evo
                # f2 slow update (tunable again.. but should be more acceptable)
                
                # DEFAULT:
                smallestIndices = smallestIndices[:self.numOfi]
                
                '''################## incorp. NEIGHBORS ##########################
                #print('i:', smallestIndices)
                #print('J[i,:]:', self.J[smallestIndices, :]) # shape: (numOfi, 128)
                numOfNi = 1
                NiIDs = np.argpartition(-np.abs(self.J[smallestIndices, :]), numOfNi, axis=1)[:, :numOfNi] # (i, Ni)
                #print(NiIDs[0, :])
                #print(smallestIndices[0])
                #print('max Neighbor i:', self.J[smallestIndices[0], NiIDs[0, :]])
                
                smallestIndices = np.concatenate((np.expand_dims(smallestIndices, 1), NiIDs), axis=1).flatten()
                #print('i w j:', smallestIndices)
                #raise ValueError()
                
                '''#############
                
                ##### incorp NICHE: REPLACE SMALLESTINDICES
                
                
                
                #############
            
            f0 = self.Qs[smallestIndices].min() #self.Qs[smallestIndices[0]]
            
            if self.BSpool and self.BSdist == 'max':
                i_max = np.argmax(self.Qs)
                pool = self.trueEvo_ps[i_max, :] #pool = np.array([self.lmbd1s[i_max]]) #pool = np.array([self.lmbd2s[i_max]])
                
            elif self.BSpool:
                pool = np.delete(self.trueEvo_ps, smallestIndices, axis=0)
                #pool = np.delete(self.lmbd1s, smallestIndices) #pool = np.delete(self.lmbd2s, smallestIndices)
            else:
                pool = None
            
            '''################ incorp: ALTERNATE F1, F2 ###################
            aPeriod = 2
            if (self.iter-1) % aPeriod == 0:
                #print('iter:', self.iter)
                #print('bf:', self.agents[0].adapt_p)
                for agent in self.agents:
                    agent.adapt_p_t = [True, True] # [True, False] for alpha only
                #print('aft:', self.agents[0].adapt_p)
                
            elif (self.iter-1) % aPeriod == 1:
                for agent in self.agents:
                    agent.adapt_p_t = [False, True]
            
            #print('iter:', self.iter, self.agents[smallestIndices[0]].adapt_p_t, self.agents[smallestIndices[0]].evo_p)
            #print(self.trueEvo_ps)
            '''
            
            '''############## control mutant classes
            if False: #self.iter==1500:
                #numOfi = 2
                #smallestIndices = np.argpartition(self.Qs, numOfi)
                #smallestIndices = smallestIndices[:numOfi]
                
                pool = np.array([[.25, .5]])
                self.BSdist = 'uniform'
                for agent in self.agents:
                    agent.pool_p = [True, True]
                 
                #print('agent10 FORGET, bf:', self.agents[10].Q, self.agents[10].spin)
                #self.agents[10].Q = np.array([0, 0])
                #self.agents[10].spin *= -1 
                #print('agent10 FORGET, aft:', self.agents[10].Q, self.agents[10].spin)
                #raise ValueError()
                
                print('SHIFT DIST TO UNIFORM AT STEP 1498 TO 1499')
                '''
                
            '''################ Q-Evo (by a,T class) after freeze
            aTraj = 1/bTraj[:, :, 0, 0]
            tempTraj = 1/bTraj[:, :, 0, 1]
            fTraj = qTraj[:, :, 0]
            rwdTraj = rTraj[:, :, 0]
            
            #for tID in range(1499, 1502):
            #    plt.plot(range(128), qTraj[:, tID])
            
            def getBound(x, y, xE, yE):
                xminID = (xE - x < 0).sum() #- 1
                yminID = (yE - y < 0).sum() #- 1
                
                return [xE[xminID], xE[xminID+1], yE[yminID], yE[yminID+1]]
            
            
            a, temp = .75, 1.5
            binNum = 8
            aE = np.linspace(0., 1., binNum+1)
            tempE = np.linspace(0., 2., binNum+1)
            
            bounds = getBound(a, temp, aE, tempE)
            
            #tStart, tEnd = 1489, 1514
            #tStart, tEnd = 0, 2000
            tStart, tEnd =  1498, 2000
            for i in range(128):
                a_i, T_i = aTraj[i, 1499], tempTraj[i, 1499]
                f_i = fTraj[i, 1499]
                F_i = fTraj[i, tStart:tEnd]
                fitness = True
                
                #f_i = rwdTraj[i, 1499]
                #F_i = rwdTraj[i, tStart:tEnd]
                #fitness = False
                
                #f_i = mTraj[i, 1499, 0]
                #F_i = mTraj[i, tStart:tEnd, 0]
                #fitness = False
                
                if  a_i >= bounds[0] and a_i < bounds[1] and T_i >= bounds[2] and T_i < bounds[3]: #(aTraj[i, 1499], tempTraj[i, 1499]) == (.25, 1.5):
                    plt.plot(range(tStart, tEnd), F_i, label='i={}, a={}, T={}, f={}'.format(i, np.round(a_i,4), np.round(T_i, 4), np.round(f_i, 4)))
                    plt.scatter([1499], [f_i])
            
            if fitness:
                plt.scatter([1499], [fTraj[:, 1499].min()], label='i={}, fmin={}'.format(fTraj[:, 1499].argmin(), 
                                                                                         np.round(fTraj[:, 1499].min(), 4)))
            
            plt.legend(prop={'size': 7}, framealpha=.5, loc='upper right')
            
            bins = [aE, tempE]
            vals = qTraj[:, 1499, 0]
            data = 1/bTraj[:, 1499, 0]
            Hval, xE, yE = plot2dhist.get_binnedFitness(vals, data, bins)
            
            Hval.T
            
            '''
            ###################################################################    
            self.respawnAgents(smallestIndices, pool, fRange=self.getRange())    
            #print('smallId:', smallestIndices)
            
            self.iMin[:] = 0
            self.iMin[smallestIndices] = 1
            
            if self.tempflip:
                #print('HERE')
                ########## TEMPFLIP: insert R,Q,spin,m ##########
                #print('updated spin ~Q=0?', self.agents[smallestIndices[0]].Q)
                # update World's state/attr.
                self.spins = np.array([self.agents[i].spin for i in range(self.nAgents)])
                
                # update Agents' state/attr.
                for agent in self.agents:
                    agent.updateReward(self.J, self.spins, self.J0perN, self.h)
                    agent.updateQ()
                    agent.updateSpin()
                    agent.updateIntention()
            
        else: #self.lmbd1s = np.array([self.agents[i].lmbd1 for i in range(self.nAgents)]) #self.lmbd2s = np.array([self.agents[i].lmbd2 for i in range(self.nAgents)])
            self.trueEvo_ps = np.array([self.agents[i].evo_p[self.agents[i].adapt_p] for i in range(self.nAgents)])
            f0 = None
            
        return f0

    
    def varyJorQ(self):
        
        #'''
        self.varyJ = False #True
        self.Jperiod = 100
        if self.varyJ and self.iter>1 and (self.iter-1) % self.Jperiod == 0:
            '''
            #self.J = self.generateInteractions(self.iter) # only J-init differ, follow up same.
            if (self.iter-1) % (2*self.Jperiod) == 0:
                self.J = self.generateInteractions(self.seedJ)
            else:
                self.J = self.generateInteractions(self.seedJ+1)
            # self.eps = eps[iter], J(seedJ, eps[iter])
            # 1. periodic eps (self.iter-1)//Jperiod == 1, eps= ..; else, eps=..
            # 2. 
            # J(seed[iter], eps=0)
            
            print(self.iter, self.J[0, :10])'''
            
            print('VARY J/Q')
            #print(self.iter, [agent.Q for agent in self.agents[:10]])
            #print(self.iter, [agent.spin for agent in self.agents[:10]])
            for agent in self.agents:
                agent.Q = np.array([0, 0])
                agent.updateSpin()
            #print(self.iter, [agent.Q for agent in self.agents[:10]])
            #print(self.iter, [agent.spin for agent in self.agents[:10]])
        #'''

    def step(self, Ti=False, mArr=None):
        
        self.iter += 1
        
        if self.withPerturb:
            self.perturb()
            
        #print(self.trueEvo_ps.shape)
        
        if self.adaptBS and (self.iter-1) % self.BSperiod == 0:
            if self.withEvent:
                self.BScall_()
            else:
                self.BScall()
                
        #self.varyJorQ()
        
        # update World's state/attr.
        self.spins = np.array([self.agents[i].spin for i in range(self.nAgents)])
        
        # update Agents' state/attr.
        for agent in self.agents:
            agent.updateReward(self.J, self.spins, self.J0perN, self.h)
            
            # CHECK TEMPERATURE, ADJUST ALPHA HERE???
            if Ti:
                agent.updateMemory(Ti=Ti)
                
            elif mArr is not None:
                isStationary = True
                numLags = 2 #7
                steadyIters = mArr.shape[1] #100
                _, acvf, _ = computeAcf(mArr, numLags=numLags, steadyIters=steadyIters)
                
                agent.updateMemory(correl=acvf[0])
                # temps = self.trueEvo_ps[:, 0]
            #elif self.iter == 1:
            #    agent.alpha_init = agent.alpha
            
            agent.updateQ()
            agent.updateSpin()
            agent.updateIntention()
            
        try:
            print('q:', acvf[0])
        except:
            pass
        
        spinId = (self.spins < 0).astype(int) # +1: 0, -1: 1
        self.rewards = np.array([self.agents[i].reward[spinId[i]] for i in range(self.nAgents)])
        self.negEnergy = np.average(self.rewards)
        
        if self.adaptBS: # For BS compute, use Qs or rewards?
            self.Qs = np.array([self.agents[i].QBs[spinId[i]] for i in range(self.nAgents)])
        
        self.avgIntention = np.average(self.spins)
        
        # Record spins evo
        if self.Qpos is not None:
            self.oldQSign =  1* (self.Qpos > 0) + (-1) * (self.Qpos < 0)
        self.Qpos = np.array([self.agents[i].Q[0] for i in range(self.nAgents)])
        
        #start = time.time()
        
        # Record .intensions = E[.spins] evo -- .spins are special case of m
        if self.iter > 1:
            self.mflips = np.abs(np.array([self.agents[i].m for i in range(self.nAgents)]) - self.intentions)
            #print(self.mflips)
            #print(self.intentions)
        self.intentions = np.array([self.agents[i].m for i in range(self.nAgents)])

        #print(self.mflips)
        #print(self.intentions)
        #print('--')
        #print('8 ', time.time()-start)
        #print('inside', time.time()-start0)
        
        # update World's order parameters
        #self.n_up = np.count_nonzero(self.spins+1)
        #self.n_down = self.nAgents - self.n_up
        
        #order_param_old = self.order_param #print(order_param_old, self.n_up, self.n_down)
        #self.order_param = abs(self.n_up - self.n_down) / (self.nAgents + 0.0)
        #self.order_param_delta = (self.order_param - order_param_old) / order_param_old
    