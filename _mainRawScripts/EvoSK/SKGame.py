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

def computeAcf(mTraj, isStationary=True, numLags=7, steadyIters=1000):
    
    ts = mTraj[:, -steadyIters:, :]
    
    acvf = np.zeros(numLags)
    acf = np.zeros(numLags)
    
    for tau in range(numLags):
        
        #print('tau:', tau)
        
        t1 = 0
        t2 = t1 + tau
        
        while t2 < ts.shape[1]:
            
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
        
        # beta: 1/temperature; **beta->inf: argmax (NOISELESS), beta->0: random-uniform
        self.beta = None #1/.1
        
        # alpha: memory rate; a=1: forgetful, a->0+: long memory
        self.rand_ = None
        self.tempMid = None
        self.tempMin = 0.
        self.tempMax = 2.
        
        self.alpha = .01 #default: 0.01
        
        self.a = None
        self.updateSpin(init=True)
        self.pActs = None
        
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
    
    def initBeta(self, beta=None, adaptBS=False, distribution='uniform', pool=None, spinOnly=False):
        
        if not adaptBS:
            assert beta is not None
            return beta
        
        else:
            if spinOnly:
                return np.inf
            
            #print('here')
            if pool is None:
                return 1/np.random.uniform(self.tempMin, self.tempMax, 1)[0]
            elif distribution=='uniform':
                # pool = beta array; if more precise, input J-matrix to get nearest-neighbors
                return np.random.choice(pool, 1)[0] #1/np.random.choice(pool, 1)[0]
            elif distribution=='avg':
                return np.average(pool) #1/np.average(pool)
            elif distribution=='median':
                return np.median(pool)
            elif distribution=='max':
                return pool[0]
            else:
                raise NotImplementedError()
                
            
    def initRiskVariables(self):
        
        if self.riskMeasure == 'mean':
            # self.alpha
            Qparams = ''
            self.lmbd = ''
            self.lrV = ''
            self.lrE = ''
            
            self.QBs = np.array([0, 0]) # for BS, need lrV?
            self.lrBs = .5 #1.
            
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
            self.Q = (1-self.alpha) * self.Q + self.alpha * self.reward
            self.QBs = (1-self.lrBs) * self.QBs + self.lrBs * self.reward
            
            #print('Q', self.Q)
            #if self.id == 1:
            #    raise ValueError()
                
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
            
            '''
            mvpAct = np.round(softmax(self.Q), 4)#softmax(self.Q) #
            epAct = np.round(softmax(self.Qe), 4) #softmax(self.Qe) #
            
            if mvpAct[1] - epAct[1] > .2: #mvDiff!=eDiff:
                print(self.id)
                print('rwd:', self.reward)
                print('**Qe:', self.Qe)
                print('Var:', self.Qv, self.E, self.Esq)
                print('**Q:', self.Q) #, 'diff:', self.Q[1]-self.Q[0])
                
                #print('Q(+)>Q(-)', mvDiff)
                #print('Qe(+)>Qe(-)', eDiff)
                #print('check:', mvDiff!=eDiff)
                #print('---')
                
                # next check:
                print('---')
                print('Q_pActs:', mvpAct, np.round(softmax(self.Q, 100), 4))
                print('numer:', self.beta*self.Q - max(self.beta*self.Q), 'vs: ', self.Q-max(self.Q))
                print('denom:', np.exp(self.beta*self.Q - max(self.beta*self.Q)).sum())
                print('---')
                print('Qe_pActs:', epAct, np.round(softmax(self.Qe, 100), 4))
                print('===')'''
                
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
        self.BSpool = False # True # False
        self.BSdist = 'avg' #'max' # 'avg', #'uniform'
        self.BSperiod = 1
        self.numOfi = numOfi #5
        
        self.initType = initType
        self.tempflip = tempflip
        self.spinflip = spinflip
        
        self.betas = None
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
        
        self.Qpos = None
        self.Qneg = None
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
            
            '''
            for i in range(N):
                X[i, i] = 0.
                
            return X 
            '''
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

    def respawnAgents(self, smallestIndices, pool=None, fRange=None):
        
        for i in smallestIndices: #[:self.numOfi]:            
            # Reinit agent[i]
            rand_ = self.agents[i].rand_
            tMid = self.agents[i].tempMid
            alpha = self.agents[i].alpha
            lrBs = self.agents[i].lrBs
            
            if not self.spinflip:
                oldspin = self.agents[i].spin
            
            self.agents[i] = SKAgent(i, self.agent_view_sight, self.riskMeasure, self.paramDict)
            self.agents[i].rand_ = rand_
            self.agents[i].tempMid = tMid
            self.agents[i].lrBs = lrBs
            self.agents[i].alpha = alpha

            if not self.spinflip:            
                self.agents[i].spin = oldspin
            
            self.agents[i].beta = self.agents[i].initBeta(adaptBS=self.adaptBS, 
                                                          distribution=self.BSdist,
                                                          pool=pool, spinOnly=self.spinOnly)
            
            if fRange is not None:
                fmin, fmax = fRange
                fmin = max(fmin, 0.)
                fmax = max(fmax, 0.)
                #print('fmin, fmax:', fmin, fmax)
                fnew = np.random.uniform(low=fmin, high=fmax, size=1)[0]
                self.agents[i].QBs = np.array([fnew, fnew]) #np.array([-fnew, fnew])
            
            self.betas[i] = self.agents[i].beta
            
        # i; off for one iter.
    
    def getRange(self):
       
        if self.initType == 0:
            return None
       
        elif self.initType == np.inf:
            return (self.Qs.min(), self.Qs.max())
       
        else:
            return (0., self.initType)
    
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
    

    def BScall(self):
        
        if self.iter > 1:   
            #print(self.iter, list(self.iMin))
            if self.nrandom:
                smallestIndices = np.random.choice(range(self.nAgents), size=self.numOfi)
            else:
                '''
                tempQs = self.Qs[:]
                tempQs[np.where(self.iMin == 1)] = max(self.Qs)
                smallestIndices = np.argpartition(tempQs, self.numOfi) #(self.Qs, self.numOfi)
                
                if self.iter==2:
                    print('w jump: PREVENT CONSECUTIVE KILLING')
                '''
                smallestIndices = np.argpartition(self.Qs, self.numOfi)
                smallestIndices = smallestIndices[:self.numOfi]
            '''
            if self.iter > #3800: (47) #5200: (45)
                print('iter:', self.iter, 'old:', np.where(self.iMin == 1), 'new:', smallestIndices)
                if self.iter > #4000: (47) #5500: (45)
                    raise ValueError()
            '''
            
            f0 = self.Qs[smallestIndices].min() #self.Qs[smallestIndices[0]]
            
            if self.BSpool and self.BSdist == 'max':
                i_max = np.argmax(self.Qs)
                pool = np.array([self.betas[i_max]])
                # range = [self.betas[i], self.betas[i_max]]
            elif self.BSpool:
                pool = np.delete(self.betas, smallestIndices) #pool = np.copy(self.betas) 
            
            else:
                pool = None
            
            #spin_prev = self.agents[smallestIndices[0]].spin   
            self.respawnAgents(smallestIndices, pool, fRange=self.getRange())    
            #spin_aft = self.agents[smallestIndices[0]].spin
            
            '''
            if not self.spinflip and (self.iter > 885 and self.iter < 895):
                print('iter:', self.iter, 'old:', np.where(self.iMin == 1), 'new:', smallestIndices)
                print('bf vs aft respawn:', spin_prev, spin_aft)
                print('f0/fmin:', f0)
            
            if not self.spinflip and self.iter > 2 and np.where(self.iMin == 1)[0] == smallestIndices[0]:
                
                if spin_prev != spin_aft:
                    print('iter:', self.iter, 'old:', np.where(self.iMin == 1), 'new:', smallestIndices)
                    print('bf vs aft respawn:', spin_prev, spin_aft)
                    
                    raise ValueError()
            '''
            
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
                    
        else:
            self.betas = np.array([self.agents[i].beta for i in range(self.nAgents)])
            f0 = None
            
        return f0
        
    def step(self):
        
        self.iter += 1
        
        #start0 = time.time()
        
        if self.adaptBS and (self.iter-1) % self.BSperiod == 0: #self.iter % self.BSperiod == 1:
            if self.withEvent:
                self.BScall_()
            else:
                self.BScall()
        
        # update World's state/attr.
        self.spins = np.array([self.agents[i].spin for i in range(self.nAgents)])
        
        # update Agents' state/attr.
        for agent in self.agents:
            agent.updateReward(self.J, self.spins, self.J0perN, self.h)
            agent.updateQ()
            agent.updateSpin()
            agent.updateIntention()
        
        spinId = (self.spins < 0).astype(int) # +1: 0, -1: 1
        self.rewards = np.array([self.agents[i].reward[spinId[i]] for i in range(self.nAgents)])
        self.negEnergy = np.average(self.rewards)
        
        if self.adaptBS: # For BS compute, use Qs or rewards?
            self.Qs = np.array([self.agents[i].QBs[spinId[i]] for i in range(self.nAgents)])
        
        self.avgIntention = np.average(self.spins)
        
        # Record spins evo
        if self.Qpos is not None:
            self.oldQSign =  1* (self.Qpos > 0) + (-1) * (self.Qpos < 0)
        
        #self.Qpos = np.array([self.agents[i].Q[0] for i in range(self.nAgents)])
        #self.Qneg = np.array([self.agents[i].Q[1] for i in range(self.nAgents)])
        
        #start = time.time()
        # Record .intentions = E[.spins] evo -- .spins are special case of m
        self.intentions = np.array([self.agents[i].m for i in range(self.nAgents)])
        
        #print('8 ', time.time()-start)
        #print('inside', time.time()-start0)
        
        # update World's order parameters
        #self.n_up = np.count_nonzero(self.spins+1)
        #self.n_down = self.nAgents - self.n_up
        
        #order_param_old = self.order_param #print(order_param_old, self.n_up, self.n_down)
        #self.order_param = abs(self.n_up - self.n_down) / (self.nAgents + 0.0)
        #self.order_param_delta = (self.order_param - order_param_old) / order_param_old
    