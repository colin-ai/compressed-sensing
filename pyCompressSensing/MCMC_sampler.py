import sys,os
import numpy as np
import readline
from rpy2.rinterface import R_VERSION_BUILD
print(R_VERSION_BUILD)
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import math
import datetime
import pandas as pd
import seaborn as sns

from scipy.fftpack import fft, ifft


#Import compressivesesning libs
#sys.path.insert(0,"..")

from SignalFrame import *
#Set proxy if needed
#os.environ["http_proxy"] = "http://proxy-internet-aws-eu.subsidia.org:3128"
#os.environ["https_proxy"] = "http://proxy-internet-aws-eu.subsidia.org:3128"

#https://rpy2.readthedocs.io/en/version_2.8.x/introduction.html#r-packages

# import rpy2's package module
import rpy2.robjects.packages as rpackages

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# Install dependencies
packnames = ['GeneralizedHyperbolic','MASS']
names_to_install = [x for x in packnames if (not rpackages.isinstalled(x))]

# R vector of strings
from rpy2.robjects.vectors import StrVector

# Selectively install what needs to be install.
# We are fancy, just because we can.

if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
    
    
#Imports R packages
#see https://cran.r-project.org/web/views/Distributions.html
from rpy2.robjects.packages import importr
r_GIG = importr('GeneralizedHyperbolic') #used for Generalized Gaussian Inverse distribution
r_stats = importr('stats') #used for Normal, gamma distribution
r_MASS = importr('MASS') #used for multivariate Normal distribution




#General functions
def sample_gauss(size):
    return np.random.normal(loc=0.0, scale=1.0,size=size)

def psi_inv(x,scale):
    return ifft(x)*scale

def psi_invt(x,scale):
    return fft(x)/scale



def sample_x(y, tau, _lambda, gamma_r,gamma_i,v,alpha,phi,psi_inv_phit_y):
    N=phi.shape[1]
    M=phi.shape[0]
    
    
    LHS_r = np.array([math.pow((1/alpha) + _lambda*(1/gamma_ind),-1) for gamma_ind in gamma_r])
    LHS_i = np.array([math.pow((1/alpha) + _lambda*(1/gamma_ind),-1) for gamma_ind in gamma_i])
    
    m = tau*psi_inv_phit_y+v
    
    mu_tilde_r = np.real(LHS_r*np.real(m))
    mu_tilde_i = np.real(LHS_i*np.imag(m))
    
    gauss_v3 = sample_gauss(N)
    gauss_v4 = sample_gauss(N)
    
    
    LHS2_r = np.array([math.pow((1/alpha) + _lambda*(1/gamma_ind),-1/2) for gamma_ind in gamma_r])
    LHS2_i = np.array([math.pow((1/alpha) + _lambda*(1/gamma_ind),-1/2) for gamma_ind in gamma_i])
    
    x_r = mu_tilde_r + LHS2_r * gauss_v3
    x_i = mu_tilde_i + LHS2_i * gauss_v4
    
    x = x_r + x_i*1j
    
    return x


def sample_v(x,tau,alpha,phi):
    N=phi.shape[1]
    M=phi.shape[0]
    
    mu = (1/alpha)*x-tau*psi_invt(phi.T@phi@psi_inv(x,math.sqrt(N)),math.sqrt(N))
    
    
    V2_1 = sample_gauss(N) + sample_gauss(N)*1j
    V2_1 = math.sqrt(1/2)*V2_1
    V2_2 = sample_gauss(M) + sample_gauss(M)*1j
    V2_2 = math.sqrt(1/2)*V2_2
    
    sqrt_alpha = math.sqrt(alpha)
    sqrt_1_alpha = math.sqrt(1-alpha)
    
    
    V2 = math.pow(alpha,-1/2)*V2_1-sqrt_alpha*phi.T@phi@V2_1 + (sqrt_1_alpha)*phi.T@V2_2
    
    V1 = psi_invt(V2,math.sqrt(N))
    
    V = mu + V1
    
    #Condition for positive definite condition
    #init tau à 1/std(y)
    
    return V


def sample_gamma(_lambda,x):
    gamma_n = []
    for x_n in x :
        b_chi =  _lambda*math.pow(np.real(x_n),2)
        a_psi = 1
        c_lambda = 1/2
        param = robjects.FloatVector([b_chi,a_psi,c_lambda])
        #Dnas le package les param : c(chi, psi, lambda) = b,a,c
        gamma_n.append(np.array(r_GIG.rgig(1,param = param))[0])
    return np.array(gamma_n)


def sample_lambda(x,gamma_r,gamma_i,a_lambda = 1e-6,b_lambda = 1e-6):
    N = x.shape[0]
    shape = N+a_lambda
    x_r = np.real(x)
    x_i = np.imag(x)
    
    rate = np.real(b_lambda + (1/2)*((x_r.T@np.linalg.inv(np.diag(gamma_r))@x_r) +\
                              (x_i.T@np.linalg.inv(np.diag(gamma_i))@x_i)))

    return np.array(r_stats.rgamma(1, shape = shape, rate = rate))

def sample_tau(x,y,phi,a_tau = 1e-6,b_tau = 1e-6):
    M = y.shape[0]
    N = x.shape[0]
    #M au lieu de M/2 car partie relle et partie complexe
    shape = (M)+a_tau
    rate = b_tau + (1/2)*(math.pow(np.linalg.norm(y - phi@psi_inv(x,math.sqrt(N)),2),2))
    return np.array(r_stats.rgamma(1, shape = shape, rate = rate)) 


def gibbs(y, iters, init, hypers,phi):
    #Init parameters
    _lambda = init["lambda"]
    tau = init["tau"]
    gamma_r = init["gamma"]
    gamma_i = init["gamma"]
    v = init["v"]
    
    
    #Optimisation
    M=phi.shape[0]
    N=phi.shape[1]
    psi_inv_phit_y = psi_inv(phi.T@y,math.sqrt(N))
    print(psi_inv_phit_y.shape)
    
    trace = np.zeros((iters, 3), dtype=object) ## trace to store values of x,tau,lambda,gamma
    
    for it in range(iters):
        alpha = 0.1*tau
        
        x = sample_x(y, tau, _lambda, gamma_r,gamma_i,v,alpha,phi,psi_inv_phit_y)
        x_r = np.real(x)
        x_i = np.imag(x)
        _lambda = sample_lambda(x,gamma_r,gamma_i,hypers["a_lambda"],hypers["b_lambda"])
        tau = sample_tau(x,y,phi,hypers["a_tau"],hypers["b_tau"])
        
        gamma_r = sample_gamma(_lambda,x_r)
        gamma_i = sample_gamma(_lambda,x_i)
        
        v = sample_v(x,tau,alpha,phi)
        
#         print(datetime.datetime.now()-now)
#         if (it % 100) == 1 :
        print(str(it)+": "+str(np.linalg.norm(x,1))+" "+str(tau)+" "+str(_lambda))
        
        trace[it,:] = np.array((np.linalg.norm(x,1), _lambda, tau), dtype=object)
        
    trace = pd.DataFrame(trace)
    trace.columns = ['x','lambda', 'tau']

    return trace


#Compression of signal ( uniform)
sf = SignalFrame()
s01 = sf.read_wave('../data/signal.wav', coeff_amplitude=1/10000,trunc=0.0125)
s01.sampler_uniform(rate=0.3)
signal_sampled_uni_r03 = s01.time_sampled
phi_uni_r03 = s01.phi



y = signal_sampled_uni_r03
N = len(s01.time_sampled)


#Init of hyperparameters
## specify initial values
init = {"x": np.zeros(N),
        "tau": 1,
        "lambda": 1,
        "gamma": np.ones(N),
        "v": np.zeros(N),
        }

## specify hyper parameters
hypers = {"a_lambda": 1e-6,
         "b_lambda": 1e-6,
         "a_tau": 1e-6,
         "b_tau": 1e-6
         }

iters = 600
trace = gibbs(signal_sampled_uni_r03, iters, init, hypers,phi_uni_r03)
