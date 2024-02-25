#Fichier contenant toutes nos constantes


import numpy as np
from scipy.integrate import solve_ivp
from scipy import constants
import matplotlib.pyplot as plt


#Geometry
Rcell = 8 #µM
fR = 0.25
fV = 0.01
fA = 30
Cm = 28 #fF/µm^2

#Ions and potentials:
T = 310 #Kelvin
V0 = -60 #mV
V_ER0 = -60
C0 = 0.1 #µM
C_ER0 = 0.4 # mM
C_ext = 2 #mM
delta_V_C = 78 #mV
delta_V_C_ER = 63 #mv

#Calcium buffer: µM
b0 = 100
Kb = 0.1
b_ER0 = 30
K_ERb = 0.1

#Second messengers: nM
P0 = 8.7
beta_p = 0.6 #nM/s
gamma_p = 0.01149 #nM/s
Cp = 0.5 #µM
n_p = 1

#Densités: µm^2
rho_IP3R = 11.35
rho_SERCA = 700
rho_PMCA = 68.57
rho_CRAC0 = 0.6
rho_CRAC_pos = 3.9
rho_CRAC_neg = 0.5115


Far_cte = 96485.33212 #Faraday constant C mol^-1
zCA = 2

#--------Fonctions--------

def Hill_function(X,K,n): #fonction de hill (8)
    return X**n/(X**n + K**n)

def BC(b,K,C): #cytosolic calcium-buffer (2)
    return (b*K) / (C+K)**2

def fC(b0,C,Kb):  #fraction of free calcium (3)
    return 1 / ( 1 + b0/(C+Kb) )

#-------Déterminations de constantes-----

I_SERCA = 3.10**(-6) * Hill_function(C, 0.4, 2) #(32)
I_PMCA = 10**(-5)* g_PMCA
#I_CRAC =    #(23)

Vcyt = 4/3 * np.pi * Rcell**3 * (1-fV-fR**3) #(20)
V_ER_tilde = 4/3 * np.pi * Rcell**3 *fV #(21)
A_ER = 4*np.pi*fA*(3*V_ER_tilde/4*np.pi)**(2/3) #(22)

Xi= 804.2/Vcyt #(16)
Xi_ER = A_ER/Vcyt   #(17)
XI_ERC = A_ER/V_ER_tilde #(19)

g_IP3R = 0.81 * Hill_function(C, 0,21, 1.9) # (27) cte dispo sur le papier
C_IP3R_inh = 52*H(P, 0.05, 4)
h_IP3R = Hill_function(C_IP3R_inh, C, 3.9)

