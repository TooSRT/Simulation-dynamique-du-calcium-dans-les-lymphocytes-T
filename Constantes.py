#Fichier contenant toutes nos constantes


import numpy as np
from scipy.integrate import solve_ivp
from scipy import constants
import matplotlib.pyplot as plt

#Numérotation correspondent à celles de l'article 

#-------Constantes du tableau------
#Geometry
Rcell = 8 #µM
fR = 0.25
fV = 0.01
fA = 30
Cm = 28 #fF/µm^2

#Ions and potentials:
Temp = 310 #Kelvin
V0 = V_ER = -60 #mV #V=V0=V_ER
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

#--------Fonctions--------

def Hill_function(X,K,n): #fonction de hill (8)
    return X**n/(X**n + K**n)

def BC(b,K,C): #cytosolic calcium-buffer (2)
    return (b*K) / (C+K)**2

def fC(b0,C,Kb):  #fraction of free calcium (3)
    return 1 / ( 1 + b0/(C+Kb) )

#-------Déterminations de constantes--------
Faraday = 96485.33212 #Faraday constant C mol^-1
zCA = 2
V_C_barre = 50 #(9)

Vcyt = 4/3 * np.pi * Rcell**3 * (1-fV-fR**3) #(20)
V_ER_tilde = 4/3 * np.pi * Rcell**3 *fV #(21)
A_ER = 4*np.pi*fA*(3*V_ER_tilde/4*np.pi)**(2/3) #(22)

Xi = 804.2/Vcyt #(16)
Xi_ER = A_ER/Vcyt   #(17)
Xi_ERC = A_ER/V_ER_tilde #(19)

#Mettre tous les paramtrères de la fonction
def dC_dt(t,Y0,Y1,Y2,Y3,Y4,Y5,Y6): 
    #-------Variables du système-------
    C = Y0
    C_ER = Y1
    P = Y2
    rho_CRAC = Y3
    g_IP3R = Y4
    h_IP3R = Y5
    g_PMCA = Y6

    #--------Initialisation de différentes fonctions/paramètres qui dépendent de nos variables--------
    B_C = BC(b0,Kb,C)
    B_CER = BC(b_ER0,K_ERb,C_ER)
    I_SERCA = 3.10**(-6) * Hill_function(C, 0.4, 2) #(32)
    I_PMCA = 10**(-5)* g_PMCA #(30)
    I_CRAC = 2*(V0 - V_C_barre)    #(23) car V=V0
    V_C_ER_barre = 8.315*Temp*np.ln(C_ER/C)/(zCA*Faraday) - delta_V_C_ER
    rho_CRAC_barre = rho_CRAC_neg  + (rho_CRAC_pos - rho_CRAC_neg)*(1-Hill_function(C_ER,169, 4.2)) #(25) 
    g_IP3R = 0.81 * Hill_function(C, 0,21, 1.9) # (27) Constante dispo sur le papier
    C_IP3R_inh = 52 * Hill_function(P, 0.05, 4)
    h_IP3R = Hill_function(C_IP3R_inh, C, 3.9)
    I_IP3R = 0.064*g_IP3R*h_IP3R*(V0 - V_ER - V_C_ER_barre)

    #--------Constantes--------
    g_IP3R_max = 0.81
    C_IP3R_act = 0.21
    n_IP3R_act = 1.9
    tau_IP3R = 100 # compris entre 1sec et 100ms
    
    theta = 300
    n_IP3R_inh = 3.9

    C_PMCA = 0,1 #µM
    tau_PMCA = 50 #s

    #--------Système d'ODE--------
    dC_dt = -1/(zCA*(Faraday*(1 + B_C))) * (Xi*rho_PMCA*I_PMCA + Xi*rho_CRAC*I_CRAC + Xi_ERC*rho_SERCA*I_SERCA + Xi_ERC*rho_IP3R*I_IP3R)
    dC_ER_dt = Xi_ER*(rho_SERCA*I_SERCA +rho_IP3R*I_IP3R)/(zCA*(Faraday*(1 + B_CER)))       # (4)
    dP_dt = beta_p * Hill_function(C,Cp,n_p)*T(t) - gamma_p*P         # (7)  T(t) vaut toujours 1 ?
    drho_CRAC_dt = (rho_CRAC_barre - rho_CRAC )/ 5      #(24)
    dg_IP3R_dt = (g_IP3R_max*Hill_function(C,C_IP3R_act,n_IP3R_act) - g_IP3R) /tau_IP3R      # (29)
    dh_IP3R_dt = (Hill_function(C_IP3R_inh, C, n_IP3R_inh) - h_IP3R)/theta      # (29)
    dg_PMCA_dt = (Hill_function(C,C_PMCA,2) - g_PMCA)/tau_PMCA      # (31)
    
    return 0



