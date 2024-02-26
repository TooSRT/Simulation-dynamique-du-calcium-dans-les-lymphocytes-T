import Constantes
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


#Numérotation correspondent à celles de l'article 
#-------EDO-------

def dC_dt(t,Y): #Evolution de calcium cytosolic (1)
    C = Y
    #g_pmca, P, rho_crac = Y car variables, utiliser leurs edo dans la fonction ?
    B_C = BC(b0,Kb,C)
    dC = -1/(zCA*(Faraday*(1 + B_C))) * (Xi*rho_PMCA*I_PMCA + Xi*rho_CRAC*I_CRAC + Xi_ERC*rho_SERCA*I_SERCA + Xi_ERC*rho_IP3R*I_IP3R)
    return dC

def dCER_dt(t,Y): #dynamics of the calcium concentration in the ER (4)
    C_ER = Y
    B_CER = BC(b_ER0,K_ERb,C_ER)
    dCer = Xi_ER*(rho_SERCA*I_SERCA +rho_IP3R*I_IP3R)/(zCA*(Faraday*(1 + B_CER)))
    return dCer

def dP_dt(t,Y): #(7)
    return 0

def drho_CRAC_dt(t,Y): #(24)
    rho_CRAC = Y
    tau_CRAC = 5 #à revoir
    drho_CRAC = (rho_CRAC_barre - rho_CRAC )/ tau_CRAC
    return drho_CRAC

def dg_IP3R_dt(t,Y): #(29)
    g_IP3R, C = Y
    g_IP3R_max = 0.81
    C_IP3R_act = 0.21
    n_IP3R_act = 1.9
    tau_IP3R = 100 # compris entre 1sec et 100ms
    dg_IP3R = (g_IP3R_max*Hill_function(C,C_IP3R_act,n_IP3R_act) - g_IP3R) /tau_IP3R
    return dg_IP3R

def dh_IP3R_dt(t,Y): #(29)
    C = Y #rajouter P ?
    theta = 300
    n_IP3R_inh = 3.9
    dh_IP3R = (Hill_function(C_IP3R_inh, C, n_IP3R_inh) - h_IP3R)/theta #définir c_IP3R dans la fonction car variable avec P ?
    return dh_IP3R

def dg_PMCA_dt(t,Y): # (31)
    return 0



#--------Graphiques-------

#Plage de temps
time =10

#Résolution de l'équation différentielle
#C0 condition initiale
sol = solve_ivp(dC_dt, [0, time], C0, args=(), dense_output=True) #g_PMCA, P; rho_crac etc varient

# Temps pour l'interpolation
t = np.linspace(0, 10, 300)

# Interpolation de la solution
C_interp = sol.sol(t)

# Affichage
plt.plot(t, C_interp[0], label='C')
plt.xlabel('Temps')
plt.ylabel('Concentration en calcium')
plt.legend()
plt.title('Evolution de la concentration en Ca2+ au cour du temps')
plt.show()
