import Constantes
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


#Numérotation correspondent à celles de l'article 
#-------EDO-------

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
    dP_dt = beta_p * Hill_function(C,Cp,n_p)*T(t) - gamma_p*P         # (7)  T(t)=1 ?
    drho_CRAC_dt = (rho_CRAC_barre - rho_CRAC )/ 5      #(24)
    dg_IP3R_dt = (g_IP3R_max*Hill_function(C,C_IP3R_act,n_IP3R_act) - g_IP3R) /tau_IP3R      # (29)
    dh_IP3R_dt = (Hill_function(C_IP3R_inh, C, n_IP3R_inh) - h_IP3R)/theta      # (29)
    dg_PMCA_dt = (Hill_function(C,C_PMCA,2) - g_PMCA)/tau_PMCA      # (31)
    
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
