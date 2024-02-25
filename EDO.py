import Constantes
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#-------EDO-------

def Cytoclosic_calcium(t): #Evolution de calcium cytosolic (1)
    B_C=BC(b0,Kb,dC_dt)
    dC_dt = -1/(zCA*(Far_cte*(1 + B_C))) * (Xi*rho_PMCA*I_PMCA + Xi*rho_CRAC*I_CRAC + Xi_ERC*rho_SERCA*I_SERCA + Xi_ERC*rho_IP3R*I_IP3R)
    return dC_dt

def C_ER(t): #dynamics of the calcium concentration in the ER (4)
    B_CER = BC(b_ER0,K_ERb,C_ER)
    dCer_dt = Xi_ER*(rho_SERCA*I_SERCA +rho_IP3R*I_IP3R)/(zCA*(Far_cte*(1 + B_CER)))
    return dCer_dt 

def IP3(t): #(7)
    return 0
