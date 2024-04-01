#Fichier contenant toutes nos constantes


import numpy as np
from scipy.integrate import solve_ivp
from scipy import constants
import matplotlib.pyplot as plt

#Numérotation correspondent à celles de l'article 

#Unités utilisés : nmol/dm^3 = nM (nanomolar) = nmol/µm^3 e-15 / µm (mètre) / s (secondes) / mV (Volt) / nA (Ampère) / mS (Siemens) / mF (Farad) 

class Parameters_system_ODE:
    def __init__(self):
        #-------Constantes du tableau------
        self.dict_params = {}
        #Geometry
        self.dict_params["Rcell"] = 8 #µm
        self.dict_params["fR"] = 0.25 #Pas d'unité
        self.dict_params["fV"] = 0.01
        self.dict_params["fA"] = 30
        self.dict_params["Cm"] = 28e-12 #mF/µm^2
        self.dict_params["Acell"] = 804.2 #µm^2

        #Ions and potentials:
        self.dict_params["Temp"] = 310 #Kelvin
        self.dict_params["V0"] = self.dict_params["V_ER"] = -60 #mV #V=V0=V_ER
        self.dict_params["V_ER0"] = -60 #mV
        self.dict_params["C0"] = 0.1e-12 #nmol/µm^3
        self.dict_params["C_ER0"] = 0.4e-9 #nmol/µm^3
        self.dict_params["C_ext"] = 2e-9 #nmol/µm^3
        self.dict_params["delta_V_C"] = 78 #mV
        self.dict_params["delta_V_C_ER"] = 63 #mV

        #Calcium buffer: nM
        self.dict_params["b0"] = 100e-12 #nmol/µm^3
        self.dict_params["Kb"] = 0.1e-12 #nmol/µm^3
        self.dict_params["b_ER0"] = 30e-9 #nmol/µm^3
        self.dict_params["K_ERb"] = 0.1e-9 #nmol/µm^3

        #Second messengers: nM
        self.dict_params["P0"] = 8.7e-15 #nmol/µm^3
        self.dict_params["beta_p"] = 0.6e-15 #nmol/(µm^3.s)
        self.dict_params["gamma_p"] = 0.01149e-15 #nmol/µm^3.s
        self.dict_params["Cp"] = 0.5e-12 #nmol/µm^3
        self.dict_params["n_p"] = 1 #Pas d'unité

        #Densités surfacique: µm^2 
        self.dict_params["rho_IP3R"] = 11.35
        self.dict_params["rho_SERCA"] = 700
        self.dict_params["rho_PMCA"]= 68.57
        self.dict_params["rho_CRAC0"] = 0.6
        self.dict_params["rho_CRAC_pos"] = 3.9
        self.dict_params["rho_CRAC_neg"] = 0.5115
        
        #-------Détermination de constantes--------
        self.dict_params["Faraday"] = 96485.33212 #Faraday constant C/mol =  nA.s/nmol 
        self.dict_params["R_cte"] = 8.315e-3 #Molar gaz constant J/(K.mol) = kg.µm^2/(s^2.K.nmol) * e-3 (9)
        self.dict_params["zCA"] = 2. #Pas d'unité
        self.dict_params["V_C_barre"] = 50 #mV (9)

        self.dict_params["Vcyt"] = 4/3 * np.pi * self.dict_params["Rcell"]**3 * (1-self.dict_params["fV"]-self.dict_params["fR"]**3) #(20)
        self.dict_params["V_ER_tilde"] = 4/3 * np.pi * self.dict_params["Rcell"]**3 *self.dict_params["fV"] #(21)
        self.dict_params["A_ER"] = 4*np.pi*self.dict_params["fA"]*(3*self.dict_params["V_ER_tilde"]/4*np.pi)**(2./3.) #(22)

        self.dict_params["Xi"] = self.dict_params["Acell"]/self.dict_params["Vcyt"] #(16) µm^2 
        self.dict_params["Xi_ER"] = self.dict_params["A_ER"]/self.dict_params["Vcyt"]   #(17)
        self.dict_params["Xi_ERC"] = self.dict_params["A_ER"]/self.dict_params["V_ER_tilde"] #(19)
        

         #--------Constantes--------     
        self.dict_params["tau_IP3R"] = 0.1 #s
        self.dict_params["theta"] = 0.3 #s (29)
        self.dict_params["tau_PMCA"] = 50 #s (31)
        self.dict_params["tau_CRAC"] = 5 #s (24)

        #Pas d'unité
        self.dict_params["n_IP3R_act"] = 1.9 
        self.dict_params["n_PMCA"] = 2. #(31)
        self.dict_params["n_IP3R_inh"] = 3.9 #(27)
        self.dict_params["n_CRAC"] = 4.2 #(25)
        self.dict_params['n_IP3R_C'] = 4. #(27)
        self.dict_params['n_SERCA'] = 2. #(32)

        # nmol/µm^3
        self.dict_params["C_IP3R_act"] = 0.21e-12 #nmol/µm^3
        self.dict_params["C_PMCA"] = 0.1e-12 #nmol/µm^3
        self.dict_params["C_IP3R_inh_barre"] = 52e-12 #nmol/µm^3
        self.dict_params["C_CRAC"] = 169e-12 #nmol/µm^3 (25)
        self.dict_params["C_SERCA"] = 0.4e-12 #nmol/µm^3 (32)
        self.dict_params["P_IP3R_C"] = 0.05e-12 #nmol/µm^3 (27)
        
        self.dict_params["I_SERCA_BARRE"] = 3e-9 #nA (32)
        self.dict_params["I_PMCA_BARRE"] = 1e-8 #nA (30)
        self.dict_params["g_IP3R_max"] = 0.81e-9 #Unité non précisé sur l'article ? mS
        self.dict_params["g_PMCA_BARRE"] = 2e-12 #mS (23)
        self.dict_params["g_IP3R_barre"] = 0.064e-9 #mS (28)


class Calcium_simulation:
    """ Cette classe implémente la simulation du système d'EDO. 
    Elle rassemble donc les paramêtres et le système d'équation ainsi que la méthode de résolution. 
    """
    def __init__(self):
        """ Constructeurs par défaut de la classe. 
        Le constructeur permets de remplir les paramêtres ainsi que de spécifier l'intégrateur temporel.
        """
        self.params = Parameters_system_ODE()
        self.method_integ = "RK23"
        
        self.Y = self.initial_conditions() # solution des EDOs initialisée avec les conditions initiales 
        
        self.t = 0. # time
        
    def initial_conditions(self):
        C_IP3R_inh = self.params.dict_params["C_IP3R_inh_barre"] * Hill_function(self.params.dict_params["P0"], self.params.dict_params['P_IP3R_C'], self.params.dict_params['n_IP3R_C'])
    
        return [self.params.dict_params["C0"], self.params.dict_params["C_ER0"], self.params.dict_params["P0"], self.params.dict_params["rho_CRAC0"], self.params.dict_params["g_IP3R_max"] * Hill_function(self.params.dict_params["C0"], self.params.dict_params["C_IP3R_act"], self.params.dict_params["n_IP3R_act"]),  Hill_function(C_IP3R_inh, self.params.dict_params["C0"], self.params.dict_params['n_IP3R_inh']),  Hill_function(self.params.dict_params["C0"],self.params.dict_params["C_PMCA"], self.params.dict_params["n_PMCA"])  ] # retour d'une array de la taille de la solution (donc 7)
 
# Not part of class    

#--------Fonctions--------
def Hill_function(X,K,n): #fonction de hill (8)
    return X**n/(X**n + K**n)

def BC(b,K,C): #cytosolic calcium-buffer (2)
    return (b*K) / (C+K)**2

def fC(b0,C,Kb):  #fraction of free calcium (3)
    return 1 / ( 1 + b0/(C+Kb) )
    
    

def ODE_sys(t, Y, C0, b0, Kb, b_ER0, K_ERb, V0, V_C_barre, Temp, R_cte, zCA, Faraday, delta_V_C_ER, rho_CRAC_neg, rho_CRAC_pos, V_ER,Xi, rho_PMCA, Xi_ERC, rho_SERCA, rho_IP3R, Xi_ER, beta_p , Cp, n_p, gamma_p, g_IP3R_max, C_IP3R_act, n_IP3R_act, tau_IP3R, n_IP3R_inh, theta , C_PMCA, tau_PMCA, C_IP3R_inh_barre, tau_CRAC, I_SERCA_BARRE, I_PMCA_BARRE, g_CRAC_BARRE, g_IP3R_barre, C_CRAC, C_SERCA, P_IP3R_C,n_CRAC, n_IP3R_C, n_SERCA, n_PMCA): 
    #-------Variables du système-------
    C = Y[0]
    C_ER = Y[1]
    P = Y[2]
    rho_CRAC = Y[3]
    g_IP3R = Y[4]
    h_IP3R = Y[5]
    g_PMCA = Y[6]

    #--------Initialisation de différentes fonctions/paramètres qui dépendent de nos variables--------
    B_C = BC(b0,Kb,C)
    B_CER = BC(b_ER0,K_ERb,C_ER)


    #Passer les pico-ampère en mili-ampère pour s'adapter au mili-Volt 
    I_SERCA = I_SERCA_BARRE * Hill_function(C, C_SERCA, n_SERCA) #(32)
    I_PMCA = I_PMCA_BARRE * g_PMCA #(30) 
    I_CRAC = g_CRAC_BARRE*(V0 - V_C_barre)    #(23) car V=V0
    V_C_ER_barre = R_cte*Temp*np.log(C_ER/C)/(zCA*Faraday) - delta_V_C_ER #(9) 
    
    rho_CRAC_barre = rho_CRAC_neg  + (rho_CRAC_pos - rho_CRAC_neg)*(1-Hill_function(C_ER,C_CRAC, n_CRAC)) #(25) 
    C_IP3R_inh = C_IP3R_inh_barre * Hill_function(P, P_IP3R_C, n_IP3R_C)
    I_IP3R = g_IP3R_barre *g_IP3R*h_IP3R*(V0 - V_ER - V_C_ER_barre) # (28) 


    #--------Système d'ODE--------
    dC_dt = -1/(zCA*(Faraday*(1 + B_C))) * (Xi*rho_PMCA*I_PMCA + Xi*rho_CRAC*I_CRAC + Xi_ERC*rho_SERCA*I_SERCA + Xi_ERC*rho_IP3R*I_IP3R)
    dC_ER_dt = Xi_ER*(rho_SERCA*I_SERCA + rho_IP3R*I_IP3R)/(zCA*(Faraday*(1 + B_CER)))       # (4)
    dP_dt = beta_p * Hill_function(C,Cp,n_p)*t - gamma_p*P         # (7) 
    drho_CRAC_dt = (rho_CRAC_barre - rho_CRAC )/ tau_CRAC    #(24)
    dg_IP3R_dt = (g_IP3R_max*Hill_function(C,C_IP3R_act,n_IP3R_act) - g_IP3R) /tau_IP3R     # (29)
    dh_IP3R_dt = (Hill_function(C_IP3R_inh, C, n_IP3R_inh) - h_IP3R)/theta      # (29)
    dg_PMCA_dt = (Hill_function(C,C_PMCA,n_PMCA) - g_PMCA)/tau_PMCA      # (31)

    return [dC_dt,dC_ER_dt,dP_dt,drho_CRAC_dt,dg_IP3R_dt,dh_IP3R_dt,dg_PMCA_dt]


def main():
    """ script part
    """
    T = 1 # final time
    # comment
    calc_sim = Calcium_simulation()
    sol = solve_ivp(ODE_sys, [0, T], calc_sim.Y ,method= calc_sim.method_integ, args= (calc_sim.params.dict_params["C0"],calc_sim.params.dict_params["b0"], calc_sim.params.dict_params["Kb"],   calc_sim.params.dict_params["b_ER0"], calc_sim.params.dict_params["K_ERb"],  calc_sim.params.dict_params["V0"],  calc_sim.params.dict_params["V_C_barre"],  calc_sim.params.dict_params["Temp"],  calc_sim.params.dict_params["zCA"],  calc_sim.params.dict_params["Faraday"],  calc_sim.params.dict_params["delta_V_C_ER"],  calc_sim.params.dict_params["rho_CRAC_neg"],  calc_sim.params.dict_params["rho_CRAC_pos"],  calc_sim.params.dict_params["V_ER"], calc_sim.params.dict_params["Xi"],  calc_sim.params.dict_params["rho_PMCA"],  calc_sim.params.dict_params["Xi_ERC"],  calc_sim.params.dict_params["rho_SERCA"],  calc_sim.params.dict_params["rho_IP3R"],  calc_sim.params.dict_params["Xi_ER"],  calc_sim.params.dict_params["beta_p"] ,  calc_sim.params.dict_params["Cp"],  calc_sim.params.dict_params["n_p"],  calc_sim.params.dict_params["gamma_p"],  calc_sim.params.dict_params["g_IP3R_max"],  calc_sim.params.dict_params["C_IP3R_act"],  calc_sim.params.dict_params["n_IP3R_act"],  calc_sim.params.dict_params["tau_IP3R"],  calc_sim.params.dict_params["n_IP3R_inh"],  calc_sim.params.dict_params["theta"] ,  calc_sim.params.dict_params["C_PMCA"],  calc_sim.params.dict_params["tau_PMCA"] ,calc_sim.params.dict_params["C_IP3R_inh_barre"], calc_sim.params.dict_params["tau_CRAC"], calc_sim.params.dict_params["I_SERCA_BARRE"], calc_sim.params.dict_params["I_PMCA_BARRE"], calc_sim.params.dict_params["g_PMCA_BARRE"], calc_sim.params.dict_params["g_IP3R_barre"], calc_sim.params.dict_params["C_CRAC"], calc_sim.params.dict_params["R_cte"], calc_sim.params.dict_params['C_SERCA'], calc_sim.params.dict_params['P_IP3R_C'], calc_sim.params.dict_params['n_CRAC'], calc_sim.params.dict_params['n_IP3R_C'], calc_sim.params.dict_params['n_SERCA'], calc_sim.params.dict_params['n_PMCA']),  dense_output=True)
    

    t = np.linspace(0, T, 300)
    z = sol.sol(t)
    plt.plot(t, z.T)
    plt.xlabel('t')
    plt.legend([r"$C$", r"$C_{ER}$", r"$P$", r"$\rho_{CRAC}$", r"$g_{IP3R}$", r"$h_{IP3R}$", r"$g_{PMCA}$"])
    plt.title('Calcium simulation')
    plt.show()
    
    #Tracer chaque courbe séparément
    plt.figure(figsize=(10, 8))
    for i, var_name in enumerate([r"$C$", r"$C_{ER}$", r"$P$", r"$\rho_{CRAC}$", r"$g_{IP3R}$", r"$h_{IP3R}$", r"$g_{PMCA}$"]):
        plt.subplot(4, 2, i+1)
        plt.plot(t, z[i])
        plt.xlabel('t')
        plt.title(var_name)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    
        





