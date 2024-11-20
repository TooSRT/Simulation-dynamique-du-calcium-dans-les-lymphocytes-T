import numpy as np

#Units used : nmol/dm^3 = nM (nanomolar) / dm (mètre) / s (secondes) / V (Volt) / A (Ampère) / S (Siemens) / F (Farad) 

class Parameters_system_ODE:
    def __init__(self):
        #-------Constantes du tableau------
        self.Rcell = 8e-5 #dm
        self.fR = 0.25 #Pas d'unité
        self.fV = 0.01
        self.fA = 30
        self.Cm = 28*1e-5 #F/dm^2

        #Ions and potentials:
        self.Temp = 310 #Kelvin
        self.V0 = -60*1e-3 #V 
        self.V_ER = -60*1e-3 #V
        self.V_ER0 = -60*1e-3 #V
        self.C0 = 0.1e3 #nM
        self.C_ER0 = 0.4e6 #nM
        self.C_ext = 2.e3 #nM
        self.delta_V_C = 78*1.e-3 #V
        self.delta_V_C_ER = 63*1.e-3 #mV

        #Calcium buffer: nM
        self.b0 = 100*1.e3 #nM
        self.Kb = 0.1*1.e3 #nM
        self.b_ER0 = 30*1.e6 #nM
        self.K_ERb = 0.1*1.e6 #nM

        #Second messengers: nM
        self.P0 = 8.7 #nM
        self.beta_p = 0.6 #nM/s
        self.gamma_p = 0.01149 #1/s
        self.Cp = 0.5*1.e3 #nM
        self.n_p = 1 #Pas d'unité

        #/µm^2
        self.rho_IP3R = 11.35e10
        self.rho_SERCA = 700e10
        self.rho_PMCA= 68.57e10
        self.rho_CRAC0 = 0.6e10
        self.rho_CRAC_pos = 3.9e10
        self.rho_CRAC_neg = 0.5115e10
        
        #-------Déterminations de constantes--------
        self.Faraday = 96485.3321*1e-9 # A.s/nmol 
        self.R_cte = 8.315e-9 #Molar gaz constant J/(K.nmol) (9)
        self.zCA = 2. #Pas d'unité
        self.V_C_barre = 50*1.e-3 #V (9)
        self.Acell = 804.2e-10 # µm^2 = dm^2 * e-10

        self.Vcyt = 4/3 * np.pi * self.Rcell**3 * (1-self.fV-self.fR**3) #(20)
        self.V_ER_tilde = 4/3 * np.pi * self.Rcell**3 *self.fV #(21)
        self.A_ER = 4*np.pi*self.fA*(3*self.V_ER_tilde/(4*np.pi))**(2./3.) #(22)

        self.Xi = self.Acell/self.Vcyt #(16) dm^2 
        self.Xi_ER = self.A_ER/self.V_ER_tilde   #(17)
        self.Xi_ERC = self.A_ER/self.Vcyt  #(18)

        #--------Constantes--------
        self.g_IP3R_max = 0.81 # C'est une probabilité d'ouverture
        
        self.tau_IP3R = 0.1 #s
        self.tau_PMCA = 50 #s (31)
        self.tau_CRAC = 5. #s (24)
        self.theta = 0.3 #s (29)

        self.n_IP3R_act = 1.9 #Pas d'unité
        self.n_IP3R_inh = 3.9 #Pas d'unité (27)
        self.n_IP3R_C = 4. #pas d'unité (27)
        self.n_SERCA = 2. #pas d'unité (32)
        self.n_CRAC = 4.2 #pas d'unité (25)
        self.n_PMCA = 2. # Hill coefficient

        self.C_IP3R_act = 0.21e3 #nM (27)
        self.C_PMCA = 0.2e3 #nM
        self.C_IP3R_inh_barre = 52.e3 #nM (27)
        self.C_CRAC = 169.e3 #nM (25)
        self.P_IP3R_C = 0.05e3 #nM (27)
        self.C_SERCA= 0.4e3 #nM
        
        self.I_SERCA_BARRE = 3.e-18 #A (32)
        self.I_PMCA_BARRE = 1.e-17 #A (30)
        self.g_CRAC_BARRE = 2.e-15 #S (23)
        self.g_IP3R_barre = 0.064e-12 #S (28)