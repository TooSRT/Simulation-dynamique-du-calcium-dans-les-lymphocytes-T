#Fichier contenant toutes nos constantes


import numpy as np
from scipy.integrate import solve_ivp
from scipy import constants
import matplotlib.pyplot as plt

#Numérotation correspondent à celles de l'article 

#Unités utilisés : nmol/dm^3 = nM (nanomolar) / dm (mètre) / s (secondes) / V (Volt) / A (Ampère) / S (Siemens) / F (Farad) 
#Unités utilisés : nmol/dm^3 = nM (nanomolar) / dm (mètre) / s (secondes) / V (Volt) / A (Ampère) / S (Siemens) / F (Farad) 

class Parameters_system_ODE:
    def __init__(self):
        #-------Constantes du tableau------
        self.Rcell = 8e-5 #dm
        self.fR = 0.25 #Pas d'unité
        self.fV = 0.01
        self.fA = 30
        self.Cm = 28*1e-5 #F/dm^2
        self.Cm = 28*1e-5 #F/dm^2

        #Ions and potentials:
        self.Temp = 310 #Kelvin
        self.V0 = self.V_ER =-60*1e-3 #V 
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

        #Densités surfacique: C/µm^2 = C/dm^2 * e10 = A.s/dm^3 * e10 = mA.s/dm^3 * e13
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
        self.Xi_ER = self.A_ER/self.Vcyt   #(17)
        self.Xi_ERC = self.A_ER/self.V_ER_tilde #(19)
        
        #--------Constantes--------
        self.g_IP3R_max = 0.81 # C'est une probabilité d'ouverture
        
        self.tau_IP3R = 0.1 #s
        self.tau_PMCA = 50. #s (31)
        self.tau_CRAC = 5. #s (24)
        self.tau_PMCA = 50. #s (31)
        self.tau_CRAC = 5. #s (24)
        self.theta = 0.3 #s (29)

        self.n_IP3R_act = 1.9 #Pas d'unité
        self.n_IP3R_inh = 3.9 #Pas d'unité (27)
        self.n_IP3R_C = 4. #pas d'unité (27)
        self.n_SERCA = 2. #pas d'unité (32)
        self.n_CRAC = 4.2 #pas d'unité (25)
        self.n_PMCA = 2. # Hill coefficient

        self.C_IP3R_act = 0.21e3 #nM (27)
        self.C_PMCA = 0.1e3 #nM
        self.C_IP3R_inh_barre = 52.e3 #nM (27)
        self.C_CRAC = 169.e3 #nM (25)
        self.P_IP3R_C = 0.05e3 #nM (27)
        self.C_SERCA= 0.4e3 #nM
        
        self.I_SERCA_BARRE = 3.e-18 #A (32)
        self.I_PMCA_BARRE = 1.e-17 #A (30)
        self.g_CRAC_BARRE = 2.e-15 #S (23)
        self.g_IP3R_barre = 0.064e-12 #S (28)
    
        
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
    
        C_IP3R_inh = self.params.C_IP3R_inh_barre * Hill_function(self.params.P0, self.params.P_IP3R_C, self.params.n_IP3R_C)
        return [self.params.C0, 
                self.params.C_ER0, 
                self.params.P0, 
                self.params.rho_CRAC0 ,  
                self.params.g_IP3R_max * Hill_function(self.params.C0, self.params.C_IP3R_act, self.params.n_IP3R_act),  
                Hill_function(C_IP3R_inh, self.params.C0, self.params.n_IP3R_inh),  
                Hill_function(self.params.C0,self.params.C_PMCA,self.params.n_PMCA),
                self.params.I_SERCA_BARRE * Hill_function(self.params.C0, self.params.C_SERCA, self.params.n_SERCA), #condition init pour I_SERCA
                self.params.I_PMCA_BARRE * self.params.g_IP3R_max * Hill_function(self.params.C0, self.params.C_IP3R_act, self.params.n_IP3R_act), #condition init pour I_PMCA
                self.params.g_CRAC_BARRE*(self.params.V0 - self.params.V_C_barre) ] #condition init pour I_CRAC
    
    # retour d'une array de la taille de la solution (donc 10)
    
    def functionT(self, t):
        if t < 10:
            return 1.
        else:
            return 1.6
        
    
    def ODE_sys(self, t, Y): 
        
        #print("t = " + str(t))
        #print(Y)
        #print("t = " + str(t))
        #print(Y)
        #-------Variables du système-------
        C = Y[0]
        C_ER = Y[1]
        P = Y[2]
        rho_CRAC = Y[3]
        g_IP3R = Y[4]
        h_IP3R = Y[5]
        g_PMCA = Y[6]
        #print("Y = " +str(Y))
        #print("Y = " +str(Y))

        #--------Initialisation de différentes fonctions/paramètres qui dépendent de nos variables--------
        B_C = BC(self.params.b0,self.params.Kb,C)
        B_CER = BC(self.params.b_ER0,self.params.K_ERb,C_ER)
        C_ext = self.params.C0 * np.exp((self.params.V0-self.params.delta_V_C)*self.params.zCA*self.params.Faraday/(self.params.R_cte*self.params.Temp))

        I_SERCA = self.params.I_SERCA_BARRE * Hill_function(C, self.params.C_SERCA, self.params.n_SERCA) #(32)
        I_PMCA = self.params.I_PMCA_BARRE * g_PMCA #(30)

        V_C_ER_barre = self.params.R_cte*self.params.Temp*np.log(C_ER/C)/(self.params.zCA*self.params.Faraday) - self.params.delta_V_C_ER #(9) 
        V_C_barre2 = self.params.R_cte*self.params.Temp*np.log(C_ext/C)/(self.params.zCA*self.params.Faraday) - self.params.delta_V_C #(9) 
        I_CRAC = self.params.g_CRAC_BARRE*(self.params.V0 - V_C_barre2)    #(23) car V=V0

        rho_CRAC_barre = self.params.rho_CRAC_neg  + (self.params.rho_CRAC_pos - self.params.rho_CRAC_neg)*(1.-Hill_function(C_ER,self.params.C_CRAC, self.params.n_CRAC)) #(25) 
        C_IP3R_inh = self.params.C_IP3R_inh_barre * Hill_function(P, self.params.P_IP3R_C,  self.params.n_IP3R_C)
        I_IP3R = self.params.g_IP3R_barre *g_IP3R*h_IP3R*(self.params.V0 - self.params.V_ER - V_C_ER_barre) # (28) 

        


        #--------Système d'ODE--------
        dC_dt = -1./(self.params.zCA*(self.params.Faraday*(1. + B_C))) * (self.params.Xi*self.params.rho_PMCA*I_PMCA 
            + self.params.Xi*rho_CRAC*I_CRAC 
            + self.params.Xi_ERC*self.params.rho_SERCA*I_SERCA 
            + self.params.Xi_ERC*self.params.rho_IP3R*I_IP3R)
        dC_ER_dt = self.params.Xi_ER*(self.params.rho_SERCA*I_SERCA + self.params.rho_IP3R*I_IP3R)/(self.params.zCA*(self.params.Faraday*(1 + B_CER)))       # (4)
        dP_dt = self.params.beta_p * Hill_function(C,self.params.Cp,self.params.n_p)*self.functionT(t) - self.params.gamma_p*P         # (7) 
        drho_CRAC_dt = (rho_CRAC_barre - rho_CRAC )/ self.params.tau_CRAC    #(24)
        dg_IP3R_dt = (self.params.g_IP3R_max*Hill_function(C,self.params.C_IP3R_act,self.params.n_IP3R_act) - g_IP3R) /self.params.tau_IP3R     # (29)
        dh_IP3R_dt = (Hill_function(C_IP3R_inh, C, self.params.n_IP3R_inh) - h_IP3R)/self.params.theta      # (29)
        dg_PMCA_dt = (Hill_function(C,self.params.C_PMCA,self.params.n_PMCA) - g_PMCA)/self.params.tau_PMCA      # (31)

        return [dC_dt,dC_ER_dt,dP_dt,drho_CRAC_dt,dg_IP3R_dt,dh_IP3R_dt,dg_PMCA_dt,I_SERCA,I_PMCA,I_CRAC]

 
# Not part of class    

#--------Fonctions--------
def Hill_function(X,K,n): #fonction de hill (8)
    return X**n/(X**n + K**n)

def BC(b,K,C): #cytosolic calcium-buffer (2)
    return (b*K) / (C+K)**2

def fC(b0,C,Kb):  #fraction of free calcium (3)
    return 1 / ( 1 + b0/(C+Kb) )



def main():
    """ script part
    """
    T = 300 # final time
    T = 300 # final time
    # comment
    calc_sim = Calcium_simulation()
    
    sol = solve_ivp(calc_sim.ODE_sys, [0, T], calc_sim.Y ,method= calc_sim.method_integ,  dense_output=True,  rtol=1e-6, atol=1e-10)
    

    t = np.linspace(0, T, 300)
    z = sol.sol(t)
    plt.plot(t, z.T)
    plt.xlabel('temps')
    plt.xlabel('temps')
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

    #représentation des courbes de l'article
    #figure 3-B
    plt.plot(t, z[7]) 
    plt.plot(t, z[8])
    plt.plot(t, z[9])
    plt.xlabel('temps ')
    plt.legend([r"$I_{SERCA}$", r"$I_{PMCA}$", r"$I_{CRAC}$"])
    plt.title('Calcium simulation')
    plt.show()

    #Figure 5
    plt.semilogy(t,z[1],'r--')
    plt.plot(t,1000*z[2],'b')
    plt.plot(t,1000*z[0],'k')
    plt.xlabel('temps')
    plt.legend([r"$C_{ER}$", r"$1000IP3$", r"$1000C$"])
    plt.show()

if __name__ == "__main__":
    main()   
    #représentation des courbes de l'article
    #figure 3-B
    plt.plot(t, z[7]) 
    plt.plot(t, z[8])
    plt.plot(t, z[9])
    plt.xlabel('temps ')
    plt.legend([r"$I_{SERCA}$", r"$I_{PMCA}$", r"$I_{CRAC}$"])
    plt.title('Calcium simulation')
    plt.show()

    #Figure 5
    plt.semilogy(t,z[1],'r--')
    plt.plot(t,1000*z[2],'b')
    plt.plot(t,1000*z[0],'k')
    plt.xlabel('temps')
    plt.legend([r"$C_{ER}$", r"$1000IP3$", r"$1000C$"])
    plt.show()

if __name__ == "__main__":
    main()   