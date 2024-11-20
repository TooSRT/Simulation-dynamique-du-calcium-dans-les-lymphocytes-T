import numpy as np
from scipy.integrate import solve_ivp
from scipy import constants
import matplotlib.pyplot as plt
from Parameters_EDO import Parameters_system_ODE
        
class Calcium_simulation:
    """ Cette classe implémente la simulation du système d'EDO. 
    Elle rassemble donc les paramêtres et le système d'équation ainsi que la méthode de résolution. 
    """
    def __init__(self):
        """ 
        Constructeurs par défaut de la classe. 
        Le constructeur permets de remplir les paramêtres ainsi que de spécifier l'intégrateur temporel.
        """
        self.params = Parameters_system_ODE()
        self.method_integ = "RK23"
        
        self.Y = self.initial_conditions() # solution des EDOs initialisée avec les conditions initiales 
        self.t = 0. # time

        self.I_PMCA_values= []
        self.I_SERCA_values= []
        self.I_IP3R_values= []
        self.I_CRAC_values= []
        self.net_Ca_out=[]
        self.net_Ca_to_ER=[]

    def initial_conditions(self):
    
        C_IP3R_inh_0 = self.params.C_IP3R_inh_barre * Hill_function(self.params.P0, self.params.P_IP3R_C, self.params.n_IP3R_C)
        
        return [self.params.C0, #C
                self.params.C_ER0, #C_ER
                self.params.P0, #P
                self.params.rho_CRAC0 ,  #rho
                self.params.g_IP3R_max * Hill_function(self.params.C0, self.params.C_IP3R_act, self.params.n_IP3R_act),  #g_IP3R
                Hill_function(C_IP3R_inh_0, self.params.C0, self.params.n_IP3R_inh),  #h_IP3R
                Hill_function(self.params.C0,self.params.C_PMCA,self.params.n_PMCA) #g_PMCA
                ]
    
    # retour d'une array de la taille de la solution (donc 7)
    
    def functionT(self, t): #Utilisé pour déterminer quand le pic de calcium aura lieu
        if t < 10:
            return 1.
        else:
            return 1.6
        
    
    def ODE_sys(self, t, Y): 
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

        #--------Initialisation de différentes fonctions/paramètres qui dépendent de nos variables--------
        B_C = BC(self.params.b0,self.params.Kb,C)
        B_CER = BC(self.params.b_ER0,self.params.K_ERb,C_ER)

        I_SERCA = self.params.I_SERCA_BARRE * Hill_function(C, self.params.C_SERCA, self.params.n_SERCA) #(32)
        I_PMCA = self.params.I_PMCA_BARRE * g_PMCA #(30)

        V_C_ER_barre = self.params.R_cte*self.params.Temp*np.log(C_ER/C)/(self.params.zCA*self.params.Faraday) - self.params.delta_V_C_ER #(9) 
        C_ext = self.params.C0* np.exp(( self.params.V0 + self.params.delta_V_C)*(self.params.zCA*self.params.Faraday)/(self.params.R_cte*self.params.Temp))#calcium extérieur
        V_C_barre2 = self.params.R_cte*self.params.Temp*np.log(C_ext/C)/(self.params.zCA*self.params.Faraday) - self.params.delta_V_C #(9) 
        I_CRAC = self.params.g_CRAC_BARRE*(self.params.V0 - V_C_barre2)   #(23) car V=V0

        rho_CRAC_barre = self.params.rho_CRAC_neg + (self.params.rho_CRAC_pos - self.params.rho_CRAC_neg)*(1.-Hill_function(C_ER,self.params.C_CRAC, self.params.n_CRAC)) #(25) 
        C_IP3R_inh = self.params.C_IP3R_inh_barre * Hill_function(P, self.params.P_IP3R_C,  self.params.n_IP3R_C)
        I_IP3R = self.params.g_IP3R_barre *g_IP3R*h_IP3R*(self.params.V0 - self.params.V_ER - V_C_ER_barre) # (28) 

        #Courant pour une membrane
        '''
        self.I_PMCA_values.append(I_PMCA) #PM
        self.I_CRAC_values.append(I_CRAC) #PM
        self.I_SERCA_values.append(I_SERCA) #ER
        self.I_IP3R_values.append(I_IP3R) #ER
        '''

        
        #Courant de l'ensemble de la cellule
        self.I_PMCA_values.append((I_PMCA*self.params.rho_PMCA)*self.params.Acell) #PM
        self.I_CRAC_values.append((I_CRAC*rho_CRAC)*self.params.Acell) #PM
        self.I_SERCA_values.append((I_SERCA*self.params.rho_SERCA)*self.params.Acell) #ER
        self.I_IP3R_values.append((I_IP3R*self.params.rho_IP3R)*self.params.Acell) #ER
        
        #Totaux des courants entrants et sortants
        self.net_Ca_out.append((I_PMCA*self.params.rho_PMCA + I_CRAC*rho_CRAC)*self.params.Acell)
        self.net_Ca_to_ER.append((I_IP3R*self.params.rho_IP3R + I_SERCA*self.params.rho_SERCA)*self.params.Acell)
        

        '''
        if t < 10:
            I_SERCA = self.params.I_SERCA_BARRE * Hill_function(C, self.params.C_SERCA, self.params.n_SERCA) #(32)
        else:
            I_SERCA = 0
        '''
        
        #--------Système d'ODE--------
        dC_dt = -1./(self.params.zCA*(self.params.Faraday*(1. + B_C))) * (self.params.Xi*self.params.rho_PMCA*I_PMCA 
            + self.params.Xi*rho_CRAC*I_CRAC 
            + self.params.Xi_ERC*self.params.rho_SERCA*I_SERCA 
            + self.params.Xi_ERC*self.params.rho_IP3R*I_IP3R)
        dC_ER_dt = self.params.Xi_ER*(self.params.rho_SERCA*I_SERCA + self.params.rho_IP3R*I_IP3R)/(self.params.zCA*(self.params.Faraday*(1 + B_CER)))       # (4)
        dP_dt = self.params.beta_p * Hill_function(C,self.params.Cp,self.params.n_p)*self.functionT(t) - self.params.gamma_p*P         # (7) 
        drho_CRAC_dt = (rho_CRAC_barre - rho_CRAC)/ self.params.tau_CRAC    #(24)
        dg_IP3R_dt = (self.params.g_IP3R_max*Hill_function(C,self.params.C_IP3R_act,self.params.n_IP3R_act) - g_IP3R) /self.params.tau_IP3R     # (29)
        dh_IP3R_dt = (Hill_function(C_IP3R_inh, C, self.params.n_IP3R_inh) - h_IP3R)/self.params.theta      # (29)
        dg_PMCA_dt = (Hill_function(C,self.params.C_PMCA,self.params.n_PMCA) - g_PMCA)/self.params.tau_PMCA      # (31)

        return [dC_dt,dC_ER_dt,dP_dt,drho_CRAC_dt,dg_IP3R_dt,dh_IP3R_dt,dg_PMCA_dt]


# Not part of class    

#--------Fonctions--------
def Hill_function(X,K,n): #fonction de hill (8)
    return X**n/(X**n + K**n)

def BC(b,K,C): #cytosolic calcium-buffer (2)
    return (b*K) / (C+K)**2

def fC(b0,C,Kb):  #fraction of free calcium (3)
    return 1 / ( 1 + b0/(C+Kb) )

