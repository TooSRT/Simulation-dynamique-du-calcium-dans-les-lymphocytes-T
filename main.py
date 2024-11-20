import numpy as np
from scipy.integrate import solve_ivp
from scipy import constants
import matplotlib.pyplot as plt
from Parameters_EDO import Parameters_system_ODE
from EDO_sys_calcium import Calcium_simulation

def main():
    """ 
    Script part
    """
    T = 300 # final time
    calc_sim = Calcium_simulation()
    
    sol = solve_ivp(calc_sim.ODE_sys, [0, T], calc_sim.Y ,method= calc_sim.method_integ,  dense_output=True,  rtol=1e-6, atol=1e-10)

    t = np.linspace(0, T, 300)
    z = sol.sol(t)
    '''
    #Tracer chaque courbe séparément
    plt.figure(figsize=(10, 8))
    for i, var_name in enumerate([r"$C$", r"$C_{ER}$", r"$P$", r"$\rho_{CRAC}$", r"$g_{IP3R}$", r"$h_{IP3R}$", r"$g_{PMCA}$"]):
        plt.subplot(4, 2, i+1)
        plt.plot(t, z[i])
        plt.xlabel('t')
        plt.title(var_name)
    
    plt.tight_layout()
    plt.show()
    '''
    
    #représentation des courbes de l'article
    plt.figure(figsize=(12, 8))
    plt.plot(t, z[0],'k')
    plt.xlabel('Temps [s]')
    plt.ylabel("Calcium [nM]")
    plt.title("Calcium sans les canaux CRAC")
    plt.legend([r"$C$"])
    plt.show()
    
    plt.figure(figsize=(12, 8))
    plt.plot(t, z[3],'k')
    plt.xlabel('Temps [s]')
    plt.ylabel("Densité des canaux CRAC actifs [#/dm^2]")
    plt.legend([r"$\rho_{CRAC}$"])
    plt.show()
    
    #figure 3-B
    
    #Graphe courants
    
    plt.figure(figsize=(12, 8))
    plt.plot(np.linspace(0, T, len(calc_sim.I_PMCA_values)), calc_sim.I_PMCA_values,'k:')
    plt.plot(np.linspace(0, T, len(calc_sim.I_CRAC_values)), calc_sim.I_CRAC_values,'k')
    plt.plot(np.linspace(0, T, len(calc_sim.I_SERCA_values)), calc_sim.I_SERCA_values,'r:')
    plt.plot(np.linspace(0, T, len(calc_sim.I_IP3R_values)), calc_sim.I_IP3R_values,'r')
    plt.plot(np.linspace(0, T, len(calc_sim.net_Ca_to_ER)),calc_sim.net_Ca_to_ER,'--',color='red')
    plt.plot(np.linspace(0, T, len(calc_sim.net_Ca_out)),calc_sim.net_Ca_out,'--',color='black')
    plt.legend([r"$I_{PMCA} (PM)$",r"$I_{CRAC} (PM)$",r"$I_{SERCA} (ER)$",r"$I_{IP3R} (ER)$",r"net Ca to ER",r"net Ca out"])
    plt.xlabel('Temps [s]')
    plt.ylabel("Courrant total de la cellule [A]")
    plt.title('Évolution de nos courants en fonction du temps')
    plt.show()
    
    
    #Figure 5
    plt.figure(figsize=(12, 8))
    plt.semilogy(t,z[1],'r--')
    plt.plot(t,1000*z[2],'b')
    plt.plot(t,1000*z[0],'k')
    plt.xlabel('Temps [s]')
    plt.ylabel('Concentration [nM]')
    plt.legend([r"$Ca_{ER}$", r"$1000IP_3$", r"$1000Ca$"])
    plt.show()

    

if __name__ == "__main__":
    main()   
