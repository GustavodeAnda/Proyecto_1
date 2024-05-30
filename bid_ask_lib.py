import numpy as np
from scipy.integrate import quad

# Función distribución Weisell
def weibull_pdf(x, lambda_, k):
    return (k / lambda_) * (x / lambda_)**(k - 1) * np.exp(-(x / lambda_)**k)

# Funciones de probabilidad de liquidez
# Bid
def pi_LB(S):
    return max(0, min(0.5, 0.5 - 0.08 * (S)))

#Ask
def pi_LS(S):
    return max(0, min(0.5, 0.5 - 0.08 * (S)))

# Función para maximizar la utilidad esperada U usando el modelo de Copeland & Galai
def utility_function(params, S0, lambda_, k):
    B, A = params
    P_e = S0
    
    # Ganancia
    G = pi_LS(P_e - B) * (P_e - B) + pi_LB(A - P_e) * (A - P_e)
    
    # Pérdida esperada (integrales)
    loss_LS, _ = quad(lambda P: (B - P) * weibull_pdf(P, lambda_, k), 0, B)
    loss_LB, _ = quad(lambda P: (P - A) * weibull_pdf(P, lambda_, k), A, np.inf)
    
    L = pi_LS(B) * loss_LS + pi_LB(A) * loss_LB
    
    # Utilidad esperada
    U = G - L
    
    # Queremos maximizar U, así que devolvemos -U para minimizar
    return -U