import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# Function to transfer from dBW to W (power)
def db2pow(db):
    return 10**(db/10)

# Function to transfer from W to dBW (power)
def pow2db(pow):
    return 10*np.log10(pow)

# Hermitian transpose of a matrix
def HermTranspose(x):
    return x.conj().T

def chanGen(zeta, d, dim1, dim2):
    """Function to generate Rayleigh fading channel coefficients

    Args:
        zeta: ξ is the path loss exponent
        d: the distance between the transmitter and the receiver
        dim1: the number of rows in the channel matrix
        dim2: the number of columns in the channel matrix
    """
    pl_ref = -30                                    # pathloss (dBW) at reference distance
    pl = db2pow(pl_ref - 10*zeta*np.log10(d))       # pathloss model at distance d
    y = np.sqrt(0.5*pl)*(np.random.randn(dim1,dim2)\
        + 1j*np.random.randn(dim1,dim2))            # Rayleigh distribution
    return y

def generate_random_beamforming_vectors():
    
    # Generate random complex numbers for each element of the beamforming vectors
    realPart = np.random.randn(number_of_users, N)
    imagPart = np.random.randn(number_of_users, N)
    beamforming_vectors = realPart + 1j * imagPart
    
    # Normalize the vectors
    beamforming_vectors = beamforming_vectors / np.linalg.norm(beamforming_vectors, axis=1, keepdims=True)
    return beamforming_vectors

def generate_random_theta(number_of_users, Nris):
    realPart = np.random.randn(number_of_users, Nris)
    imagPart = np.random.randn(number_of_users, Nris)
    theta = realPart + 1j * imagPart
    theta = np.exp(1j * np.angle(theta))  # Chuẩn hóa theta về vòng tròn đơn vị
    return theta

def generateChannel():
    normFact = 1/np.sqrt(sigma)
    Hai = chanGen(zetaAI, dAI, Nris, N)                                                         # Alice to RIS channel
    hib = [normFact*chanGen(zetaIB, dIB[i], 1, Nris) for i in range(number_of_users)]           # Channel between the RIS and the legitimate receivers
    hie = [normFact*chanGen(zetaIE, dIE[i], 1, Nris) for i in range(number_of_eavesdroppers)]   # Channel between the RIS and the eavesdroppers
    hab = [normFact*chanGen(zetaAB, dAB[i], 1, N) for i in range(number_of_users)]              # Channel between Alice and the legitimate receivers
    hae = [normFact*chanGen(zetaAE, dAE[i], 1, N) for i in range(number_of_eavesdroppers)]      # Channel between Alice and the eavesdroppers
    return Hai, hib, hie, hab, hae

# Objective function to maximize the secrecy rate
def secrecy_rate_objective_function(theta, w):
    secrecy_rate = []
    for k in range(number_of_users):
        # List to store all value of secrecy rate of user k
        R_bk = []
        
        # Legitimate user k
        numGamma_bk = np.abs(np.dot(hib[k] @ np.diag(theta[k]) @ Hai + hab[k], w[k]))**2
        denGamma_bk = 1 + np.sum([np.abs(np.dot(hib[k] @ np.diag(theta[k]) @ Hai + hab[k], w[i]))**2 for i in range(number_of_users) if i != k])
        gamma_bk = numGamma_bk/denGamma_bk
        C_bk = np.log2(1 + gamma_bk)
        
        for i in range(number_of_eavesdroppers):
            # Eavesdropper i
            numGamma_ei = np.abs(np.dot(hie[i] @ np.diag(theta[k]) @ Hai + hae[i], w[k]))**2
            denGamma_ei = 1 + np.sum([np.abs(np.dot(hie[i] @ np.diag(theta[k]) @ Hai + hae[i], w[j]))**2 for j in range(number_of_users) if j != k])
            gamma_ei = numGamma_ei/denGamma_ei
            C_ei = np.log2(1 + gamma_ei)
            R_bk.append(C_bk - C_ei)
        
        secrecy_rate.append(max(min(R_bk),0))
    
    # print(secrecy_rate)
    return sum(secrecy_rate)

def compute_gradient_w(theta, w):
    grad_w = np.zeros_like(w, dtype=complex)
    
    for k in range (number_of_users):
        Z_k = hib[k] @ np.diag(theta[k]) @ Hai + hab[k]
        numGamma_bk = np.abs(np.dot(Z_k, w[k]))**2
        denGamma_bk = 1 + np.sum([np.abs(np.dot(Z_k, w[i]))**2 for i in range(number_of_users) if i != k])
        gamma_bk = numGamma_bk/denGamma_bk
        
        grad_gamma_bk = (HermTranspose(Z_k) @ Z_k @ w[k] + HermTranspose(HermTranspose(Z_k) @ Z_k @ w[k])) / denGamma_bk 
        grad_Cbk = (grad_gamma_bk / np.log(2)) / (1 + gamma_bk)
        
        gamma_e = []
        for i in range (number_of_eavesdroppers):
            Z_i = hie[i] @ np.diag(theta[k]) @ Hai + hae[i]
            numGamma_ei = np.abs(np.dot(Z_i, w[k]))**2
            denGamma_ei = 1 + np.sum([np.abs(np.dot(Z_i, w[j]))**2 for j in range(number_of_users) if j != k])
            gamma_ei = numGamma_ei/denGamma_ei
            gamma_e.append(gamma_ei)
        
        index_eve_max = gamma_e.index(max(gamma_e))     # Find the eavesdropper with the highest channel gain 
        Z_e_max = hie[index_eve_max] @ np.diag(theta[k]) @ Hai + hae[index_eve_max]
        numGamma_e_max = np.abs(np.dot(Z_e_max, w[k]))**2
        denGamma_e_max = 1 + np.sum([np.abs(np.dot(Z_e_max, w[l]))**2 for l in range(number_of_users) if l != k])
        gamma_e_max = numGamma_e_max/denGamma_e_max
        
        grad_gamma_e_max = (HermTranspose(Z_e_max) @ Z_e_max @ w[k] + HermTranspose(HermTranspose(Z_e_max) @ Z_e_max @ w[k])) / denGamma_e_max
        grad_Ce_max = (grad_gamma_e_max / np.log(2)) / (1 + gamma_e_max)
        
        grad_w[k] += -(grad_Cbk - grad_Ce_max)

            
        for j in range (number_of_users):
            if (j != k):
                numGamma_wj_Cbk = -numGamma_bk * (HermTranspose(Z_k) @ Z_k @ w[j] + HermTranspose(HermTranspose(Z_k) @ Z_k @ w[j]))
                denGamma_wj_Cbk = np.log(2) * (1 + gamma_bk) * denGamma_bk**2
                grad_wj_Cbk = numGamma_wj_Cbk / denGamma_wj_Cbk
                
                Z_em = hie[index_eve_max] @ np.diag(theta[k]) @ Hai + hae[index_eve_max]
                numGamma_em = np.abs(np.dot(Z_em, w[k]))**2
                denGamma_em = 1 + np.sum([np.abs(np.dot(Z_em, w[l]))**2 for l in range(number_of_users) if l != k])
                numGamma_wj_em = -numGamma_em * (HermTranspose(Z_em) @ Z_em @ w[j] + HermTranspose(HermTranspose(Z_em) @ Z_em @ w[j]))
                denGamma_wj_em = np.log(2) * denGamma_em**2 * (1 + gamma_e_max)
                
                grad_wj_em = numGamma_wj_em / denGamma_wj_em
                grad_w[j] += -(grad_wj_Cbk - grad_wj_em)
        
    return grad_w

def gradient_descent_update_w(theta, w, learning_rate):
    grad_w = compute_gradient_w(theta, w)
    w_new = w - learning_rate * grad_w
    
    # Normalize the vectors
    w_new = w_new / np.linalg.norm(w_new, axis=1, keepdims=True)
    return w_new

def gradient_descent_w(theta, initial_w, learning_rate=0.005, total_iter=100, epsilon=1e-3):
    """Main Gradient Descent Algorithm

    Args:
        theta: the phase shift matrix
        initial_w: beamforming vectors
        learning_rate: learning rate of the gradient descent algorithm. Defaults to 0.005.
        total_iter: total number of iterations. Defaults to 100.
        epsilon: the stopping criterion. Defaults to 1e-3.

    Returns:
        np.array: the optimized beamforming vectors
    """
    
    best_w = initial_w.copy()
    best_value = -secrecy_rate_objective_function(theta, best_w)
    
    current_w = initial_w.copy()
    previous_value = best_value
    
    for iteration in range(total_iter):
        current_w = gradient_descent_update_w(theta, current_w, learning_rate)
        current_value = -secrecy_rate_objective_function(theta, current_w)
        
        # print("Current Value:", current_value, "Previous Value:", previous_value)
        
        # Check for convergence
        if abs(current_value - previous_value) < epsilon:
            print("Stopped at iteration", iteration+1)
            print(epsilon, current_value[0] - previous_value[0])
            break
                
        if (current_value < best_value):
            best_w = current_w
            best_value = current_value
            
        # Update the previous value
        previous_value = current_value
        
    return best_w

def compute_gradient_theta(theta, w, epsilon=1e-3):
    grad_theta = np.zeros_like(theta, dtype=complex)
    for k in range(number_of_users):
        for i in range(Nris):
            theta_plus = theta.copy()
            theta_plus[k][i] += epsilon*1j
            theta_minus = theta.copy()
            theta_minus[k][i] -= epsilon*1j
            grad_theta[k][i] = (secrecy_rate_objective_function(theta_plus, w) - secrecy_rate_objective_function(theta_minus, w))/(2*epsilon)
            
    return grad_theta

def gradient_descent_update_theta(theta, w, learning_rate):
    grad_theta = compute_gradient_theta(theta, w)
    theta_new = theta - learning_rate * grad_theta
    
    # Normalize the vectors
    theta_new = np.exp(1j * np.angle(theta_new))  # Chuẩn hóa theta về vòng tròn đơn vị
    return theta_new

def gradient_descent_theta(initial_theta, w, learning_rate=0.005, total_iter=100, epsilon=1e-3):
    """Main Gradient Descent Algorithm

    Args:
        initial_theta: initial phase shift matrix
        w: beamforming vectors
        learning_rate: learning rate of the gradient descent algorithm. Defaults to 0.005.
        total_iter: total number of iterations. Defaults to 100.
        epsilon: the stopping criterion. Defaults to 1e-3.

    Returns:
        np.array: the optimized phase shift matrix
    """
    
    best_theta = initial_theta.copy()
    best_value = -secrecy_rate_objective_function(best_theta, w)
    
    current_theta = initial_theta.copy()
    previous_value = best_value
    
    for iteration in range(total_iter):
        current_theta = gradient_descent_update_theta(current_theta, w, learning_rate)
        current_value = -secrecy_rate_objective_function(current_theta, w)
        
        # print("Current Value:", current_value, "Previous Value:", previous_value)
        
        # Check for convergence
        if abs(current_value - previous_value) < epsilon:
            print("Stopped at iteration", iteration+1)
            print(epsilon, current_value[0] - previous_value[0])
            break
        
        if (current_value < best_value):
            best_theta = current_theta
            best_value = current_value
            
        # Update the previous value
        previous_value = current_value
        
    return best_theta


if __name__ == "__main__":
    
    # System parameters
    sigma = db2pow(-75)                                                                 # noise power
    N = 4                                                                               # number of transmit antennas
    Nris = 32                                                                           # number of RIS elements
    number_of_users = 4                                                                 # number of users
    number_of_eavesdroppers = 2                                                         # number of eavesdroppers
    zetaAI = 2.2                                                                        # Path loss exponent of the channel between the Alice and the RIS
    zetaIB = 2.5                                                                        # Path loss exponent of the channel between the legitimate receivers and the RIS
    zetaIE = 2.5                                                                        # Path loss exponent of the channel between the eavesdroppers and the RIS
    zetaAB = 3.5                                                                        # Path loss exponent of the channel between the Alice and the legitimate receivers
    zetaAE = 3.5                                                                        # Path loss exponent of the channel between the Alice and the eavesdroppers

    dAI = 50                                                                            # distance between Alice and the RIS
    dv = 2                                                                              # Vertical distance between the Alice and the Eve and Bob
    dABh = np.random.uniform(5, 10, size=number_of_users)                               # Horizontal distance between Alice and the legitimate receivers
    dAEh = np.random.uniform(50, 150, size=number_of_eavesdroppers)                     # Horizontal distance between Alice and the eavesdroppers
    dAB = [np.sqrt(dABh[i]**2 + dv**2) for i in range(number_of_users)]                 # Distance between Alice and the legitimate receivers
    dAE = [np.sqrt(dAEh[i]**2 + dv**2) for i in range(number_of_eavesdroppers)]         # Distance between Alice and the eavesdroppers
    dIB = [np.sqrt((dABh[i]-dAI)**2 + dv**2) for i in range(number_of_users)]           # Distance between the legitimate receivers and the RIS
    dIE = [np.sqrt((dAEh[i]-dAI)**2 + dv**2) for i in range(number_of_eavesdroppers)]   # Distance between the eavesdroppers and the RIS

    # Channel generation
    Hai, hib, hie, hab, hae = generateChannel()
    
    # Generate random initial values for theta and w
    theta_init = generate_random_theta(number_of_users, Nris)
    w_init = generate_random_beamforming_vectors()
    
    print("distance between Alice and the receivers: ", dAB)
    print("distance between Alice and the eavesdroppers: ", dAE)
    
    current_secrecy_rate = secrecy_rate_objective_function(theta_init, w_init)
    print("Secrecy rate with initial value: ", current_secrecy_rate)
    
    # Gradient Descent Algorithm
    num_cycles = 100
    for i in range(num_cycles):
        w_opt = gradient_descent_w(theta_init, w_init, learning_rate=0.01, total_iter=50, epsilon=1e-3)
        theta_opt = gradient_descent_theta(theta_init, w_opt, learning_rate=0.01, total_iter=50, epsilon=1e-3)
        theta_init = theta_opt
        w_init = w_opt
        
        new_secrecy_rate = secrecy_rate_objective_function(theta_opt, w_opt)
        print("Secrecy rate after cycle", i+1, ":", new_secrecy_rate)
        
        if (new_secrecy_rate - current_secrecy_rate) < 1e-3:
            print("Converged")
            break
        
        current_secrecy_rate = new_secrecy_rate
        