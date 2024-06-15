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
    y = np.sqrt(0.5*pl)*(np.random.randn(dim1,dim2) + 1j*np.random.randn(dim1,dim2))  # Rayleigh distribution
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
    theta = np.exp(1j * np.angle(theta))  # Normalize theta to the unit circle
    return theta

def generateChannel():
    normFact = 1/np.sqrt(sigma)
    Hai = chanGen(zetaAI, dAI, Nris, N)                                                         # Alice to RIS channel
    hib = [normFact*chanGen(zetaIB, dIB[i], 1, Nris) for i in range(number_of_users)]           # Channel between the RIS and the legitimate receivers
    hie = [normFact*chanGen(zetaIE, dIE[i], 1, Nris) for i in range(number_of_eavesdroppers)]   # Channel between the RIS and the eavesdroppers
    hab = [normFact*chanGen(zetaAB, dAB[i], 1, N) for i in range(number_of_users)]              # Channel between Alice and the legitimate receivers
    hae = [normFact*chanGen(zetaAE, dAE[i], 1, N) for i in range(number_of_eavesdroppers)]      # Channel between Alice and the eavesdroppers
    return Hai, hib, hie, hab, hae

def secrecy_rate_objective_function(theta, w):
    secrecy_rate = []
    for k in range(number_of_users):
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
    # print("Sum Rate:", sum(secrecy_rate))
    # print("Secrecy Rate:", secrecy_rate)
    return sum(secrecy_rate)

class Particle:
    def __init__(self):
        self.theta = theta_init.copy()
        self.w = w_init.copy()
        self.pbest_theta = self.theta.copy()
        self.pbest_w = self.w.copy()
        self.pbest_value = secrecy_rate_objective_function(self.theta, self.w)
        self.velocity_theta = generate_random_theta(number_of_users, Nris)
        self.velocity_w = generate_random_beamforming_vectors()

    def update_velocity_theta(self, gbest_theta, inertia =0.5, c1=1.0, c2=1.0, user_k=0):
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive_velocity_theta = c1 * r1 * (self.pbest_theta[user_k] - self.theta[user_k])
        social_velocity_theta = c2 * r2 * (gbest_theta[user_k] - self.theta[user_k])
        self.velocity_theta[user_k] = inertia * self.velocity_theta[user_k] + cognitive_velocity_theta + social_velocity_theta

    def update_velocity_w(self, gbest_w, inertia = 0.5, c1=1.0, c2=1.0, user_k=0):
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive_velocity_w = c1 * r1 * (self.pbest_w[user_k] - self.w[user_k])
        social_velocity_w = c2 * r2 * (gbest_w[user_k] - self.w[user_k])
        self.velocity_w[user_k] = inertia * self.velocity_w[user_k] + cognitive_velocity_w + social_velocity_w
    
    def update_position_theta(self, user_k=0):
        self.theta[user_k] += self.velocity_theta[user_k]
        self.theta[user_k] = np.exp(1j * np.angle(self.theta[user_k]))
    
    def update_position_w(self):
        self.w += self.velocity_w
        self.w = self.w / np.linalg.norm(self.w, axis=1, keepdims=True)
    
def PSO_optimize_theta(w, max_iter=100):
    particles = [Particle() for _ in range(number_of_users)]
    gbest_theta = particles[0].theta.copy()
    gbest_value = particles[0].pbest_value.copy()
    # print("gbest_value PSO_theta:", gbest_value)
    
    for iteration in range(max_iter):
        for k in range(number_of_users):
            particles[k].update_velocity_theta(gbest_theta, user_k=k)
            particles[k].update_position_theta(user_k=k)
                        
            fitness_value = secrecy_rate_objective_function(particles[k].theta, w)
            
            if fitness_value > particles[k].pbest_value:
                particles[k].pbest_value = fitness_value
                particles[k].pbest_theta = particles[k].theta.copy()
            
            if fitness_value > gbest_value:
                gbest_value = fitness_value
                gbest_theta = particles[k].theta.copy()
        # print(f"Iteration {iteration+1}/{max_iter}, Global Best Value: {gbest_value}\n")
    return gbest_theta

def PSO_optimize_w(theta, max_iter=100):
    particles = [Particle() for _ in range(number_of_users)]
    gbest_w = particles[0].w.copy()
    gbest_value = particles[0].pbest_value.copy()
    # print("gbest_value PSO_w:", gbest_value)
    
    for iteration in range(max_iter):
        for k in range(number_of_users):
            particles[k].update_velocity_w(gbest_w, user_k=k)
            particles[k].update_position_w()
            
            fitness_value = secrecy_rate_objective_function(theta, particles[k].w)
            
            if fitness_value > particles[k].pbest_value:
                particles[k].pbest_value = fitness_value
                particles[k].pbest_w = particles[k].w.copy()
            
            if fitness_value > gbest_value:
                gbest_value = fitness_value
                gbest_w = particles[k].w.copy()
        # print(f"Iteration {iteration+1}/{max_iter}, Global Best Value: {gbest_value}\n")
    return gbest_w

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

    total_iter = 50
    num_cycles = 50

    # Channel generation
    Hai, hib, hie, hab, hae = generateChannel()

    # Generate random theta and w
    theta_init = generate_random_theta(number_of_users, Nris)
    w_init = generate_random_beamforming_vectors()
    
    print("distance between Alice and the receivers: ", dAB)
    print("distance between Alice and the eavesdroppers: ", dAE)
    print("Secret Rate:", secrecy_rate_objective_function(theta_init, w_init))

    # PSO optimization
    particles = [Particle() for _ in range(number_of_users)]
    for cycle in range(num_cycles):
        theta_opt = PSO_optimize_theta(w_init, max_iter=total_iter)
        theta_init = theta_opt
        w_opt = PSO_optimize_w(theta_opt, max_iter=total_iter)
        w_init = w_opt
        print(f"Cycle {cycle+1}/{num_cycles}, Secret Rate: {secrecy_rate_objective_function(theta_opt, w_opt)}")

    print("Optimized theta:", theta_opt)
    print("Optimized w:", w_opt)

    # Calculate and print the optimal secrecy rate
    optimal_secrecy_rate = secrecy_rate_objective_function(theta_opt, w_opt)
    print("Optimal Secrecy Rate:", optimal_secrecy_rate)
    