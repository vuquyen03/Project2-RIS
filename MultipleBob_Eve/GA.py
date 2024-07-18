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

def generate_random_theta():
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

class Individual:
    def __init__(self):
        self.theta = theta_init.copy()
        self.w = w_init.copy() 
        
def evaluate_population(population):
    fitness = []
    for individual in population:
        theta, w = individual.theta, individual.w
        fitness.append(secrecy_rate_objective_function(theta, w))
    # print("Fitness:", fitness)
    return fitness

def select_parents(population, fitness):
    combined = list(zip(fitness, population))
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
    sorted_population = [x[1] for x in sorted_combined]
    return sorted_population[:2]

def crossover(parent1, parent2):
    theta1, w1 = parent1.theta, parent1.w
    theta2, w2 = parent2.theta, parent2.w
    child_theta = (theta1 + theta2) / 2
    child_w = (w1 + w2) / 2
    new_individual = Individual()
    new_individual.theta = child_theta
    new_individual.w = child_w
    return new_individual

def mutate(individual):
    if np.random.rand() < mutation_rate:
        mutation_index = np.random.randint(len(individual.theta))
        individual.theta[mutation_index] = generate_random_theta()[mutation_index]
    if np.random.rand() < mutation_rate:
        mutation_index = np.random.randint(len(individual.w))
        individual.w[mutation_index] = generate_random_beamforming_vectors()[mutation_index]
    return individual
        
def genetic_algorithm():
    population = [Individual() for _ in range(population_size)]
    best_individual = None
    best_fitness = -np.inf
    
    for generation in range(num_generations):
        fitness = evaluate_population(population)
        current_best_fitness = max(fitness)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[np.argmax(fitness)]
        
        parents = select_parents(population, fitness)
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = parents
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
        print(f"Generation {generation + 1}/{num_generations}, Best Fitness: {best_fitness}")

    return best_individual

if __name__ == "__main__":
    
    # System parameters
    sigma = db2pow(-75)                                                                 # noise power
    N = 4                                                                               # number of transmit antennas
    Nris = 32                                                                           # number of RIS elements
    number_of_users = 2                                                                 # number of users
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

    # Generate random theta and w
    theta_init = generate_random_theta()
    w_init = generate_random_beamforming_vectors()
    
    print("Secret Rate:", secrecy_rate_objective_function(theta_init, w_init))
    
    # Genetic Algorithm parameters
    population_size = 30
    num_generations = 1000
    mutation_rate = 0.1

    # Run the genetic algorithm
    best_individual = genetic_algorithm()
    print("Best Individual:", best_individual.theta, best_individual.w)
    print("Best Fitness:", secrecy_rate_objective_function(best_individual.theta, best_individual.w))

