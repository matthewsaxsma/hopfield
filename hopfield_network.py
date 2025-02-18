import numpy as np
from sklearn.metrics import matthews_corrcoef


# sign function to create binary activation vector
def binary_sign(vector):
    vector = np.array(vector, dtype=np.float64)
    vector[vector > 0] = 1
    vector[vector < 0] = -1
    vector[vector == 0] = np.random.choice([-1, 1]) # 50-50 chance for a 0 to be -1 or +1 
    
    if np.any(np.isnan(vector)):
        raise ValueError("Warning: NaN values detected in binary activation vector")
    return vector

# define Hopfield network class

class HopfieldNetwork:
    def __init__(self, N): # specify the size of the network to be created
        self.N = N
        self.eta = 0.01
        # random vector upon class initialization
        self.current_activity = np.random.choice([-1, 1], size=N) # randomly intialize the activity vector
        self.N = len(self.current_activity)
        self.W = np.zeros((N, N)) # intial zero weights
        self.W = np.array(self.W, dtype=np.float64)

    # method for getting network activation after exposed to a pattern
    def stabilizing_activity(self, start, cycles=40, E=0.15, D=0.15,output=False):
        start = np.array(start, dtype=np.float64)
        for _ in range(cycles):
            activity_change_vector = np.zeros(self.N, dtype=np.float64) # initialize change (delta) vector
            for i in range(self.N): # this loop fills change vector within each run of cycle
                net_input_vector = np.dot(self.W, start)
                if net_input_vector[i] > 0: # update for positive net input
                    activity_change_vector[i] = E * net_input_vector[i] * (1 - start[i]) - D * start[i]
                elif net_input_vector[i] <= 0: # update for negative net input
                    activity_change_vector[i] = E * net_input_vector[i] * (1 - start[i]) - D * start[i]
            start = binary_sign(start + activity_change_vector) # update to activation after one cycle
        stable_output_activation = start
        if np.any(np.isinf(stable_output_activation)):
            print(f"Warning: Inf values detected in activation vector : {stable_output_activation}")
        self.current_activity = stable_output_activation
        if output: return self.current_activity

    # method for updating weight matrix according to Hebbian learning rule
    def update_weight_matrix(self, activity_vector):
        activity_vector = np.array(activity_vector, dtype=np.float64)
        change_weight_matrix = self.eta * np.outer(activity_vector, activity_vector) # simple Hebbian learning rule
        updated_W = self.W + change_weight_matrix # now just automatically calls the current weights
        np.fill_diagonal(updated_W, 0)
        max_weight = np.max(np.abs(self.W))
        if max_weight > 0:  # Avoid division by zero
            self.W /= max_weight
        # a check for values
        if np.any(np.isnan(updated_W)) or np.any(np.isinf(updated_W)):
            print("Warning: NaN or Inf values detected in updated_W")
        self.W = updated_W

N = 100
derivation_correlations = []
profile_correlations = []


for simulations in range(10):
    latent_profile = np.random.choice([-1, 1], size=N)
    pattern_matrix = np.zeros((50, N)) # 50 patterns row, N node columns
    # Creating the patterns
    for i in range(np.shape(pattern_matrix)[0]):
        pattern_matrix[i, :] = np.array(latent_profile * np.random.choice([-1, 1], size=N, p=[0.3, 0.7]))

    hopfield1 = HopfieldNetwork(N=N)

    # training the network
    for row in range(pattern_matrix.shape[0]):
        hopfield1.current_activity = binary_sign(hopfield1.current_activity)
        hopfield1.stabilizing_activity(start=pattern_matrix[row, :]) # let network settle
        hopfield1.update_weight_matrix(activity_vector=hopfield1.current_activity)


    # testing the network on derivations
    for row, _ in enumerate(pattern_matrix):
        network_response = hopfield1.stabilizing_activity(start=pattern_matrix[row, :], output=True)
        # if network_response is None:
        #     raise ValueError("NETWORK RESPONSE IS NONE")
        derivation_correlations.append(round(matthews_corrcoef(network_response, pattern_matrix[row,:]),6))

    # testing the network on latent profiles
    network_response = binary_sign(hopfield1.stabilizing_activity(start=latent_profile, output=True))
    profile_correlations.append(round(matthews_corrcoef(network_response, latent_profile),6))

np.mean(derivation_correlations), np.mean(profile_correlations)