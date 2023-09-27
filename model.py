import torch

class ML_Actor(object):
    
    def __init__(self,init_a0=0,init_a1=0,learn_step_size=0.01,gamma=0.85,inf_cutoff=10):
        self.a0 = init_a0
        self.a1 = init_a1
        self.gamma = gamma # Dont like this, but I'll leave it for the sake of convenience
        self.episode_counter = 0
        self.inft_cutoff = inf_cutoff
    
        print("Learning with step size : {}".format(learn_step_size))
        self.learn_step_size = learn_step_size
    
    def generate_episodes(self,steps,initial_state=0):
        # We generate the true values of the ideal distribution
        actions = np.ones(shape=(steps))
        perturbations = np.random.randint(0,2,size=(steps))
        for i,per in enumerate(perturbations):
            perturbations[i] = -1 if per == 0 else 1
        
        states = [initial_state]
        for i,action in enumerate(actions):
            last_state = states[-1]
            states.append(last_state + action + perturbations[i])
        
        return np.array(actions),np.array(states[:-1])


"""
Reconstruction Model will simply create a reconstruction of the larger function points
through a single deep net.
"""
class ReconstructionModel(nn.Module):
    def __init__(sampled_points, reconstructed_points, hidden_state_size):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(sampled_points,hidden_state_size), 
                nn.ReLU(),
                nn.Linear(hidden_state_size,hidden_state_size),
                nn.ReLU(),
                nn.Linear(hidden_state_size,hidden_state_size),
                nn.ReLU(),
                nn.Linear(hidden_state_size,reconstructed_points)
                )
    def forward(self,signal):
        return self.net(signal)


"""
ReconstructionCNN will do same but with a convolution procedure that would ideally
learn the sinc function.
"""
class ReconstructionCNN(nn.Module):
    def __init__(sampled_points, reconstructed_points, hidden_state_size):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(sampled_points,hidden_state_size), 
                nn.ReLU(),
                nn.Linear(hidden_state_size,hidden_state_size),
                nn.ReLU(),
                nn.Linear(hidden_state_size,hidden_state_size),
                nn.ReLU(),
                nn.Linear(hidden_state_size,reconstructed_points)
                )
    def forward(self,signal):
        return self.net(signal)
