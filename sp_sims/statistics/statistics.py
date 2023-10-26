import numpy as np
from abc import ABC,abstractmethod

# TODO: This document ought to give usw ways to going through the history 
# and being able to summarize certain statistics. Such as waiting times and 
# Recurrence Times(Avg Time to go from one state to another)
# Queueing Size

# This guy right here receives realized sequence of i.i.d. r.v.  
# Then creates statistics on this allone(till I find it impossible)
class AbsStatisticGenerator(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod # Will generate our statistics
    def generate(self):
        pass
    @abstractmethod
    def show_state_dist(self) :
        pass

# I regret the previous name so this is a quick bandaid 
def trans_matrix(states):
    return state_transitions(states)

# I changed the funciton to not use times as these matches the MLE.
def state_transitions(states):
    # Unique_Elements might seem redundant but maybe we wont always have a state space that starts at 0
    unique_elements,inverse, counts, = np.unique(
            states,
            axis=0,
            return_counts=True,
            return_inverse=True)
    
    no_unique_states_visited = len(unique_elements)
    transition_matrix = np.zeros((no_unique_states_visited,no_unique_states_visited))

    # Get State Pairs TODO: This only works when we have initial state = 0
    for itx in range(len(inverse)-1):
        i = inverse[itx]
        j = inverse[itx+1]
        #  transition_matrix[i,j] += times[itx]
        transition_matrix[i,j] += 1

    # Normalize
    row_sums = transition_matrix.sum(axis=1)
    
    return transition_matrix / row_sums[:,np.newaxis]


def eventd_stationary_state(state_tape,holding_t_tapes):
    # Still assuming tape matrix
    unique_elements = np.unique(state_tape)
    counter = []

    for elem in unique_elements:
        index_list = (state_tape == elem)
        counter.append(np.sum(holding_t_tapes[index_list]))
    return unique_elements, counter

def __init__(self, tapes, dt):
    # We will receive a list of list
    self.transition_times = tapes['holding_times']
    self.states = tapes['states']
    self.delta_t = dt

# Tapes will be a column vectors
def sampd_stationary_state(sampling_rate,state_tapes,holding_t_tapes):
    # We just have to get a certain percentage of the states
    # Create array of unique elements
    sampling_time = 1/sampling_rate
    if len(state_tapes) != len(holding_t_tapes):
        print("Sizes of tapes are not equivalent. Please make sure they \
                correspond to each other.")
        return
    # Need to find final time
    # Final Exponential Time only tells us  about
    transition_times = np.cumsum(holding_t_tapes)
    starting_times = np.copy(transition_times) - holding_t_tapes
    ending_time = transition_times[-1]

    # Need to generate sampling arrays
    #  sampling_array = np.arange(0,ending_time,step=sampling_rate)

    # Unique Elements
    unique, return_inverse = np.unique(state_tapes,return_inverse=True)
    # Unique has ordering. It can give us a bias to our array 
    # bias is simply the distance from 0 to first  element
    bias = unique[0]
    counter = np.zeros_like(unique) # Will keep counters for state

    # Get Time at which first sample on each interval happens
    first_sample_per_interval = starting_times+(starting_times%sampling_time)
    indices  = first_sample_per_interval<transition_times
    sampling_amounts_per_state = (transition_times-first_sample_per_interval)/sampling_time
    sampling_amounts_per_state[sampling_amounts_per_state<0] = 0
    sampling_amounts_per_state[indices] += 1
    
    # At this point we get how many counts we've got per interval 
    for i,state in enumerate(unique): counter[i] = np.sum(sampling_amounts_per_state[return_inverse == state])
    
    return unique,counter

def emp_steady_state_distribution(state_tape):
    unique, count = np.unique(state_tape, return_counts=True)
    total_count = np.sum(count)
    return count/total_count, unique

    

# Tapes will be a column vectors
def simple_sample(sampling_rate,state_tapes,holding_t_tape, max_samples=None):
    # We just have to get a certain percentage of the states
    # Create array of unique elements
    sampling_time = 1/sampling_rate
    if len(state_tapes) != len(holding_t_tape):
        print("Sizes of tapes are not equivalent. Please make sure they \
                correspond to each other.")
        return
    # Need to find final time
    # Final Exponential Time only tells us  about
    transition_times = np.cumsum(holding_t_tape)
    starting_times = np.copy(transition_times) - holding_t_tape

    # Get Sample Tape
    states = []
    state_tapo = np.asarray(state_tapes)
    no_samples = 1
    for i,time in enumerate(np.arange(0,transition_times[-1],sampling_time)):
        # Get Index of State
        state_index  = (time >= starting_times) & (time < transition_times)
        state_fallen_into = state_tapo[state_index]
        assert len(state_fallen_into) == 1
        #  assert (len(state_fallen_into) != 1)
        states.append(state_fallen_into[0])
        if max_samples != None:
            if no_samples >= max_samples:
                break
            else: 
                no_samples += 1

    return np.asarray(states)
    
def quick_sample(sampling_rate,state_tapes,holding_t_tapes):
    sampling_time = 1/sampling_rate
    if len(state_tapes) != len(holding_t_tapes):
        print("Sizes of tapes are not equivalent. Please make sure they \
                correspond to each other.")
        return
    
    transition_times = np.cumsum(holding_t_tapes)

    states = [state_tapes[0]] # Place the initial state at t=0
    state_tapo = np.asarray(state_tapes)
    currT = 0
    currStatIdx = 0
    
    while currT < transition_times[-1]:
        #  print(str(currT) + ' out of ' + str(transition_times[-1]))
        if currStatIdx >= len(transition_times):
            break
        idxIncrement = 1
        while currT + sampling_time > transition_times[currStatIdx + idxIncrement - 1]:
            # This is for the case whereby we skipped some of the transition because it is too short
            if currStatIdx + idxIncrement >= len(transition_times):
                break
            idxIncrement += 1
        
        NoOfReplica = int(np.floor((transition_times[currStatIdx + idxIncrement - 1] - currT) / sampling_time))
        states = states + ([state_tapo[currStatIdx + idxIncrement - 1]] * NoOfReplica)
        currT = currT + (NoOfReplica * sampling_time)
        currStatIdx = currStatIdx + idxIncrement
        #  print(currStatIdx)

    return np.asarray(states)

