import numpy as np
import sympy as ym
import sympy.printing as printing
import torch


def event_driven_mle(state_tape, holdTimes_tape):
    (
        unique_elements,
        inverse,
        counts,
    ) = np.unique(state_tape, axis=0, return_counts=True, return_inverse=True)

    no_unique_states_visited = len(unique_elements)
    gen_matrix = np.zeros((no_unique_states_visited, no_unique_states_visited))
    tot_hold_time = np.zeros((no_unique_states_visited, 1))
    # Diagonal Matrix -> PSD(?) -> Nice properties we can analyize

    # Get State Pairs WARN: This only works when we have initial state = 0
    for itx in range(len(inverse) - 1):
        i = inverse[itx]
        j = inverse[itx + 1]
        gen_matrix[i, j] += 1
        tot_hold_time[i] += holdTimes_tape[itx]
    gen_matrix = gen_matrix / np.repeat(tot_hold_time, no_unique_states_visited, axis=1)

    return gen_matrix


def power_series_exp(Q, power=512):
    assert Q.shape[0] == Q.shape[1]
    # Let us first get the norm of Q for sanity reasons
    # print("Q's Frob Norm is ",np.linalg.norm(Q,ord='fro'))

    # Test for convergence
    final_mat = np.zeros_like(Q)
    for k in range(0, power):
        powered_matrix = np.power(Q, k)
        cur_mat = (1 / np.math.factorial(k)) * powered_matrix
        final_mat += cur_mat
    return final_mat


def power_series_log(mat, power):
    assert mat.shape[0] == mat.shape[1]

    # Test for convergence
    # print("||B-I||=", np.linalg.norm(mat - torch.eye(mat.shape[0]), ord="fro"))
    final_mat = torch.zeros_like(mat)
    for k in range(1, power):
        cur_mat = (-1) ** (k + 1) * (1 / k) * (mat - torch.eye(mat.shape[0]))
        final_mat += cur_mat
    return final_mat


def frequency_matrix(tape, num_states):
    arr = torch.zeros((num_states, num_states))
    for i in range(len(tape) - 1):
        arr[tape[i], tape[i + 1]] += 1
    tots_trans = torch.sum(arr)
    arr /= tots_trans
    return arr


# def power_series_log(mat,power):
# assert mat.shape[0] == mat.shape[1]

# # Test for convergence
# print('||B-I||={}'.format(np.linalg.norm(mat-np.eye(mat.shape[0]),ord='fro')))
# final_mat = np.zeros_like(mat)
# for k in range(1,power):
# cur_mat = (-1)**(k+1) * (1/k) *(mat-np.eye(mat.shape[0]))
# final_mat += cur_mat

# return final_mat


def viterbi(obs_tape, inital_hs_probs, trans_probs, emission_probs):
    # Two Hidden States
    tape_length = len(obs_tape)
    obs = obs_tape

    # Emission Probs must be KXN where K is num of hidden states and N is num of emitted states
    num_hidden_states = emission_probs.shape[0]

    T1 = np.ndarray(num_hidden_states, tape_length, dtype=np.float16)
    T2 = np.ndarray(num_hidden_states, tape_length, dtype=np.float16)

    # Fill them
    for i in range(num_hidden_states):
        T1[i, 0] = inital_hs_probs[i] * emission_probs[i, obs[i]]
        T2[i, 0] = 0

    # Now onto the sequence
    for j in range(1, tape_length):
        for i in range(num_hidden_states):
            T1[i, j] = max(T1[:, j - 1] * trans_probs[:, i] * emission_probs[i, obs[j]])
            T2[i, j] = np.argmax(
                T1[:, j - 1] * trans_probs[:, i] * emission_probs[i, obs[j]]
            )

    best_last_hs = []
    Zt = np.argmax(T1[:, -1])
    for o in range(len(statess) - 1, -1, -1):
        best_path.insert(0, best_last_hs)
        best_last_hs = T2[best_last_hs, o]

    return best_path
