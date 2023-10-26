import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
from sp_sims.simulators.stochasticprocesses import *
from sp_sims.statistics.statistics import *

def argparser(parser: argparse.ArgumentParser):
    parser.add_argument('--length',
                        dest='length',
                        default=10000,
                        type=int,
                        help='Length of episode in discrete realizations.')
    parser.add_argument('--mu',
                        dest='mu',
                        default = .15,
                        type=float,
                        help='Service Rate')
    parser.add_argument('--lambda',
                        dest='lam',
                        default=.10,
                        type=float,
                        help='Birth Rate')
    parser.add_argument('--samprate',
                        dest='samprate',
                        default=1.0,
                        type=float,
                        help='Rate at which we sample real line.')
    parser.add_argument('--init_state',
                        dest='init_state',
                        type=int,
                        default = 0,
                        help='Initial State in the real line.(Amnt of current events)')

def get_stationary(state_tape,holdTimes_tape,samp_rate ):
    #  trans_matrx_samp = state_transitions(np.full_like(state_tape,samprate,dtype=np.float16), state_tape)
    #
    #  initial = np.zeros((1,trans_matx_samp.shape[1]))
    #  initial[0,0] = 1
    #  res = np.copy(trans_matx_samp)
    #  frob_hor_ax = []
    #  frob_norms = []
    #  axs[0].plot(res,label="Power Method n={}".format(1))
    #
    #  for i in range(2,power_val):
    #      res = res@trans_matx_samp
    #      if np.log2(i) % 1 == 0:
    #          frob_norms.append(np.linalg.norm(trans_matx_samp-res,'fro'))
    #          fin = initial@res
    #          axs[0].plot(fin.flatten(),label="Power Method n={}".format(i))
    #
    #  axs[0].hist(state_tape,bins=trans_matx_samp.shape[1],density=True,label="Sampled Histogram")
    #
    #  # Plot the true distributoin
    #  maxx = trans_matx_samp.shape[1]+1
    #  x = np.linspace(1,maxx,100)
    #  # Theres a different expression for n = 0. Will add later
    #  meep = lambda expo: (args.lam/args.mu)**expo
    #  y = meep(x)
    #  y /= 1 + np.sum([meep(i-1) for i in range(1,1000)])
    #  y0 = 1/(1+ np.sum([meep(i-1) for i in range(1,1000)]))
    #  x = np.insert(x,0,0)
    #  y = np.insert(y,0,y0)
    #  axs[0].plot(x,y,label="Closed Form Solution")
    pass

def tri_digmatalgo(N,a,b,c,d,x):
    pass




def frob_comparison(state_tape,holdTimes_tape,samp_rate=1,power_val=64):

    fig,axs = plt.subplots(1,1)

    # Save Event Driven(For Reference)
    event_driven = trans_matrix(holdTimes_tape, state_tape)
    fig, ax = plt.subplots(1,1)
    fig.tight_layout()
    fig.set_size_inches(10,10)
    for i in range(event_driven.shape[0]):
        for j in range(event_driven.shape[1]):
            ax.text(j,i,"{:03.2f} ".format(event_driven[i,j]),ha="center",va="center",color="w")
    ax.imshow(event_driven)
    ax.set_title("Event-Driven Transitions Matrix")
    plt.savefig('./Images/event_driven_matrx.png' ,dpi=150)
    plt.clf()
    plt.close()

    # Get ready for Storing
    event_diff_norms = []
    trans_matx_samp = []
    squared_nth_deg = []
    frob_norms = []

    # Cycle through division of intervals
    for i in range(0,int(np.log2(power_val)+1)):
        pow2  = np.power(2,i)

        sampled_tape = simple_sample(samp_rate*pow2, state_tape, holdTimes_tape)

        trans_matx_samp.append(state_transitions(np.full_like(sampled_tape,1/(samp_rate*pow2),dtype=np.float16), sampled_tape))

        nth_mat = np.linalg.matrix_power(trans_matx_samp[-1],pow2)
        #  nth_mat = np.linalg.matrix_power(trans_matx_samp[-1],i+1)

        frob_norms.append(np.linalg.norm(trans_matx_samp[0]-nth_mat,ord='fro'))
        event_diff_norms.append(np.linalg.norm(event_driven-nth_mat,ord='fro'))

    # Frobeneius Norm
    xticks = [2**i for i in range(int(np.log2(power_val)+1))]

    axs.plot(xticks,frob_norms,label="Norms of diff from rate 1")
    axs.plot(xticks,event_diff_norms,label="Norms of diff from event driven",c='r')
    axs.set_xticks(xticks)
    
    #  axs.set_xscale('log')

    fig.tight_layout()
    fig.set_size_inches(10,10)
    plt.savefig('from_norms'.format(
        args.samprate,args.lam,args.mu,
        format='eps',dpi=200
        ))
    plt.legend()
    plt.show()

def power_matrix(holdTimes_tape, state_tape, powers=16, samp_rate=-1):
    # We will be using event driven distributions for the moemnt being
    
    # Event Driven Empirical Probabilities
    if samp_rate < 0:
        trans_matx_event = state_transitions(holdTimes_tape, state_tape)
    else:
        samped_tape = simple_sample(samp_rate, state_tape, holdTimes_tape)
        trans_matx_event = state_transitions(
                np.full_like(samped_tape,samp_rate),
                samped_tape)

    frob_norm_diff = []
    for z in range(powers):
        fig, ax = plt.subplots(1,1)
        fig.tight_layout()
        fig.set_size_inches(10,10)

        for i in range(trans_matx_event.shape[0]):
            for j in range(trans_matx_event.shape[1]):
                ax.text(j,i,"{:03.2f} ".format(trans_matx_event[i,j]),ha="center",va="center",color="w")
        ax.imshow(trans_matx_event)
        ax.set_title("Transitions Matrix, rate {}, pow {}".format(samp_rate,str(z+1)))
        plt.savefig(
                './Images/transition_matrices_samped_rate{}/trasmat_rate_{}_pow_{:02d}.png'
                .format(int(samp_rate),int(samp_rate),z+1),dpi=150)
        plt.clf()
        plt.close()
        frob_norm_diff.append( np.linalg.norm(trans_matx_event - (trans_matx_event@trans_matx_event)))
        trans_matx_event = trans_matx_event @ trans_matx_event


    fig, ax = plt.subplots(1,1)
    fig.tight_layout()
    fig.set_size_inches(10,10)
    plt.plot(range(len(frob_norm_diff)),frob_norm_diff)
    plt.title('From Norms of powers')
    plt.savefig('./Images/transition_matrices_samped_rate{}/forb_norm.png'.format(int(samp_rate)))
    print('Done with saving images')

def show_trans_matrix(holdTimes_tape, state_tape,samp_rate):

    fig, ax = plt.subplots(1,3)
    
    # Sample the Tapes
    sampled_tape_ori = simple_sample(samp_rate, state_tape, holdTimes_tape)
    sampled_tape_half = simple_sample(samp_rate/2, state_tape, holdTimes_tape)
    
    # Event Driven Empirical Probabilities
    trans_matx_event = state_transitions(holdTimes_tape, state_tape)
    im = ax[0].imshow(trans_matx_event)
    for i in range(trans_matx_event.shape[0]):
        for j in range(trans_matx_event.shape[1]):
            ax[0].text(j,i,"%2.2f " % trans_matx_event[i,j],ha="center",va="center",color="w")
    ax[0].set_title("Event Driven Transitions")

    # Sampled Transition Probabilities
    trans_matx_samp = state_transitions(np.full_like(sampled_tape_ori,args.samprate), sampled_tape_ori)
    im = ax[1].imshow(trans_matx_samp)
    for i in range(trans_matx_samp.shape[0]):
        for j in range(trans_matx_samp.shape[1]):
            ax[1].text(j,i,"%.2f " % trans_matx_samp[i,j],ha="center",va="center",color="w")
    ax[1].set_title("Transitions from sampled states at sr:{}".format(args.samprate))

    # Half the sampling rate
    trans_matx_samp_half = state_transitions(np.full_like(sampled_tape_half,args.samprate/2,dtype=np.float16), sampled_tape_half)
    im = ax[2].imshow(trans_matx_samp_half)
    for i in range(trans_matx_samp_half.shape[0]):
        for j in range(trans_matx_samp_half.shape[1]):
            ax[2].text(j,i,"%.2f " % trans_matx_samp_half[i,j],ha="center",va="center",color="w")
    ax[2].set_title("Transitions from sampled states at sr:{}".format(args.samprate/2))

    fig.tight_layout()
    fig.set_size_inches(50,50)
    plt.savefig('samprate_{}_lam_{}_mu_{}.jpg'.format(
        args.samprate,args.lam,args.mu,
        format='eps',dpi=600
        ))
    plt.show()



def convergence_of_transitionmat(holdTimes_tape, state_tape,samp_rate):

    # We will be using event driven distributions for the moemnt being
    # Event Driven Empirical Probabilities
    # Create Multiple Tapes
    multi_samp_tape = [] 
    print("Working with state tape of size : "+str(len(state_tape)))
    mats = []
    max_no_states = 0
    for k in range(2,int(np.log2(len(state_tape)))+1):
        print("Getting Tape for : {}".format(2**k))
        multi_samp_tape.append(state_tape[:2**k])
        cur_mat = state_transitions(np.full_like(state_tape[:2**k],samp_rate),
                        simple_sample(samp_rate,state_tape[:2**k],np.full_like(state_tape[:2**k],samp_rate)))
        max_no_states = max(max_no_states,cur_mat.shape[0])
        print("Size of array {} is {} ".format(k,cur_mat.shape))
        mats.append(cur_mat)

    # 
    print("Biggest Shape : {}".format(max_no_states))
    print("Information about our new arrays:")
    np_mats = np.zeros((len(mats),max_no_states,max_no_states))

    for i in range(len(mats)): np_mats[i,:,:] = np.pad(mats[i],(
            (0,max_no_states-mats[i].shape[0]), (0,max_no_states-mats[i].shape[1])
        ),'constant',constant_values=0)
    print(np_mats[0,:,:])
    frob_norm_diff = []
    for k,trans_matrx in enumerate(np_mats):
        fig, ax = plt.subplots(1,1)
        fig.tight_layout()
        fig.set_size_inches(10,10)
        ax.imshow(trans_matrx)
        for i in range(trans_matrx.shape[0]):
            for j in range(trans_matrx.shape[1]):
                ax.text(j,i,"%2.2f " % trans_matrx[i,j],ha="center",va="center",color="w")
        ax.set_title("Estimation of Transition at {} samples".format(2**k))
        file_name = './Images/transition_mat_per_samp/trasmat_samps_{:03d}.png'.format(k)
        print('Storing {} ...'.format(file_name))
        plt.savefig(file_name,dpi=150)
        plt.clf()
        if k >0:
            frob_norm_diff.append(np.linalg.norm(trans_matrx-np_mats[k-1]))

    # Save Frob Norm
    fig, ax = plt.subplots(1,1)
    fig.tight_layout()
    fig.set_size_inches(10,10)
    plt.plot(range(len(frob_norm_diff)),frob_norm_diff)
    plt.title("Frob Norm of PathLength Vs Convergence of Sampled TransMatx")
    plt.savefig('./Images/transition_mat_per_samp/frob_norm.png',dpi=150)

    print('Done with saving images')

#  <<<<<<< HEAD
def get_generator_matrix(holdTimes_tape,state_tape,basel_samprate=1, max_doubling_of_rate=12):

    mats = []
    frob_norm_diff = []
    last_trans_matx = np.zeros((0,0))
    longest_state_spaces = 0
    for k in range(0,max_doubling_of_rate+1):
        # Set Figuresj
        fig, ax = plt.subplots(1,1)
        fig.tight_layout()
        fig.set_size_inches(10,10)
        # Sample
        sampled_tape = simple_sample(basel_samprate*2**k, state_tape, holdTimes_tape)
        trans_matrx = state_transitions(np.full_like(sampled_tape,basel_samprate),sampled_tape)
        # Normalize
        trans_matrx = (basel_samprate)*trans_matrx
        mats.append(trans_matrx)
        longest_state_spaces = max(longest_state_spaces,trans_matrx.shape[0])

        # Show/Save Figure
        ax.imshow(trans_matrx)
        for i in range(trans_matrx.shape[0]):
            for j in range(trans_matrx.shape[1]):
                ax.text(j,i,"%2.2f " % trans_matrx[i,j],ha="center",va="center",color="w")
        ax.set_title("Estimation of Transition at {} samples".format(2**k))
        file_name = './Images/gen_matrx_est/trasmat_samps_{:03d}.png'.format(k)
        print('Storing {} ...'.format(file_name))
        plt.savefig(file_name,dpi=150)
        plt.clf()
    
    # Pad the Matrices
    mats = [
            np.pad(
                mat, 
                ((longest_state_spaces-mat.shape[0]),
                 (longest_state_spaces-mat.shape[1])),mode="constant",constant_values=0) for mat in mats]

    # Frob Norm here
    for k,mat in enumerate(mats):
        if k >0:
            frob_norm_diff.append(np.linalg.norm(mat-mat[k-1]))

    plt.plot(range(frob_norm_diff),frob_norm_diff)
    plt.savefig('./Images/gen_matrx_est/forb_norm.png')
    print("Estimated Generator matrices saved")

#  =======
def GeneratorFromTransition(holdTimes_tape, state_tape,samp_rate):

    fig, ax = plt.subplots(2,2)
    font_size = 4
    samp_time = 1/samp_rate
    alt_rate = samp_rate*(2**6)
    alt_samp_time = 1/alt_rate
    
    # Sample the Tapes
    sampled_tape_ori = simple_sample(samp_rate, state_tape, holdTimes_tape)
    sampled_tape_half = simple_sample(alt_rate, state_tape, holdTimes_tape)
    
    # Sampled Transition Probabilities
    trans_matx_samp = state_transitions(np.full_like(sampled_tape_ori,args.samprate), sampled_tape_ori)
    im = ax[0,0].imshow(trans_matx_samp)
    for i in range(trans_matx_samp.shape[0]):
        for j in range(trans_matx_samp.shape[1]):
            ax[0,0].text(j,i,"%.2f " % trans_matx_samp[i,j],ha="center",va="center",color="w",fontsize=font_size)
    ax[0,0].set_title("Transitions from sampled states at sr:{}".format(args.samprate))

    # Half the sampling rate
    trans_matx_samp_half = state_transitions(np.full_like(sampled_tape_half,args.samprate/2,dtype=np.float16), sampled_tape_half)
    im = ax[1,0].imshow(trans_matx_samp_half)
    for i in range(trans_matx_samp_half.shape[0]):
        for j in range(trans_matx_samp_half.shape[1]):
            ax[1,0].text(j,i,"%.2f " % trans_matx_samp_half[i,j],ha="center",va="center",color="w",fontsize=font_size)
    ax[1,0].set_title("Transitions from sampled states at sr:{}".format(args.samprate/2))

    # Now the Generator Matrices
    Q_ori = (trans_matx_samp/samp_time)-np.eye(trans_matx_samp.shape[0],trans_matx_samp.shape[1])/samp_time # + tineh
    Q_halfy = (trans_matx_samp_half/alt_samp_time)-np.eye(trans_matx_samp_half.shape[0],trans_matx_samp_half.shape[1])/alt_samp_time # + tineh
    im = ax[0,1].imshow(Q_ori)
    for i in range(Q_ori.shape[0]):
        for j in range(Q_ori.shape[1]):
            ax[0,1].text(j,i,"%.2f " % Q_ori[i,j],ha="center",va="center",color="w",fontsize=font_size)
    ax[0,1].set_title("Generator Matrix from sampled states at sr:{}".format(args.samprate))

    im = ax[1,1].imshow(Q_halfy)
    for i in range(Q_halfy.shape[0]):
        for j in range(Q_halfy.shape[1]):
            ax[1,1].text(j,i,"%.2f " % Q_halfy[i,j],ha="center",va="center",color="w",fontsize=font_size)
    ax[1,1].set_title("Generator Matrix from sampled states at sr:{}".format(alt_rate))

    fig.tight_layout()
    fig.set_size_inches(10,10)
    plt.savefig('GenMatrices_r{}_l{}_m{}.png'.format(
        args.samprate,args.lam,args.mu,
        format='eps',dpi=300
        ))
    plt.show()
#  >>>>>>> fa445c4908338ead4f456689eb2a17dff18cb43e

if __name__ == '__main__':
    # Create Markov Embedded Simulator
    parser  = argparse.ArgumentParser()
    argparser(parser)
    args = parser.parse_args()

    # Created Tapes
    rates = {"lambda": args.lam,"mu":args.mu} #This should keep us within the corner
    print("Working with rates lambda:{} mu:{}".format(args.lam,args.mu))
    #  embedded_sp = EmbeddedMarkC_BD(args.length,rates)
    #  emb_hold_tape, emb_state_tape = embedded_sp.generate_history(args.init_state)
    roe = RaceOfExponentials(args.length,rates)
    holdTimes_tape, state_tape = roe.generate_history(args.init_state)

    # Sample it
    # Calculate Stationary Distribution
    
    #  frob_comparison(state_tape,holdTimes_tape,power_val=1024)
#  <<<<<<< HEAD
    #  convergence_of_transitionmat(holdTimes_tape,state_tape,1)
    #  show_trans_matrix(holdTimes_tape, state_tape,args.samprate)
    # power_matrix(holdTimes_tape,state_tape,64)
    #frob_comparison(state_tape, holdTimes_tape)
#  =======
    # convergence_of_transitionmat(holdTimes_tape,state_tape,1)
    #  show_trans_matrix(holdTimes_tape, state_tape,args.samprate)
    # power_matrix(holdTimes_tape,state_tape,64)
    #frob_comparison(state_tape, holdTimes_tape)
    GeneratorFromTransition(holdTimes_tape, state_tape, args.samprate)
#  >>>>>>> fa445c4908338ead4f456689eb2a17dff18cb43e
    #  power_matrix(holdTimes_tape,state_tape,16,samp_rate=args.samprate)
    #  get_stationary()
    #  get_generator_matrix(holdTimes_tape, state_tape)


