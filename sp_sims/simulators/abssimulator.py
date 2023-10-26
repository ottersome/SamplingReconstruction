from abc import ABC, abstractmethod
# This will unify methods for General Simulators
# So we can use have a common API
class StochasticSimulator(object):

    def __init__(self, sequence_length):
        self.length = sequence_length 
        self.tape = []

    def generate_data(self):
        pass

    # Something else like this
    #@abstractmehod
    # def solve_some_soe():
        # pass

# These classes will just intermingle the different processes and
# manage them
class SPManager(ABC):

    @abstractmethod
    def __init__(self,length,rates):
        pass

    @abstractmethod
    def generate_history(self):
        pass
    # Run n Processesin parallel to accumuluate 
    # experience faster
    #  @abstractmethod
    #  def simulate_n_processes(self):
    #      pass



