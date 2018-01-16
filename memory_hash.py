import torch
from torch.nn import Module
from torch.autograd import Function, Variable
from torch import FloatTensor

class HashedMemory():
    def __init__(self, in_features, sigma):
        self.history = None
        self.sigma = sigma
        self.in_features = in_features

    def add_memory(self, in_history, reward):
        curr_history = torch.cat((in_history.view(-1).unsqueeze(0),
                                  Variable(FloatTensor([reward])).unsqueeze(0)),
                                 1)
        if self.history is None:
            self.history = curr_history
        else:
            # check to see if the new history is anything like any old History
            # if they are similar enough, dependent on sigma, combine them with
            # some kind of average?

            # for now, just do the append
            self.history = torch.cat((self.history, curr_history), 0)

    def retrieve_reward(self, in_history):
        rewards = self.history[:,-1]
        feature_similarity = torch.mm(in_history, self.history[:,:-1])
        output = torch.sum(feature_similarity*rewards)
        return output

    def best_reward(self, in_history):
        rewards = self.history[:,-1]
        feature_similarity = torch.mm(in_history, self.history[:,:-1])
        rewarding = feature_similarity.view(-1)*rewards.view(-1)
        out = self.history[rewarding.max(0)[1], :-1]
