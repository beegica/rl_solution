
import torch
from torch.autograd import Variable

import numpy as np

from sith import SITH
from memory_hash import HashedMemory
# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

"""
Reinforcement learning without going too deep.
"""

class SithSR(object):
    """SITH-based Successor Representation"""
    def __init__(self, state_len, action_len,
                 gamma=.9, alpha=.9, num_drifts=3,
                 info_rate=1./30., dt=1./30./10., dur=1./30./10.,
                 dtype=np.float32):
        # save the vars
        self._dtype = dtype
        self._state_len = state_len
        self._action_len = action_len
        self._info_rate = info_rate
        self._dt = dt
        self._dur = dur
        self._delay = info_rate - dt

        self._num_drifts = num_drifts

        self._gamma = gamma
        self._alpha = alpha

        self.history = None

        self._actions = torch.eye(action_len)

        # init sith
        self._in_sith = state_len + action_len
        self._sith = SITH(self._in_sith, dt=self._dt, ntau=25, dtype=self._dtype)
        self._p0 = torch.zeros((self._in_sith, 1))

        # allocate for M
        self._in_M = self._sith.T.size()[0]
        self._M = torch.zeros((self._in_sith, self._in_M)) # (outM, inM)

    def reset_T(self):
        self._sith.reset()

    def add_memory(self, reward):
        curr_history = torch.cat((self._sith.T.view(-1).unsqueeze(0),
                                  FloatTensor([reward]).unsqueeze(0)),
                                 1)
        if self.history is None:
            self.history = curr_history
        else:
            self.history = torch.cat((self.history, curr_history), 0)


    def _grab_goal(self, state):
        # Save current t for later
        t_save = self._sith._t.clone()

        # Pull out rewards and features from our history. They are needed
        # in seperate steps
        rewards = self.history[:,-1]
        rewarding = torch.zeros(self.history.size(0))
        historical_features = self.history[:, :-1]

        # Delay sith and find the reward values from the has table.
        for i in range(self._num_drifts):
            self._sith.update_t(item=None, dur=self._info_rate)
            feature_similarity = torch.mm(historical_features,
                                          self._sith.flatten_T().unsqueeze(1))
            # Take how closely related the historical features, and multiply
            # those by the rewards. This will tell use which history is the
            # closest to ours and the most rewarding.
            rewarding += feature_similarity.view(-1)*rewards

        # Return t to its previous
        self._sith._t = t_save
        self._sith._t_changed = True

        # Pull out the maximumly rewarding state of T from the hash table.
        # That is the goal for our actor
        out = self.history[rewarding.max(0)[1]][:,:-1]
        return out

    def pick_action(self, state):
        # try out various actions and get max reward
        goal_state = self._grab_goal(state)
        t_save = self._sith.t.clone()

        # try out each action
        potential_futures = None

        # we need to reset t to the old t every loop. &
        for a in self._actions:
            
            # update sith
            sa = torch.cat((state, a), 0)
            self._sith.update_t(sa, dur=self._dur)
            self._sith.update_t(item=None, dur=self._delay)

            # pass through M to get reward
            if potential_futures is None:
                potential_futures = self._M.mm(self._sith.T.view(-1, 1)).view(1, -1)
            else:
                potential_futures = torch.cat((potential_futures,
                                               self._M.mm(self._sith.T.view(-1, 1)).view(1, -1)),
                                              0)
            self._sith._t = t_save
            self._sith._t_changed = True

        evidence = potential_futures.mm(goal_state.view(-1, 1)[-1*self._in_sith:])

        value, action_index = evidence.max(0)

        return action_index

    def learn_step(self, state, next_action):
        # turn into new sa1
        sa1 = torch.cat((state, self._actions[next_action].view(-1)), 0).unsqueeze(1)
        sa1_p = torch.cat((torch.zeros((self._sith.T.size(0) - sa1.size(0), 1)), sa1), 0)
        # calc prediction from new state
        p1 = self._M.mm(sa1_p)
        # update M based on prediction error
        perr = sa1 + self._gamma * p1 - self._p0
        self._M += self._alpha * perr.mm(self._sith.T.view(1, -1))

        # update T with that state action
        self._sith.update_t(item=sa1.view(-1), dur=self._dur)
        self._sith.update_t(item=None, dur=self._delay)

        # prepare for next loop
        #self._next_action = action
        self._p0 = p1
