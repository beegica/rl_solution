

import os
from collections import namedtuple
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from catch import Catch

from sithSR import SithSR
from memory_hash import HashedMemory

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def train_catch(catch, model,
                num_games, sessions,
                per_random_act=.1, num_actions=3,
                test_every=None, test_on_games=100, discount=.9):

    # Set up the return Loss and Scores dataframes
    rLoss = pd.DataFrame(columns = ['epoch', 'loss'])
    rLoss['epoch'] = rLoss['epoch'].astype(int)

    if test_every is not None:
        rScores = pd.DataFrame(columns = ['epoch',
                                          str('mean score over ' +
                                              str(test_on_games) +
                                              ' games')])
        rScores['epoch'] = rLoss['epoch'].astype(int)

    # train over total epochs
    for e in range(num_games):

        game_over = False
        tot_score = 0.0
        # get initial input
        input_t = Tensor(catch.observe(flatten=True))

        #iterate over each game
        while not game_over:
            # t_-1 is the previous observation
            input_tm1 = input_t.clone()


            # Randomly pick an action, or use the
            # Previously calculated action
            if (np.random.rand() <= per_random_act) or (model.history is None):
                action = (torch.rand(1) * (num_actions)).type(LongTensor)
            else:
                action = model.pick_action(input_tm1)

            # apply action, get rewards and new state
            reward, timestep, game_over = catch.act(action.cpu().numpy()[0]-1)

            # t_0, current timestep
            input_t = Tensor(catch.observe(flatten=True))

            # store experience
            model.learn_step(input_tm1, action)
            if reward != 0:
                model.add_memory(reward)
                tot_score += reward
        print(e, ": ", tot_score)
        # Reset Game and Model Queue when the game is over.
        catch.reset()
        model.reset_T()

        """# Test the current model weights if need be.
        if (test_every is not None) and ((e + 1) % test_every == 0):

            scores = test_catch(
                catch=catch, model=model,
                test_on_games=test_on_games)

            # Save and report mean Score
            ms = scores['score'].mean()
            rScores.loc[len(rScores), :] = [int(e), ms]
            print("Epoch {:03d} | MeanScore {:.2f}".format(e, ms))

            #catch.reset()"""


def test_catch(catch, model,
                num_games, sessions, num_actions=3,
                test_every=None, test_on_games=100, discount=.9):

    # Set up the return Loss and Scores dataframes
    rLoss = pd.DataFrame(columns = ['epoch', 'loss'])
    rLoss['epoch'] = rLoss['epoch'].astype(int)

    if test_every is not None:
        rScores = pd.DataFrame(columns = ['epoch',
                                          str('mean score over ' +
                                              str(test_on_games) +
                                              ' games')])
        rScores['epoch'] = rLoss['epoch'].astype(int)

    # train over total epochs
    tot_score = 0.0
    for e in range(num_games):

        game_over = False
        #tot_score = 0.0
        # get initial input
        input_t = Tensor(catch.observe(flatten=True))

        #iterate over each game
        while not game_over:
            # t_-1 is the previous observation
            input_tm1 = input_t.clone()


            # Randomly pick an action, or use the
            # Previously calculated action
            action = model.pick_action(input_tm1)

            # apply action, get rewards and new state
            reward, timestep, game_over = catch.act(action.cpu().numpy()[0]-1)

            # t_0, current timestep
            input_t = Tensor(catch.observe(flatten=True))

            # store experience
            model.learn_step(input_tm1, action)
            if reward != 0:
                tot_score += reward
        # Reset Game and Model Queue when the game is over.
        catch.reset()
        model.reset_T()
    print("TOTAL TEST SCORE : ", tot_score)


# global params
height = 10
width=10
num_actions = 3
input_size = width * height

# how many games to train on every frame
games_per_frame = 10

masks = [0] #[0, 1, 2, 4, 8, 12, 16]

q_sizes = {'RL':[1]}
           #'queue': [1, 5, 10]}
mod_type = ["RL"]
num_runs = 1
model = SithSR(state_len=input_size, action_len=num_actions)

run_base = 'catch_long_lr'
#run_base = 'catch_MSE'
num_games = 50
for r in range(num_runs):
    for mt in mod_type:
        for q_size in q_sizes[mt]:
            for mask in masks:
                # set the run name
                run_name = run_base + "_" + mt + "_M" + str(mask) + "_Q" + str(q_size) + "_R" + str(r)
                print(run_name, flush=True)

                mod_file = os.path.join('data', run_name+'.pt')
                #if os.path.exists(mod_file):
                    # already done
                #    continue

                # set the hidden_size from the queue
                hidden_size = input_size * q_size

                # set up the catch environment
                c = Catch(screen_height=height, screen_width=width,
                          game_over_conditions = {'ball_deletions': 1},
                          mask=mask, ball_spawn_rate=height+1,)


                """# set up the model
                if mt == 'queue':
                    model = Queued_DQN(input_size, hidden_size, num_actions,
                                       q_size, use_cuda=use_cuda)
                elif mt == 'sith':
                    model = Sith_DQN(input_size, hidden_size, num_actions,
                                     q_size, use_cuda=use_cuda)
                if use_cuda:
                    model.cuda()

                print(model)"""

                # Pytorch Adagrad Optimizer linked with the model
                #optimizer = optim.Adam(model.parameters())
                #optimizer = optim.RMSprop(model.parameters())


                # train
                train_catch(catch=c, model=model,
                            num_games=num_games,
                            per_random_act=.1, test_every=5,
                            sessions=games_per_frame)
                test_catch(catch=c, model=model,
                            num_games=num_games, test_every=5,
                            test_on_games=100,
                            sessions=games_per_frame)
