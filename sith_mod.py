

import torch
from torch.nn import Module
from torch.autograd import Function, Variable
import numpy as np

from sith import SITH

class SITHFunction(Function):
    """SITH autograd function."""
    def __init__(self, in_features, tau_0=.1, k=4, c=.1,
                 ntau=100, s_toskip=0, T_toskip=8, alpha=1.0,
                 dur=1/30./10., dt=1/30./10., delay=(1/30. - 1/30./10.),
                 dtype=np.float32, use_cuda=False):
        super(SITHFunction, self).__init__()

        self.sith = SITH(in_features, tau_0=tau_0, k=k, c=c,
                         ntau=ntau, s_toskip=s_toskip,
                         T_toskip=T_toskip, alpha=1.0,
                         dur=dur, dt=dt, delay=delay,
                         use_cuda=use_cuda)

    def forward(self, inputs):
        # initialize outputs so we can grow it properly
        outputs = None

        # loop over inputs from a mini-batch
        for inp in inputs:
            # update t, then delay
            self.sith.update_t(item=inp.unsqueeze(0)*self._dt,
                               dur=self._dur, dt=self._dt,
                               alpha=self._alpha)
            self.sith.update_t(item=None, dur=self._delay, dt=self._dt,
                               alpha=self._alpha)

            # pull out the T_full
            if outputs is None:
                outputs = self.sith.T_full.unsqueeze(0)
            else:
                outputs = torch.cat((outputs, self.sith.T_full.unsqueeze(0)), 0)

        return outputs

    def backward(self, grad_output):
        # return gradient for the most recent entry to the queue
        return grad_output[:,-1]

    def reset(self):
        self.sith.reset()


class SITHModule(Module):
    """Scale Invariant Temporal History (SITH) Module."""
    def __init__(self, in_features, tau_0=.1, k=4, c=.1,
                 ntau=100, s_toskip=0, T_toskip=8, alpha=1.0,
                 dur=1/30., dt=1/300., delay=(1/30. - 1/300.),
                 use_cuda=False):
        super(SITHModule, self).__init__()
        self.use_cuda = use_cuda

        # get fifo instance
        self.sith = SITHFunction(in_features, tau_0=tau_0, k=k, c=c,
                                 ntau=ntau, s_toskip=s_toskip,
                                 T_toskip=T_toskip, alpha=1.0,
                                 dur=dur, dt=dt, delay=delay,
                                 use_cuda=use_cuda)

    def reset(self):
        # reset all to zeros
        self.sith.reset()

    def forward(self, inputs):
        # update the queue
        outputs = self.sith(inputs)

        # return the updated queue
        return outputs


if __name__ == "__main__":

    #from visualize import make_dot
    from torch import nn

    # do some tests
    # [0.03333333, 0.07145296, 0.15316577, 0.32832442, 0.70379256]
    fps = 30.
    dt = 1/fps/10.
    dur = dt
    delay = (1/fps)

    #l1 = nn.Linear(3, 3)
    q1 = SITHModule(3, ntau=25, tau_0 = 1/30., T_toskip=8,
                    dt=dt, dur=dt, delay=delay)
    #q2 = SITH(15, ntau=25, tau_0 = 1/30., T_toskip=8, dt=1/300.)
    #l2 = nn.Linear(15*5, 1)

    x = Variable(torch.FloatTensor([[1,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                                    [0,1,0],[0,0,0],
                                    [0,0,1]]).unsqueeze(1)*10.,
                 requires_grad=True)

    # pass through some operations
    #y = l1(x)
    #y = q1(y.view(y.size(0), -1))
    #y = q2(y.view(y.size(0), -1))
    #out = l2(y.view(y.size(0), -1))

    out = q1((x).view(x.size(0), -1))

    err = (out-0).pow(2).sum()

    print('x:', x)
    print('out:', out)
    print('err:', err)

    #d = make_dot(err)
    #d.view()

    err.backward()

    print('x_grad_data:', x.grad.data)

    print('T_full:', q1.sith._T_full)
    #out.data

    #assert (y.grad.data/2 - y.data).sum() == 0.0
