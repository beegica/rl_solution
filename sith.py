
import torch
import numpy as np
from math import factorial


#####################
# functions for Big T
#####################
def _calc_tau_star(tau_0=.1, k=4, c=.1, ntau=100):
    ntau = ntau + 2*k
    tau_star = tau_0*(1+c)**np.arange(-k, ntau+k)
    s = k/tau_star
    return tau_star, s


def _calc_D(s):
    # calc all the differences
    s0__1 = s[1:-1] - s[:-2]
    s1_0 = s[2:] - s[1:-1]
    s1__1 = s[2:] - s[:-2]

    # calc the -1, 0, and 1 diagonals
    A = -((s1_0/s1__1)/s0__1)
    B = -((s0__1/s1__1)/s1_0) + (s1_0/s1__1)/s0__1
    C = (s0__1/s1__1)/s1_0

    # create the matrix
    D = np.zeros((len(s), len(s)))
    D.flat[len(s):-2:len(s)+1] = A
    D.flat[len(s)+1:-1:len(s)+1] = B
    D.flat[len(s)+2::len(s)+1] = C
    D = D.T
    return D


def _calc_invL(s, k):
    # easier to do with matrices than ndarray to keep dimensions straight
    D = np.matrix(_calc_D(s))
    invL = (((-1.) ** k) /
            factorial(k) * (D ** k) * np.matrix(np.diag(s ** (k+1))))[:, k:-k]

    # return as ndarray
    return invL.A.T


class SITH():
    """SITH implementation."""
    def __init__(self, in_features, tau_0=.1, k=4, c=.1,
                 ntau=100, s_toskip=0, T_every=8, alpha=1.0,
                 dt=1/30./10., dtype=np.float32, use_cuda=False):
                 
        self._in_features = in_features
        self._tau_0 = tau_0
        self._k = k
        self._c = c
        self._ntau = ntau
        self._s_toskip = s_toskip
        self._T_full_ind = slice(None, None, T_every)
        self._alpha = alpha
        self._dt = dt
        self._dtype = dtype
        self.use_cuda = use_cuda

        # calc tau_star and s
        tau_star, s = _calc_tau_star(tau_0=tau_0, k=k, c=c, ntau=ntau)
        self._tau_star = tau_star[s_toskip:].astype(self._dtype)
        self._s = s[s_toskip:].astype(self._dtype)

        # make exp diag for exponential decay
        self._sm = np.matrix(np.diag(-self._s)).astype(self._dtype)

        # var for expanding sparse item to all rows of t
        self._it = np.matrix(np.ones((len(self._s), 1))).astype(self._dtype)

        # get the inverse Laplacian
        self._invL = _calc_invL(self._s, self._k).astype(self._dtype)

        # allocate t
        self._t = np.matrix(np.zeros((self._invL.shape[1], self._in_features)),
                            dtype=self._dtype)
        self._t_changed = True

        # convert to torch as needed
        self._invL = torch.from_numpy(self._invL)
        self._t = torch.from_numpy(self._t)
        self._sm = torch.from_numpy(self._sm)
        self._it = torch.from_numpy(self._it)
        self._s = torch.from_numpy(self._s)

        # push out to cuda device as needed
        if use_cuda:
            self.cuda()

    def cuda(self, device_id=None, async=False):
        self._invL = self._invL.cuda(device=device_id, async=async)
        self._t = self._t.cuda(device=device_id, async=async)
        self._sm = self._sm.cuda(device=device_id, async=async)
        self._it = self._it.cuda(device=device_id, async=async)
        self._s = self._s.cuda(device=device_id, async=async)

    @property
    def t(self):
        return self._t

    @property
    def k(self):
        return self._k

    @property
    def tau_star(self):
        return self._tau_star

    def calc_T(self, t=None):
        # grab t from self if needed
        if t is None:
            t = self._t

        # update T from t and index into it
        T_full = (self._invL.mm(t))[self._T_full_ind, :]

        return T_full

    def flatten_T(self, T_full=None):
        if T_full is None:
            T_full = self._T_full

        # flatten T_full
        return T_full.view(T_full.numel())

    def _update_T(self):
        # must make it contiguous or we get view errors
        self._T_full = self.calc_T().contiguous()
        self._T = self.flatten_T()
        self._t_changed = False

    @property
    def T(self):
        # only update T if we need to
        if self._t_changed:
            self._update_T()
        return self._T

    @property
    def T_full(self):
        if self._t_changed:
            self._update_T()
        return self._T_full

    def update_t(self, item=None, dur=None, alpha=None):
        """Update t with input or just decay."""
        dt = self._dt
        if alpha is None:
            alpha = self._alpha
        if dur is None:
            dur = dt
        if item is None:
            # we can just do exponential decay
            self._t = torch.diag(torch.exp(-self._s*dur*alpha)).mm(self._t)
        else:
            # make sure it's a matrix
            #item = np.asmatrix(item)
            # present an item for the duration
            if item.size(0) == 1:
                # must replicate to all t
                tIN = self._it.mm(item)
            else:
                # just use the item
                tIN = item
            # PBS: Can likely vectorize this and get rid of loop
            for i in range(int(dur/dt)):
                deltat = alpha*self._sm.mm(self._t) + tIN/dt
                self._t = self._t + deltat*dt

        # we've updated t, so let other methods know (i.e., T)
        self._t_changed = True

    def reset(self):
        # reset all to zeros
        self._t.zero_()
        self._t_changed = True
