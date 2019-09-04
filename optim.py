""" Wrapper of optimizers in torch.optim for computation of exponential moving average of parameters"""

import torch

def build_ema_optimizer(optimizer_cls):
    class Optimizer(optimizer_cls):
        def __init__(self, *args, polyak=0.0, **kwargs):
            if not 0.0 <= polyak <= 1.0:
                raise ValueError("Invalid polyak decay rate: {}".format(polyak))
            super().__init__(*args, **kwargs)
            self.defaults['polyak'] = polyak
            self.ema = False

        def step(self, closure=None):
            super().step(closure)

            # update exponential moving average after gradient update to parameters
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]

                    # state initialization
                    if 'ema' not in state:
                        state['ema'] = torch.zeros_like(p.data)

                    # ema update
                    state['ema'] -= (1 - self.defaults['polyak']) * (state['ema'] - p.data)

        def use_ema(self, use_ema_for_params=True):
            """ substitute exponential moving average values into parameter values """
            if self.ema ^ use_ema_for_params:  # logical XOR; swap only when different;
                try:
                    print('Swapping EMA and parameters values. Now using: ' + ('EMA' if use_ema_for_params else 'param values'))
                    for group in self.param_groups:
                        for p in group['params']:
                            data = p.data
                            state = self.state[p]
                            p.data = state['ema']
                            state['ema'] = data
                    self.ema = use_ema_for_params
                except KeyError:
                    print('Optimizer not initialized. No EMA values to swap to. Keeping parameter values.')


        def __repr__(self):
            s = super().__repr__()
            return self.__class__.__mro__[1].__name__ + ' (\npolyak: {}\n'.format(self.defaults['polyak']) + s.partition('\n')[2]

    return Optimizer

Adam = build_ema_optimizer(torch.optim.Adam)
RMSprop = build_ema_optimizer(torch.optim.RMSprop)


if __name__ == '__main__':
    import copy
    torch.manual_seed(0)
    x = torch.randn(2,2)
    y = torch.rand(2,2)
    polyak = 0.9
    _m = torch.nn.Linear(2,2)
    for optim in [Adam, RMSprop]:
        m = copy.deepcopy(_m)
        o = optim(m.parameters(), lr=0.1, polyak=polyak)
        print('Testing: ', optim.__name__)
        print(o)
        print('init loss {:.3f}'.format(torch.mean((m(x) - y)**2).item()))
        p = torch.zeros_like(m.weight)
        for i in range(5):
            loss = torch.mean((m(x) - y)**2)
            print('step {}: loss {:.3f}'.format(i, loss.item()))
            o.zero_grad()
            loss.backward()
            o.step()
            # manual compute ema
            p -= (1 - polyak) * (p - m.weight.data)
        print('loss: {:.3f}'.format(torch.mean((m(x) - y)**2).item()))
        print('swapping ema values for params.')
        o.use_ema(True)
        assert torch.allclose(p, m.weight)
        print('loss: {:.3f}'.format(torch.mean((m(x) - y)**2).item()))
        print('swapping params for ema values.')
        o.use_ema(False)
        print('loss: {:.3f}'.format(torch.mean((m(x) - y)**2).item()))
        print()
