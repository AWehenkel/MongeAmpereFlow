import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
from symmetrize import Symmetrize
from torchdiffeq import odeint_adjoint as odeint


class MongeAmpereNodeModule(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, t, x):
        with torch.set_grad_enabled(True):
            if not (x[0].requires_grad):
                x = x[0].clone().detach().requires_grad_(True)
            else:
                x = x[0], x[1].requires_grad_(True)

                x = x[0].clone().detach().requires_grad_(True)
            #dx_dy = torch.autograd.grad(self.net.forward(x), x, create_graph=True)[0]
            #ddx_ddy = -torch.autograd.grad(dx_dy, x, grad_outputs=torch.ones(x.shape[0]), create_graph=True)[0]
        #return dx_dy, ddx_ddy
        return self.net.grad(x), -self.net.laplacian(x)


class MongeAmpereNodeFlow(nn.Module):
    '''
    continuous-time Brenier flow
    dx/dt = du(x)/dx
    dlnp(x)/dt = -d^2 u(x)/dx^2 
    '''
    def __init__(self, net, epsilon, Nsteps, device='cpu', name=None):
        super(MongeAmpereNodeFlow, self).__init__()
        self.device = device
        if name is None:
            self.name = 'MongeAmpereFlow'
        else:
            self.name = name
        self.net = net 
        self.dim = net.dim
        self.epsilon = epsilon 
        self.Nsteps = Nsteps
        self.node = True
        self.odefunc = MongeAmpereNodeModule(self.net)
        self.odefunc.to(device)

    def integrate(self, x, logp, sign=1, epsilon=None, Nsteps=None):
        #default values
        if epsilon is None:
            epsilon = self.epsilon 
        if Nsteps is None:
            Nsteps = self.Nsteps

        #integrate ODE for x and logp(x)
        if sign > 0:
            x, logp = odeint(self.odefunc, (x, logp), torch.tensor([0., epsilon*Nsteps]).to(self.device))
        else:
            x, logp = odeint(self.odefunc, (x, logp), torch.tensor([epsilon * Nsteps, 0.]).to(self.device))
                
        return x[1], logp[1]

    def sample(self, batch_size, sigma=1.0):
        #initial value from Gaussian
        x = torch.Tensor(batch_size, self.dim).normal_().requires_grad_().to(self.device)
        logp = -0.5 * x.pow(2).add(math.log(2 * math.pi* sigma**2)).sum(1) 
        x = x*sigma
        return self.integrate(x, logp, sign=1)

    def nll(self, x):
        '''
        integrate backwards, thus it returns logp(0) - logp(T)
        '''
        logp = torch.zeros(x.shape[0], device=x.device) 
        x, logp = self.integrate(x, logp, sign=-1)
        return logp + 0.5 * x.pow(2).add(math.log(2 * math.pi)).sum(1)

if __name__=='__main__':
    pass 
