import torch
import torch.optim as optim
import re
import matplotlib.pyplot as plt
from generate_curves import generate_us

def replace_derivatives(equation_str):
    max_derivative = 0

    # Replace the derivatives
    def derivative_replacement(match):
        nonlocal max_derivative
        derivative_count = len(match.group(1))  # Count the number of 'x' characters
        max_derivative = max(max_derivative, derivative_count)
        return f"y[:,{derivative_count},:]"

    transformed_str = re.sub(r'u_(x+)', derivative_replacement, equation_str)
    
    # Replace standalone 'u' (only when it's not followed by an underscore)
    transformed_str = re.sub(r'u(?![_])', 'y[:,0,:]', transformed_str)
    
    return transformed_str, max_derivative



def differentiate_f(f_str, param_list,x,y, num_derivs):

    # assumes f is a str that looks like f = {}*u*u_x+{}*u_xx...
    # param list are the coefficients
    # x and us are the curves we're working with

    # Ensure parameters are tensors with requires_grad=True

    param_tensors = [torch.tensor(p, requires_grad=True) if not isinstance(p, torch.Tensor) else p for p in param_list]

    # Replace and evaluate the expression
    f_str = f_str.format(*param_tensors)
    f_str, max_derivative = replace_derivatives(f_str)
    assert(max_derivative + num_derivs <= y.shape[1]-1) # need at least as much data as derivs 

    f = eval(f_str)
    grads = []
    for i in range(f.shape[0]):
        grad = f[i]
        for n in range(4):
            grad = torch.autograd.grad(grad.sum(), x, create_graph=True)[0]
        grads.append(grad)

    # for i in range(f.shape[0]):
    #     print(i)
    #     jacobian[i] = torch.autograd.grad(f[i].sum(), x, retain_graph=True)[0]

    # Define the L1 regularization loss
    

    # loss = torch.abs(f).sum() 

    # optimizer = optim.Adam(param_tensors, lr=0.01)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # # Plot f
    # plt.plot(x.detach().numpy(), f[10].detach().numpy(), label='f')
    
    # # Plot the derivative of f
    # plt.plot(x.detach().numpy(), jacobian[1].detach().numpy(), label='df/dx')
    # plt.axhline(y = 0)

    # plt.legend()
    # plt.show()

    
def compute_derivative(f, x):
    # Make sure x has a batch dimension to match f for broadcasting
    x_batched = x.unsqueeze(0).expand_as(f)
    
    # Create an identity matrix with the shape of f for vectorized gradient computation
    identity = torch.eye(f.shape[1]).reshape(1, f.shape[1], f.shape[1]).repeat(f.shape[0], 1, 1)

    # Compute the derivatives
    derivatives, = torch.autograd.grad(f, x_batched, grad_outputs=identity, retain_graph=True, create_graph=True)
    
    return derivatives

if __name__ == '__main__':
    f = '{}*u_xxx+{}*u*u_x'

    param_list = [-1.,6.]
    x,y = generate_us()
    print(y.shape)
    differentiate_f(f,param_list,x,y, 2)