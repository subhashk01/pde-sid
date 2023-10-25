from calculate_G_torch import create_matrix_torch, split_equation_components
from util import calculate_neff_torch, read_bases
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def optimize(f, title,lr = .001, A = 0, B = 1000, epochs = 5000, T_max=None):
    bs = read_bases()
    us = np.load('test_curves.npy')
    

    _, placeholders = split_equation_components(f)

    # Create a list of trainable torch parameters
    param_list = [torch.nn.Parameter(torch.randn(1, requires_grad=True)) for _ in placeholders]
    
    title = title+f'_{len(param_list)}params'
    precursor = f'lr{lr}_A{A}_B{B}_epochs{epochs}_cos'
    print(f'{title}/{precursor}')


    optimizer = torch.optim.Adam(param_list, lr=lr)

    # If T_max isn't provided, set it to epochs by default
    if T_max is None:
        T_max = epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)


    # Initialize the progress bar
    pbar = tqdm(range(epochs), desc="Optimizing", leave=True)

    loss_list = []
    parameters_list = []
    
    for i in pbar:
        # Forward pass: compute the loss using the current parameters
        G = create_matrix_torch(bs, f, us, param_list)
        _,s,_ = torch.linalg.svd(G)

        neff = calculate_neff_torch(s) # You'll need to define how you compute the loss using the function f and your data
        loss = -neff
        loss_list.append(loss.item())

        current_lr = optimizer.param_groups[0]['lr']

        parameter = [p.item() for p in param_list]
        parameters_list.append(parameter)

        f_inp = f.format(*["{:.2e}".format(p) for p in parameter])

        # Update the progress bar with the desired metrics
        pbar.set_postfix({"f": f_inp, "neff": neff.item(), "lr": "{:.2e}".format(current_lr)})

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Step the scheduler
        scheduler.step()

    losses = np.array(loss_list)

    directory_path = f'optimize/{title}'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


    np.save(f'{directory_path}/{precursor}_loss.npy', losses)

    parameters = np.array(parameters_list)
    np.save(f'{directory_path}/{precursor}_parameters.npy', parameters)

    return [p.item() for p in param_list]


def plot_loss_parameters(f, title, filename):
    _, placeholders = split_equation_components(f)
    title = title+f'_{len(placeholders)}params'

    losses = np.load(f'optimize/{title}/{filename}_loss.npy')
    parameters = np.load(f'optimize/{title}/{filename}_parameters.npy')

    epochs = list(range(len(losses)))
    smallest_loss_arg = np.argmin(losses)

    plt.figure(figsize = (8,8))
    plt.plot(epochs, losses,c = 'b', label = '-n_eff')
    plt.scatter(epochs[smallest_loss_arg], losses[smallest_loss_arg], c = 'r', label = f'Smallest Loss\n-n_eff = {losses[smallest_loss_arg]:.6f} at Epoch {epochs[smallest_loss_arg]}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss = -n_eff')

    title_str = f'{f}\nSmallest Loss = {losses[smallest_loss_arg]:.4f} at Epoch {epochs[smallest_loss_arg]}\n'
    title_str+=f'{title}/{filename}'
    plt.title(title_str)
    plt.legend()
    plt.show()

    for i in range(parameters.shape[1]):
        plt.figure(figsize = (12,8))
        plt.plot(epochs, parameters[:,i], c = 'b', label = f'Parameter {i+1} through training')
        if np.min(parameters[:,i])<0:
            plt.axhline(0, c = 'k')
        plt.xlabel('Epoch')
        plt.ylabel(f'Parameter {i+1}')
        plt.yscale('symlog')
        smallest_param_arg = np.argmin(np.abs(parameters[:,i]))
        smallest_param = parameters[smallest_param_arg]

    
        plt.scatter(epochs[smallest_param_arg], smallest_param[i], c='r', label = f'Smallest Parameter {i+1}\nParameter 1 = {smallest_param[i]:.3e} at Epoch {epochs[smallest_param_arg]}')

        smallest_loss_param = parameters[smallest_loss_arg]
        title_str = f'{f}\n{f.format(*["{:.2e}".format(p) for p in smallest_loss_param])} Smallest Loss at Epoch {epochs[smallest_loss_arg]}\n'
        title_str+=f'{title}/{filename}'
        plt.title(title_str)
        plt.legend()
        plt.show()

def plot_loss_landscape(f, title, filename):
    _, placeholders = split_equation_components(f)
    title = title + f'_{len(placeholders)}params'

    losses = np.load(f'optimize/{title}/{filename}_loss.npy')
    parameters = np.load(f'optimize/{title}/{filename}_parameters.npy')
    epochs = list(range(len(losses)))

    for i in range(parameters.shape[1]):
        sorted_indices = np.argsort(parameters[:, i])
        sorted_parameters = parameters[sorted_indices, i]
        sorted_losses = losses[sorted_indices]
        sorted_epochs = np.array(epochs)[sorted_indices]

        plt.figure(figsize=(8, 6))
        
        # Plotting the line
        plt.plot(sorted_parameters, sorted_losses, c='b', alpha=0.5, lw=0.5)  # with low alpha for visibility
        # Scatter plotting for color scale
        scatter = plt.scatter(sorted_parameters, sorted_losses, c=sorted_epochs, cmap='plasma', s=10)
        plt.colorbar(scatter, label='Epochs')

        # min_power = np.floor(np.log10(np.min(np.abs(sorted_parameters[sorted_parameters != 0]))))
        # max_power = np.ceil(np.log10(np.max(np.abs(sorted_parameters))))
        
        # ticks = [10**i for i in range(int(min_power), int(max_power)+1)]
        # ticks = [-tick for tick in reversed(ticks)] + ticks
        
        # plt.xticks(ticks)

        plt.title(f"{f}\n{title}/{filename}\nLoss Landscape for Parameter {i+1}")
        plt.xlabel(f"Parameter {i+1}")
        plt.ylabel("Loss = -n_eff")
        plt.xscale('symlog')
        #plt.xlim([min(ticks), max(ticks)])
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    


if __name__ == '__main__':
    f = 'u*u_x+{}*u_xx'
    #f = 'u_xxx-6*u*u_x+{}*u_xx'
    title = 'burgers'
    lrs = [1e-3,1e-4]
    for lr in lrs:
        filename = f'lr{lr}_A0_B1000_epochs5000_cos'
        plot_loss_landscape(f, title, filename)
        #optimize(f, title, lr = lr, epochs = 5000)

    
    
    f = 'u_xxx-6*u*u_x+{}*u_xx'
    title = 'kdv'
    for lr in lrs:
        filename = f'lr{lr}_A0_B1000_epochs5000_cos'
        plot_loss_landscape(f, title, filename)
        #optimize(f, title, lr = lr, epochs = 5000)

