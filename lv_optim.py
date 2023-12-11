
from util import create_polynomial_basis
from optimize import optimize, plot_loss_parameters
import torch, re

def create_messy_lv(max_exp=3):

    variables = ['x', 'y', 'z']
    bases = create_polynomial_basis(variables, max_exp)
    i = 0
    lv_eq = ''
    eqs = []
    for x_t in ['x_t', 'y_t', 'z_t']:
        lv_eq = ''
        for b in bases:
            # Use double curly braces to include them as literal characters in the string
            lv_eq += f'+{{{i}}}*{b}'
            i+=1
        eqs.append(f'{x_t} = {lv_eq[1:]}')
    return eqs

def run_many_lv(title = 'lv3d',num_runs = 100, num_epochs = 5000, messy = False):

    vals = []
    if messy:
        lv = create_messy_lv(2)
        title = title+'_messy'
    else:
        lv = ['x_t = {0}*x*y+{1}*x*z', 'y_t = {2}*y*z+{3}*y*x', 'z_t = {4}*z*x+{5}*z*y']


    numbers = [int(match) for s in lv for match in re.findall(r'{(\d+)}', s)]
    num_params = max(numbers)-min(numbers)+1
    for i in range(num_runs):
        print(f'Run {i}')
        starting_vals, best_param = optimize(lv, title=title, lr = 0.01, bases='lv3d', epochs=num_epochs, save=False)
        combined = torch.cat((starting_vals.unsqueeze(0), best_param.unsqueeze(0)), dim=0)
        vals.append(combined)
        vals_tensor = torch.stack(vals)
        torch.save(vals_tensor, f'optimize/{title}_{num_params}params/{title}_{num_epochs}epochs_{num_runs}runs.pt')
        print(vals_tensor.shape)
    # read in tensor
    vals_tensor = torch.load(f'optimize/{title}_{num_params}params/{title}_{num_epochs}epochs_{num_runs}runs.pt')

if __name__ == '__main__':
    #lv = create_messy_lv(2)
    run_many_lv(messy = False)
    #lv = ['x_t = {0}*x*y-{1}*x*z', 'y_t = {2}*y*z-{3}*y*x', 'z_t = -{4}*z*y+{5}*z*x']
    #optimize(lv, 'lv', bases = 'lv3d', sphere = True,epochs = 5000)
    #plot_loss_parameters(lv, 'lv3d', sphere = False, filename = 'lr0.001_A0_B1_epochs5000_cos_sphereparam')
