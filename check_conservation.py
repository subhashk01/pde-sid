from sympy import symbols, diff, Function, sympify, lambdify, simplify
import numpy as np
import matplotlib.pyplot as plt
from calculate_G import find_highest_derivative, calculate_matrix_columns
from model import find_cq, threshold_and_format
from util import read_bases, check_function_integral
from configs import Config
plt.rcParams['text.usetex'] = True
config = Config()



def check_bases(bases):
    # checks the integral of each basis function
    # want these to roughly be the same magnitudeå
    us = np.load('test_curves.npy')
    for b in bases:
        integral = check_function_integral(b, us)
        mean,std = np.mean(integral), np.std(integral)
        print(f'basis: {b}\n\tmean = {mean:.2e}\n\tstd = {std:.2e}\n')
    
def check_correlation(bases, us):
    # function sees how correlated the integrals of two bases are

    # Ensure there are only two bases
    assert len(bases) == 2, 'need 2 bases for scatter'

    # Get the integral values for each base
    f1 = check_function_integral(bases[0], us)
    f2 = check_function_integral(bases[1], us)
    
    # Removing outliers that are more than 10 standard deviations above the mean
    mean1, std1 = np.mean(f1), np.std(f1)
    mean2, std2 = np.mean(f2), np.std(f2)
    
    valid_indices = np.where((f1 <= mean1 + 10*std1) & (f2 <= mean2 + 10*std2))
    f1, f2 = f1[valid_indices], f2[valid_indices]
    
    # Plotting scatter
    plt.scatter(f1, f2, label='Data points')
    
    # Plotting the best-fit line
    m, b = np.polyfit(f1, f2, 1)
    plt.plot(f1, m*f1 + b, color='red', label=f'Best fit line: y = {m:.2f}x + {b:.2f}')
    
    # Calculating the correlation coefficient
    r = np.corrcoef(f1, f2)[0, 1]
    
    # Labelling and displaying plot
    plt.xlabel(f'Integral of {bases[0]}')
    plt.ylabel(f'Integral of {bases[1]}')
    plt.title(f'Correlation of Basis Functions (r = {r:.2f})')
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_basis_integral(basis,f,us):
    integral = calculate_matrix_columns(basis, f, us)
    return integral

def test_basis_integral(basis,f,us):
    res = evaluate_basis_integral( basis,f,us)
    rmse = np.sqrt(np.mean(res**2))
    return rmse





def visualize_basis_results(results,b, f):
    us = np.load('test_curves.npy')
    res = results['sol_cq_sparse']
    cq = threshold_and_format(b, res)
    data = []
    for i,c in enumerate(cq):
        ev = evaluate_basis_integral(c,f,us)
        ev = np.abs(ev)
        data.append(ev)
        rmse = np.sqrt(np.mean(ev**2))
        print(f'Eq {i+1}: {c}\n\tRMSE = {rmse:.2e}\n')
    data = np.array(data)
    num_eq = data.shape[0]

    # Same data setup as above...

    # Log-transformed box plot
    log_data = np.log10(np.abs(data))
    fig, ax = plt.subplots()
    ax.boxplot(log_data.T, vert=True, patch_artist=True)

    # Styling
    ax.set_title(f'f(x) = {f}\nLog-Transformed Box Plot')
    ax.set_xlabel('Conserved Quantities')
    ax.set_ylabel('log10sum(dH/dx * f(x))')
    ax.set_xticks(range(1, num_eq + 1))
    ax.set_xticklabels([f'Eq {i+1}\n{cq[i]}' for i in range(num_eq)])

def graph_singular_values(results, f, us):
    s = results['s_cq']
    s_cq = results['sol_cq_sparse']
    non_s_cq = results['non_sol_cq_sparse']
    bs = results['bases']

    concat = np.concatenate((s_cq, non_s_cq))

    expressions = threshold_and_format(bs, concat)
    print(expressions)
    rmses = []
    
    plt.figure(figsize = (8,6))
    for i, e in enumerate(expressions):
        if 'trivials' in results.keys() and results['trivials'][i] == 1:
            plt.scatter(i, s[i], label = f'REMOVED CQ{i+1}: {e}', c = 'k')
        else:
            ev = evaluate_basis_integral(e, f, us)
            rmse = np.sqrt(np.mean(ev**2))
            rmses.append(rmse)
            plt.scatter(i, s[i], label = f'CQ{i+1}: {e}')

    for i, e in enumerate(expressions):
        plt.annotate(f'{rmses[i]:.2e}', (i, s[i]))  # Adds RMSE text to each point
    rest_x = range(len(expressions), len(s))
    rest_y = s[len(expressions):]
    print(len(rest_x), len(rest_y))
    plt.scatter(rest_x, rest_y)
    plt.plot(range(len(s)),s, c = 'k', lw = '0.5', label = 'Singular Values')
    plt.axhline(y=1e-6, color='r', linestyle='--', label = 'TOLERANCE')


    plt.title(f'f(x) = {f}\n'+r'Annotated w RMSE of $\frac{dH}{dt} = \int \frac{dh}{dx}f+\frac{dh}{du_x}\frac{df}{dx}...dx$')
    plt.ylabel('Singular Value')
    plt.xticks(range(len(expressions)), range(1,1+len(expressions)), rotation=0)
    plt.xlabel('CQ Number')
    plt.yscale('log')
    plt.legend()
    plt.show()





if __name__ == '__main__':
    # read the file basis.txt into a variable named b
    b = read_bases()
    f = 'u*u_x'
    us = np.load('test_curves.npy')
    results = find_cq(f, b, check_trivial_bases=False, check_trivial_solutions = False)
    graph_singular_values(results, f, us)
    for t in ['sol_cq_sparse', 'non_sol_cq_sparse']:
        readable = threshold_and_format(b, results[t])
        print(t)
        for r in readable:
            print('\t'+r)

