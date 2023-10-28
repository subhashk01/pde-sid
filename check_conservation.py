from sympy import symbols, diff, Function, sympify, lambdify, simplify
import numpy as np
import matplotlib.pyplot as plt
from calculate_G import find_highest_derivative, calculate_matrix_columns
from model import find_cq, threshold_and_format
from util import read_bases, check_function_integral, extract_lhs_variables, extract_rhs, give_equation, calculate_neff
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


def evaluate_basis_integral(basis,fs,us, variables):
    # calculates int dh/dt for a basis function across all curves
    integral = None
    for i, wrt in enumerate(variables):
        integral_wrt = calculate_matrix_columns(basis, fs[i], us, wrt, variables)
        if integral is None:
            integral = np.zeros(integral_wrt.shape)
        integral+=integral_wrt
    return integral

def test_basis_integral(basis,f,us):
    # sums int dh/dt for a basis function across all curves
    res = evaluate_basis_integral( basis,f,us)
    rmse = np.sqrt(np.mean(res**2))
    return rmse




def graph_singular_values(results, fs, us):
    s = results['s_cq']
    s_cq = results['sol_cq_sparse']
    non_s_cq = results['non_sol_cq_sparse']
    bs = results['bases']

    variables = extract_lhs_variables(fs)
    rhs = extract_rhs(fs)

    concat = np.concatenate((s_cq, non_s_cq))

    expressions = threshold_and_format(bs, concat)
    rmses = []
    
    plt.figure(figsize = (8,6))
    trivials = None
    if 'trivial_non_sol' in results.keys():
        trivials = np.concatenate((results['trivial_sol'], results['trivial_non_sol']))

    for i, e in enumerate(expressions):
        ev = evaluate_basis_integral(e, rhs, us, variables)
        rmse = np.sqrt(np.mean(ev**2))
        rmses.append(rmse)
        if trivials is not None and trivials[i] == 1:
            plt.scatter(i, s[i], label = f'TRIVIAL CQ{i+1}: {e}', c = 'k')
        else:
            plt.scatter(i, s[i], label = f'CQ{i+1}: {e}')

    for i, e in enumerate(expressions):
        plt.annotate(f'{rmses[i]:.2e}', (i, s[i]))  # Adds RMSE text to each point

    rest_x = range(len(expressions), len(s))
    rest_y = s[len(expressions):]
    plt.scatter(rest_x, rest_y)
    plt.plot(range(len(s)),s, c = 'k', lw = '0.5', label = 'Singular Values')
    plt.axhline(y=1e-6, color='r', linestyle='--', label = 'TOLERANCE')

    n_eff = calculate_neff(results['s_cq_nonorm'])


    plt.title(f'{fs}\n'+r'Annotated w RMSE of $\frac{dH}{dt} = \int \frac{dh}{dx}f+\frac{dh}{du_x}\frac{df}{dx}...dx$'+f'\nneff = {n_eff:.5f}')
    plt.ylabel('Singular Value')
    plt.xticks(range(len(expressions)), range(1,1+len(expressions)), rotation=0)
    plt.xlabel('CQ Number')
    plt.yscale('log')
    plt.legend()
    plt.show()





if __name__ == '__main__':
    # read the file basis.txt into a variable named b
    eq = 'kdv'
    bs = read_bases(eq)
    fs = give_equation(eq)
    fs = ['u_t = 1*u_xxx-6*u*u_x']
    #f = ['u_t = v', 'v_t = -1*u']
    us = np.load('test_curves.npy')
    results = find_cq(fs, bs, check_trivial_bases=True, check_trivial_solutions = True)
    graph_singular_values(results, fs, us)

