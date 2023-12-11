import numpy as np
from sympy import symbols, diff, Function, sympify, lambdify, simplify, exp
from sympy.plotting import plot
import matplotlib.pyplot as plt
from configs import Config
from generate_curves import gaussians
from matplotlib.animation import FuncAnimation
from model import check_trivial
from itertools import combinations
from util import find_highest_derivative
import re
import pandas as pd
from fractions import Fraction
from scipy.stats import linregress
import math

def calculate_derivative(u, deriv_num, xinc):
    # u is the function to be differentiated
    # deriv is the order of the derivative to take
    # want to make sure u is 1d
    assert u.ndim == 1
    deriv = np.zeros(u.shape)
    if deriv_num == 0:
        deriv = u
    elif deriv_num == 1:
        D = 12

        forward_coeff = [-25, 48, -36, 16, -3]
        for i in range(0, 2):
            for j in range(len(forward_coeff)):
                deriv[i] += forward_coeff[j] * u[i + j]

        center_coeff = [1, -8, 0, 8, -1]
        for i in range(2, len(u) - 2):
            for j in range(len(center_coeff)):
                deriv[i] += center_coeff[j] * u[i + j - 2]

        backward_coeff = [3, -16, 36, -48, 25]
        for i in range(len(u) - 2, len(u)):
            for j in range(len(backward_coeff)):
                deriv[i] += backward_coeff[j] * u[i + j - len(backward_coeff) + 1]
        deriv /= (D*xinc)

    elif deriv_num == 2:
        D = 12
        forward_coeff = [45, -154, 214, -156, 61, -10]
        for i in range(0, 2):
            for j in range(len(forward_coeff)):
                deriv[i] += forward_coeff[j] * u[i + j]
        
        center_coeff = [-1, 16, -30, 16, -1]
        for i in range(2, len(u) - 2):
            for j in range(len(center_coeff)):
                deriv[i] += center_coeff[j] * u[i + j - 2]
        
        backward_coeff = [-10, 61, -156, 214, -154, 45]
        for i in range(len(u) - 2, len(u)):
            for j in range(len(backward_coeff)):
                deriv[i] += backward_coeff[j] * u[i + j - len(backward_coeff) + 1]
        deriv /= (D*xinc**2)

    elif deriv_num == 3:
        D = 8
        forward_coeff = [-49, 232, -461, 496, -307, 104, -15]
        for i in range(0, 3):
            for j in range(len(forward_coeff)):
                deriv[i] += forward_coeff[j] * u[i + j]
        
        center_coeff = [1, -8, 13, 0, -13, 8, -1]
        for i in range(3, len(u) - 3):
            for j in range(len(center_coeff)):
                deriv[i] += center_coeff[j] * u[i + j - 3]
        
        backward_coeff = [15, -104, 307, -496, 461, -232, 49]
        for i in range(len(u) - 3, len(u)):
            for j in range(len(backward_coeff)):
                deriv[i] += backward_coeff[j] * u[i + j - len(backward_coeff) + 1]
        deriv /= (D*xinc**3)
    return deriv

def test_derivative():
    xs, us = gaussians(plot = False)
    xinc = xs[1] - xs[0]
    u = us[0]
    for i in range(5):
        plt.plot(xs, us[i], label = f'u_{i}')
    plt.legend()
    plt.show()


    for i in [1,2, 3]:
        ud = us[i]
        ud_est = calculate_derivative(u, i, xinc)
        plt.title(f'Derivative {i}')
        plt.plot(xs, ud, label='analytic', c = 'r')
        plt.plot(xs, ud_est, label='estimate', c = 'b')
        plt.legend()
        plt.show()

def evaluate_pde(f_str, sym_dict):
    # sym dict maps derivatives to values, e.g. u_xx: [...]
    # evaluates the pde at sym_dict values
    x = symbols('x')
    derivatives = {}
    deriv_list = []
    u = Function('u')(x)
    values = []
    for sym in sym_dict:
        if sym == 'u':
            deriv_sym  = u
        else:
            num_derivs = sym.count('x')
            print(sym, num_derivs)
            deriv_sym = diff(u, x, num_derivs) # ith derivative of u wrt x

        values.append(sym_dict[sym])
        derivatives[sym] = deriv_sym
        deriv_list.append(deriv_sym)

    f = sympify(f_str)
    for deriv_str, deriv_sym in derivatives.items():
        f = f.subs(deriv_str, deriv_sym)
    
    f_func = lambdify([deriv_list], f) # makes useable function out of derivative
    f_vals = f_func(values)
    return f_vals

def compare_pde(f_str, comps):
    # compares analytical and forward euler estimates of a pde
    xs, us = gaussians(plot = False)
    xinc = xs[1] - xs[0]
    u = us[0]
    sym_dict_est = {}
    sym_dict_anyl = {}
    for comp in comps:
        num_derivs = comp.count('x')
        deriv = calculate_derivative(u, num_derivs, xinc)
        sym_dict_est[comp] = deriv

        sym_dict_anyl[comp] = us[num_derivs]

    f_vals_est = evaluate_pde(f_str, sym_dict_est)
    f_vals_anyl = evaluate_pde(f_str, sym_dict_anyl)
    

    plt.plot(xs, f_vals_est, label = 'estimate (forward euler)', c = 'b')
    plt.plot(xs, f_vals_anyl, label = 'analytic', c = 'r')
    plt.title(f'{f_str}')
    plt.legend()
    plt.show()

    plt.plot(xs, f_vals_est - f_vals_anyl, c = 'b')
    plt.title(f'{f_str}\nForward Euler Error (estimate - analytic)')
    plt.show()

def evolve_pde(pde, comps, dt = 0.001, t_max = 1, filename='pde_animation.mp4'):
    xs, us = gaussians(plot = False)
    xinc = xs[1] - xs[0]
    u = us[0]
    def get_values(u):
        sym_dict = {}
        for comp in comps:
            num_derivs = comp.count('x')
            deriv = calculate_derivative(u, num_derivs, xinc)
            sym_dict[comp] = deriv
        f_vals = evaluate_pde(pde, sym_dict)
        return f_vals
    
    t = 0
    while t < t_max:
        u+= dt * get_values(u)
        print(u)
        t += dt
        if t > 0:
            plt.plot(xs, u)
            plt.title(f'{pde}\nTime = {t}')
            plt.show()
    #     # Initialize the figure and axis
    # fig, ax = plt.subplots()
    # line, = ax.plot(xs, u, color='b')
    # ax.set_title(pde)
    
    # # Update function for animation
    # def update(t):
    #     global u
    #     # Calculate the PDE values
    #     f_vals = get_values(u)
        
    #     # Update u for the next time step
    #     u += dt * f_vals
        
    #     # Update the line data for the plot
    #     line.set_ydata(u)
        
    #     # Update the title with the current time
    #     ax.set_title(f'{pde}\nTime = {t * dt:.2f}')
        
    #     return line,

    # # Create the animation object
    # anim = FuncAnimation(fig, update, frames=int(t_max/dt), interval=50, blit=True)
    
    # # Show the animation


def integrate():
    from sympy import symbols, diff, integrate, Function

    # Define the symbols
    x = symbols('x')

    # Define the function u as a function of x
    u = Function('u')(x)

    # Calculate the required derivatives
    u_x = diff(u, x)
    u_xx = diff(u, x, 2)
    u_xxx = diff(u, x, 3)
    u_xxxx = diff(u, x, 4)
    u_xxxxx = diff(u, x, 5)

    # Define the integrand
    integrand = 2 * u * u_x * u_xx**2 + u * u_x**2 * u_xxx
    #integrand = 6*u_x**3*u_xx

    # Compute the integral
    integral_result = integrate(integrand, x)
    print(integral_result.simplify())

def verify_cqs():
    t1 = 'u_x**3*u_xx + u_x*u_xxx**2*u_xxxx + u_x**2*u_xx*u_xxx + u_x**3*u_xxxx + 2*u_x**2*u_xxx*u_xxxx'
    #t2 = 'u_xx*(2*u_x*u_xx**2 + u_x**2*u_xxx + 2*u_xxx*u_xxxx**2 + u_xxx**2*u_xxxxx + 2*u_xx**2*u_xxx + 2*u_x*u_xxx**2 + 4*u_x*u_xx*u_xxxx + u_x**2*u_xxxxx + 2*u_xxxx**2*u_x + 2*u_xxx*u_xxxxx*u_x + 2*u_xx*u_xxx*u_xxxx + 2*u_xxx*u_xx*u_xxxx+u_xxx**3)'
    t2 = '-2*u_x*u_xx**3 + -2*u_xx*u_xxx*u_xxxx**2 + -u_xx*u_xxx**2*u_xxxxx + -2*u_xx**3*u_xxx + -u_x*u_xx*u_xxx**2 + -4*u_x*u_xx**2*u_xxxx + -u_x**2*u_xx*u_xxxxx + -2*u_x*u_xx*u_xxxx**2 + -2*u_x*u_xx*u_xxx*u_xxxxx + -4*u_xx**2*u_xxx*u_xxxx + -u_xx*u_xxx**3'
    t = t1 + '+' + t2
    # split t into list based on +
    check_trivial([t], ['u'])
    tsplit = t.split('+')
    for i,term in enumerate(tsplit):
        print(i,term)

    groupbya = [[0], [1,12,13,14,15], [2,3,5], [4,8,9,10,11], [6,7]]
    for group in groupbya:
        eq = ''
        group_terms = []
        for index in group:
            eq+=tsplit[index]+'+'
            group_terms.append(tsplit[index])
        eq = eq[:-1]
        check_trivial([eq], ['u']) 
        #analyze_combinations(group_terms)

def analyze_combinations(list_terms):
    for i in range(1,len(list_terms)):
        for comb in combinations(list_terms, i):
            eq = ' + '.join(comb)
            v = check_trivial([eq], ['u'], toprint = False)
            if v[0] == 1:
                print(comb)

def find_derivative(f):
    x = symbols('x')
    der = find_highest_derivative(f, wrt='u')
    u = Function('u')(x)
    derivative_list = [u]
    derivative_map = {'u': u}
    for i in range(1,der+1):
        u_diff = diff(derivative_list[-1], x)
        derivative_list.append(u_diff)
        derivative_map['u_'+'x' * i] = u_diff
    func = sympify(f)
    for deriv_str, deriv_sym in derivative_map.items():
        func = func.subs(deriv_str, deriv_sym)
    diff_func = diff(func, x)
    def sym_to_string(expr):
        # Function to convert SymPy expression to the string with u_x, u_xx, etc.
        expr_str = str(expr)
        expr_str = expr_str.replace("u(x)", "u")
        # Replace the SymPy derivative notation with u_x, u_xx, etc.
        expr_str = expr_str.replace("Derivative(u, x)", "u_x")
        for i in range(2, 10):
            expr_str = expr_str.replace("Derivative(u, (x, {}))".format(i), "u_" + "x" * i)
        return expr_str
    return sym_to_string(diff_func)

def generate_exponent_combinations(terms, max_exp, current_exp=[], results=[]):
    if len(current_exp) == len(terms):
        results.append('*'.join([f"{term}**{exp}" if exp > 1 else term for term, exp in zip(terms, current_exp)]))
        return
    for exp in range(1, max_exp+1):  # Exponents from 1 to 4
        generate_exponent_combinations(terms, max_exp, current_exp + [exp], results)

def try_combinations_diff(search_strings, max_num_derivs = 5, max_num_terms = 4, max_exp = 4):
    base_terms = ['u']
    for i in range(1, max_num_derivs+1): # derivatives from 1 to 5
        base_terms.append('u_' + 'x' * i)

    return_dict = {}
    for n in range(1,max_num_terms+1): # five terms looking at
        combos = combinations(base_terms, n)
        num_res = 0
        for combo in combos:
            results = []
            generate_exponent_combinations(combo, max_exp, results=results)
            for result in results:
                derivative = find_derivative(result)
                # want to match 'd*search_string ' to derivative where d is any integer
                # Escape special characters in search_string to use in a regular expression
                derivative = ' '+derivative+' '
                for search_string in search_strings:
                    search_string_escaped = re.escape(search_string)

                    # Define the pattern: either 'd*search_string' or 'search_string' directly
                    pattern = r'(\s\d*\*{}\s|\s{}\s)'.format(search_string_escaped, search_string_escaped)
                    # Search for the pattern in the derivative string

                    if re.search(pattern, derivative):
                        print(f'FOUND {search_string} RESULT')
                        print(f'term: {result}')
                        print(f'deriv: {derivative}\n')
                        num_res+=1
                        if search_string in return_dict:
                            return_dict[search_string].append(result)
                        else:
                            return_dict[search_string] = [result]
    return return_dict
                        

def find_convergence_time():
    # Define the symbols
    x, a = symbols('x a')

    # Define the function
    f = 1/((24 * x / a**4) * exp((-2 * x**2) / a**2) * (1 - (2 * x**2) / a**2))

    # Plot the function using sympy's plot function
    # We need to provide a range for x and assume some value for 'a' since it is a constant
    # For the purpose of plotting, let's assume a = 1
    p = plot(f.subs(a, np.sqrt(2)*.3), (x, -5, 5), show=False, line_color='blue')

    # Show the plot
    p.show()

def get_data(which = 'sin'):
    # Function to convert fraction strings to float
    def convert_fraction(value):
        try:
            return float(value)
        except ValueError:
            return float(Fraction(value))

    dir = '../mathematica/'
    df_90 = pd.read_csv(dir+f'{which}_sol_90s.csv')
    small_t = '.1' if which == 'sin' else '.5'
    df_tenth = pd.read_csv(dir+f'{which}_sol_{small_t}s.csv')

    # Apply the conversion function to all elements in the DataFrame
    df_90 = df_90.applymap(convert_fraction)
    df_90.columns = ['t', 'x', 'u']
    df_tenth = df_tenth.applymap(convert_fraction)
    df_tenth.columns = ['t', 'x', 'u']

    xs = np.array(sorted(df_90['x'].unique()))

    t_tenth = set(df_tenth['t'])
    t = t_tenth.copy()
    t_90 = set(df_90['t'])
    t.update(t_90)
    t = np.array(sorted(list(t)))
    us = []
    for time in t:
        if time == 0.0:
            continue
        if time in t_tenth:
            values = df_tenth[df_tenth['t'] == time]['u'].values
        else:
            values = df_90[df_90['t'] == time]['u'].values
        us.append(values)
    t = t[1:]
    us = np.array(us)
    return t, xs, us


def analyze_mathematica_gauss(which='sin'):
    
    t, xs, us = get_data(which = which) 
    # Find the max/min ratio for each time step in us
    ratios = [max(us[i]) for i in range(len(us))]
    tb = 1/(24*math.pi**3) if which == 'sin' else .0008

    # Filter data to be within the specified x-limits
    tlims = (1e-1, 3)
    filtered_indices = (t >= tlims[0]) & (t <= tlims[1])
    t_filtered = t[filtered_indices]
    ratios_filtered = np.array(ratios)[filtered_indices]

    # Apply log transformation for linear regression
    log_t = np.log(t_filtered)
    log_ratios = np.log(ratios_filtered)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(log_t, log_ratios)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(t, ratios, 'b', label='Max u vs. Time')
    plt.axvline(x=tb, c='k', linestyle='--', label=f'Theoretical Break Time\ntb = {tb:.5f}')
    plt.plot(t_filtered, np.exp(intercept + slope * log_t), 'r--', 
             label=f'Best Fit Line\nlog(u) = {slope:.2f}log(t) + {intercept:.2f}')
    #plt.xlim()
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Max Val of u vs. Time for {which} I.C.')
    plt.xlabel('Time')
    plt.ylabel('Max u')
    plt.legend()
    plt.show()


def analyze_uderiv(which='sin'):
    ts, xs, us = get_data(which=which)
    xinc = xs[1] - xs[0]
    t = ts[-1]
    # Find index where ts = t
    t_index = np.where(ts == t)[0][0]
    deriv_last_u = calculate_derivative(us[t_index], 1, xinc)

    # Calculate mean and standard deviation
    mean_deriv = np.mean(deriv_last_u)
    std_deriv = np.std(deriv_last_u)

    # Filter points that are more than 5 standard deviations above the mean
    filter_mask = np.abs(deriv_last_u - mean_deriv) <= 5 * std_deriv

    # Ensure that filter_mask is a boolean array
    if not isinstance(filter_mask, np.ndarray) or filter_mask.dtype != bool:
        filter_mask = np.array(filter_mask, dtype=bool)

    # Plotting

    plt.plot(xs, us[t_index], c='b', label='u')
    #plt.plot(xs[filter_mask], deriv_last_u[filter_mask], c='r', label='u_x')
    plt.title(f'{which} I.C. evolved to t = {t}')
    plt.legend()
    plt.show()

    




if __name__ == '__main__':
    # strings = [
    #     'u_x*u_xxx**2*u_xxxx',
    #     'u_x*u_xx*u_xxxx**2',
    #     'u_x*u_xx*u_xxx*u_xxxxx',
    #     'u_xx**2*u_xxx*u_xxxx',
    #     'u_xx*u_xxx**3'
    # ]
    # #return_dict = try_combinations_diff(strings)
    # #for key in return_dict:
    # #    print(f'{key}: {return_dict[key]}')
    # terms = ['u_x*u_xxx**3', 'u_xx**2*u_xxx**2','u_x*u_xx*u_xxx*u_xxxx']
    # for term in terms:
    #     print(term)
    #     print(find_derivative(term)+'\n')
    analyze_mathematica_gauss(which = 'sin')
    #analyze_uderiv(which = 'sin')


    #f_str = 'u_xxx-6*u*u_x'
    # f_str = 'u_xxx**3 + 3*u_x**2*u_xxx + 3*u_x*u_xxx**2 + u_x**3'
    # sym_dict = {'u': np.array([1,2,3,4,5]),
    #           'u_x': np.array([1,2,3,4,5]),
    #           'u_xx': np.array([1,2,3,4,5]),
    #             'u_xxx': np.array([1,2,3,4,5])}
    # #print(evaluate_pde(f_str, sym_dict))
    # evolve_pde(f_str, ['u', 'u_x', 'u_xxx'])
    #integrate()


    