import re
import numpy as np
from sympy import symbols, sympify, simplify, lambdify
import torch

def extract_lhs_variables(equations):
    # Regular expression to match any word ending with _t, regardless of spaces around the equals sign
    pattern = r'(\w+)_t\s*='
    variables = []

    for equation in equations:
        match = re.search(pattern, equation)
        if match:
            variables.append(match.group(1))
    
    return variables

def extract_variables(equations):
    # extracts all variables from equations
    # Regular expression to match any word followed by an underscore
    pattern = r'(\w+)_'
    variables = []

    for equation in equations:
        matches = re.findall(pattern, equation)
        variables.extend(matches)
    
    # Use set to eliminate duplicates, then convert back to list
    return list(set(variables))

def extract_rhs(equations):
    rhs_list = []

    for equation in equations:
        # Split by '=' and get the second part (RHS), then strip to remove spaces
        rhs = equation.split('=')[1].strip()
        rhs_list.append(rhs)
    
    return rhs_list

def find_highest_derivative(f_str, wrt='u'):
    # Find all occurrences of '{wrt}_xxx...' in the string using an f-string for the pattern
    
    pattern = rf'{wrt}_x+'
    matches = re.findall(pattern, f_str)
    
    if not matches:
        return 0
    
    # Count the number of 'x' in each match and find the maximum
    max_x_count = max(len(match) - len(wrt) - 1 for match in matches)
    
    return max_x_count


def give_equation(eq):
    equations = {}

    nlse = ['u_t = -0.5*v_xx+v*u**2+v**3', 'v_t = 0.5*u_xx-u**3-u*v**2']
    equations['nlse'] = nlse

    kdv = ['u_t = u_xxx-6*u*u_x']
    equations['kdv'] = kdv

    spring = ['u_t = v', 'v_t = -u']
    equations['spring'] = spring
    
    assert eq in equations.keys(), 'equation not found'
    return equations[eq]


def read_bases(eq = None):
    expressions = []
    if eq is None:
        filename = 'basis.txt'
    else:
        filename = f'{eq}_basis.txt'
    with open(f"bases/{filename}", "r") as file:
        for line in file:
            # Remove any leading or trailing whitespace
            line = line.strip()
            
            # Skip empty lines
            if line == "":
                continue
            
            # Add the expression to the list
            expressions.append(line)
    return expressions


def fill_placeholders(strings, params):
    new_strings = []

    for s in strings:
        matches = re.findall(r'{(\d+)}', s) # Extract numbers inside {}
        
        for match in matches:
            idx = int(match)
            if idx < len(params):  # Ensure the index exists in params
                s = s.replace('{' + match + '}', str(params[idx]), 1)

        new_strings.append(s)

    return new_strings


def check_function(f, us, variables):
    # f is a string, us is a matrix of u values
    # we return a matrix of f values evaluated at u
    der = find_highest_derivative(f) # how many derivatives we have
    der_data = us.shape[2]-1 
    assert(der <= der_data) # need as much data as derivs required

    terms = []
    for v in variables:
        terms_v = [v] + [f'{v}_'+'x'*d for d in range(1,der_data+1)]
        terms.extend(terms_v)
    
    derivatives = [symbols(term) for term in terms]
    # Extract the shape of the array
    a, b, c, d = us.shape

    # Reshape and transpose
    us_repl = us.transpose(1, 0, 2, 3).reshape(b, a*c, d)
    us_repl = us_repl[:,:len(variables)*c,:]

    f_symp = sympify(f)
    for i in range(len(derivatives)):
        deriv_str, deriv_sym = terms[i], derivatives[i]
        f_symp = f_symp.subs(deriv_str, deriv_sym)
    f_symp = simplify(f_symp)
    f_func = lambdify([derivatives], f_symp)
    f_vals = []
    for p in range(us_repl.shape[0]):
        f_vals.append(f_func(us_repl[p]))
    return np.array(f_vals)

def check_function_integral(f, us, variables):
    # we integrate f over the domain of u
    vals = check_function(f, us, variables)
    integral = np.sum(vals, axis = 1)
    return integral


def calculate_neff(s, A = 0, B = 1000):
    #n(s_i) = sigmoid((A-log(s_i))/B), and then n_eff = \sum n(s_i). 
    # A is the threshold, B is the width parameter (when B->infinity, the threshold becomes hard) 
    n = 1/(1+np.exp(-(A-np.log10(s))/B))
    n = np.sum(n)
    return n


def calculate_neff_torch(s, A = 0, B = 1000):
    #n(s_i) = sigmoid((A-log(s_i))/B), and then n_eff = \sum n(s_i). 
    # A is the threshold, B is the width parameter (when B->infinity, the threshold becomes hard) 
    # Convert the operations to PyTorch equivalents
    n = 1 / (1 + torch.exp(-(A - torch.log10(s)) / B))
    n = torch.sum(n)
    return n