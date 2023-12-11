import re
import numpy as np
from sympy import symbols, sympify, simplify, lambdify
import torch

def get_component_map(f):
    # creates a mapping between the variable and the component that variable represents in f
    # e.g. u_t = {0}*u_xxx - 7*u_x + {1}*u**2 results in {0: u_xxx, 1: u**2}
    rhs = extract_rhs(f)

    component_map = {}
    for rh in rhs:
        comps, placeholder_indices, variable_num = split_equation_components(rh)
        for i, num in enumerate(variable_num):
            if num not in component_map:
                component_map[num] = []
            component_map[num].append(comps[placeholder_indices[i]])
    return component_map 

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

def split_equation_components(s):
    # Split the string by addition or subtraction. 
    # The minus sign will stay with the component after it.

    # doesn't work great w/ f strings
    # e.g. f'0*u_x+{coef}*u_xx' might not work (brackets probably confuse it)
    components = [x.strip() for x in re.split('(\+|\-)', s) if x.strip()]
    
    # If a component is just '+' or '-', then merge it with the next component
    i = 0
    while i < len(components) - 1:
        if components[i] in ['+', '-']:
            # Merge with the next component
            components[i] += components[i+1]
            # Remove the next component
            components.pop(i+1)
        else:
            i += 1
    
    # Remove '+' signs from the start of the components
    components = [comp if not comp.startswith('+') else comp[1:] for comp in components]
    
    # Placeholder list
    placeholder_indices = []
    placeholder_nums = []

    # Iterate over the components and check for placeholders
    for i, comp in enumerate(components):
        matches = re.findall(r'{(\d+)}', comp) # Extract numbers inside {}
        
        if matches:
            placeholder_indices.append(i)
            for match in matches:
                placeholder_nums.append(int(match))
                comp = comp.replace('{' + match + '}*', '')  # Remove the placeholder with its number

        components[i] = comp
            
    return components, placeholder_indices, placeholder_nums


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


def spherical_transform(points):
    # implemented from here https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    norm = torch.sum(points**2, dim=0)
    assert torch.abs(norm- 1) < 1e-6, f'points must be normalized to the unit sphere (sum = {norm})'
    assert len(points.shape) == 1, "points must be a 1-dimensional tensor"
    spherical_coords = torch.zeros((points.shape[0]-1))
    for i in range(len(points)-1):
        spherical_coords[i] = torch.atan2(torch.sqrt(torch.sum(points[i+1:]**2)), points[i]) # make sure loop iterates correctly
    spherical_coords[-1] = torch.atan2(points[-1], points[-2])
    return spherical_coords

def inverse_spherical_transform(spherical_coords):
    assert len(spherical_coords.shape) == 1, "points must be a 1-dimensional tensor"
    points = torch.zeros((len(spherical_coords)+1))
    for i in range(len(spherical_coords)):
        points[i] = torch.prod(torch.sin(spherical_coords[:i]))*torch.cos(spherical_coords[i]) # make sure this is implemented correctly
    points[-1] = torch.prod(torch.sin(spherical_coords))
    norm = torch.sum(points**2, dim=0)
    assert torch.abs(norm- 1) < 1e-6, f'points must be normalized to the unit sphere (sum = {norm})'
    return points


def check_spherical_transform():
    # Number of tensors you want to generate
    num_tensors = 5

    for _ in range(num_tensors):
        # Random size for tensor, for example between 5 and 15. You can adjust as needed.
        n = torch.randint(5, 15, (1,)).item()
        
        # Generate tensor of shape (n,) with entries between -1 and 1
        tensor = 2 * torch.rand(n) - 1
        norm = torch.norm(tensor)
        tensor = tensor / norm
        angles = spherical_transform(tensor)
        new_tensor = inverse_spherical_transform(angles)
        # show new_tensor is the same as tensor
        assert torch.allclose(new_tensor, tensor), 'spherical transform not working'


def create_polynomial_basis(variables, max_degree, write_to_file=False, remove_trivial = False, unique_var = None):
    def generate_polynomials(basis, n, current_term=None):
        def term_to_string(term):
            parts = []
            sorted_vars = sorted(term.keys(), key=lambda v: (term[v], v), reverse=True)
            for var in sorted_vars:
                power = term[var]
                if power == 1:
                    parts.append(var)
                elif power > 1:
                    parts.append(f"{var}**{power}")
            return '*'.join(parts)

        if current_term is None:
            current_term = {var: 0 for var in basis}

        if sum(current_term.values()) == n:
            yield term_to_string(current_term)
            return

        for var in basis:
            if current_term[var] < n:
                current_term[var] += 1
                if sum(current_term.values()) <= n:
                    yield from generate_polynomials(basis, n, current_term)
                current_term[var] -= 1

    all_polynomials = set()  # Using a set to automatically handle duplicates
    for i in range(1, max_degree + 1):
        all_polynomials.update(generate_polynomials(variables, i))
    all_polynomials = sorted(list(all_polynomials))  # Convert set back to list and sort for consistency

    if write_to_file:
        with open(write_to_file, 'w') as f:
            for p in all_polynomials:
                f.write(p + '\n')

    

    return all_polynomials





if __name__ == '__main__':
    # Example usage:
    basis = ['x', 'y', 'z']
    n = 3
    answers = create_polynomial_basis(basis, n, write_to_file=f"bases/lv_basis.txt", remove_trivial = True)
    print(answers)
    