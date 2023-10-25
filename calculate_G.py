from sympy import symbols, diff, Function, sympify, lambdify, simplify
import re
import numpy as np
from util import find_highest_derivative, extract_lhs_variables, extract_variables, read_bases, extract_rhs


def differentiate_f(f_str, num_derivs, us, variables):
    # f is a function of u and its derivatives
    # e.g. f = 'u*u_x'
    # calculates diff = d/dx^n f(u) for n = [0, num_derivs]
    # we then evaluate diff at each point in us (u are the P curves and their derivatives)
    # us is matrix of size P (num curvs) x num_derivs (of u) and N_p (num points evaluated at)

    # KNOWN BUG: DOES NOT WORK WHEN A COEFFICIENT IS 0.

    # all the variables in f are in variables. used for system of equations


    x = symbols('x')

    derivatives = {}
    deriv_list = []
    for variable in variables:
        # define each variable in f as a function of x
        var = Function(variable)(x)

        # Compute derivatives up to some large number
        derivatives[variable] = var
        highest_der = find_highest_derivative(f_str, wrt = variable)

        derivs_data_num = us.shape[1]-1 # how many derivatives' data we actualy have. EDIT THIS FOR EACH RESPECTIVE US
        assert(highest_der + num_derivs <= derivs_data_num) # need at least as much data as derivs

        deriv_list.append(var)
        for i in range(1, 1+derivs_data_num):
            deriv_str = f'{variable}_' + 'x' * i
            deriv_sym = diff(var, x, i) # ith derivative of u wrt x
            derivatives[deriv_str] = deriv_sym
            deriv_list.append(deriv_sym)

    # Convert string to SymPy expression
    past_f = None
    f_deriv = []
    us_repl = np.tile(us, (1,len(variables),1)) # replicates us for each variable. short term fix. NEED TO GENERATE MORE DATA

    for _ in range(num_derivs+1):
        if past_f is None: # first iteration. want to generate f as sympy expression
            f = sympify(f_str)
            for deriv_str, deriv_sym in derivatives.items():
                f = f.subs(deriv_str, deriv_sym)
        else:
            f = diff(past_f, x)

        f_func = lambdify([deriv_list], f) # makes useable function out of derivative
        f_vals = []
        for p in range(us_repl.shape[0]): # might be a faster way to do this. evaluates derivativ
            func_val = f_func(us_repl[p])
            if type(func_val) == int and func_val == 0: # only happens when 0 is the coefficient
                func_val = [0]*us.shape[2]
            f_vals.append(func_val) # gives all N_p values for each of the derivatives in u
        f_deriv.append(f_vals)
        past_f = f

    f_deriv = np.array(f_deriv)

    f_deriv = np.transpose(f_deriv, (1,0,2))
    assert(f_deriv.shape == (us.shape[0], num_derivs + 1, us.shape[2])) # should be P, num_derivs, N_p
    
    return f_deriv # returns (P, num_der, N_p)

def grad_basis_function(b, us, wrt, variables):
    # b is a string that represents b(u'). ex. b(u') = u*u_xxxx+u_xxx
    # we want to calculate gradB w.r.t each u (u, u_x, u_xx, u_xxx, etc.)

    # wrt is what we're taking grad of b w.r.t
    # variables is a list of all variables that are potentially in our basis


    terms = []
    new_us = None # need to add all relevant values of u in here

    for variable in variables:
        der = find_highest_derivative(b, wrt=variable)  # how many derivatives we need to take
        der_data = us.shape[1]-1  # how many derivatives we actually have
        assert(der <= der_data)  # need as much data as derivs required
        
        current_us_slice = us[:, :der+1, :]
        
        # Initialize new_us if it's the first iteration or concatenate to existing new_us
        if new_us is None:
            new_us = current_us_slice
        else:
            new_us = np.concatenate((new_us, current_us_slice), axis=1)
        
        if variable == wrt:
            wrt_terms = [wrt] + [f'{wrt}_' + 'x'*d for d in range(1, der+1)]
            terms.extend(wrt_terms)
        else:
            var_terms = [variable] + [f'{variable}_' + 'x'*d for d in range(1, der+1)]
            terms.extend(var_terms)
    derivatives = [symbols(term) for term in terms]
    wrt_derivatives = [symbols(term) for term in wrt_terms]

    b_func = sympify(b)

    b_deriv = []

    
    for deriv in wrt_derivatives: # each term of b(u')
        deriv_func = diff(b_func,deriv)
        deriv_func = lambdify([derivatives], deriv_func)
        b_vals = []
        for p in range(us.shape[0]):
            val = deriv_func(new_us[p]) # only takes as many derivatives as we require
            if type(val)!=list: # if its always a constant it will come out as an int
                val = val*np.ones((us.shape[2])) # N_p values
            b_vals.append(val) # P x N_p

        b_deriv.append(b_vals) # for each deriv
    b_deriv = np.array(b_deriv)
    b_deriv = np.transpose(b_deriv, (1,0,2))


    wrt_der = find_highest_derivative(b, wrt=wrt)

    assert(b_deriv.shape == (us.shape[0], wrt_der + 1, us.shape[2])) # should be num_derivs, P, N_p
    return b_deriv  # size P, num_derivs, N_p


def calculate_matrix_columns_presum(b,f,us, wrt, variables):
    b_der = find_highest_derivative(b, wrt=wrt)
    fs = differentiate_f(f, b_der, us, variables)
    bs = grad_basis_function(b, us, wrt, variables)
    assert(bs.shape == fs.shape) # should be P, num_derivs, N_p
    column_presum = np.einsum('ijk,ijk->ik', bs, fs) # should be P, N_p
    assert(column_presum.shape == (us.shape[0], us.shape[2])) # should be P, N_p
    return column_presum

def calculate_matrix_columns(b, f, us, wrt, variables):
    # calculates each column of G (g(u') for each b(u') for all P points)

    column_presum = calculate_matrix_columns_presum(b,f,us, wrt, variables)
    column = np.sum(column_presum, axis=1) # sum across N_p
    assert(column.shape == (us.shape[0],)) # should be P
    return column

def create_matrix(bs, f, us, wrt, variables):
    # creates the matrix G (P x num_bs)
    columns = []
    for b in bs:
        column = calculate_matrix_columns(b, f, us, wrt, variables)
        columns.append(column)
    columns = np.array(columns)
    columns = np.transpose(columns)
    assert(columns.shape == (us.shape[0], len(bs))) # should be P, num_bs
    return columns

def create_matrices(bs, fs, us):
    assert type(fs)==list, "fs must be a list of string equations"
    matrix = None
    variables = extract_lhs_variables(fs)
    all_vars = extract_variables(fs)
    assert set(variables) == set(all_vars), "# equations must be # total variables"
    equations = extract_rhs(fs) # we dont want the X_t = part
    for i,equation in enumerate(equations):
        matrix_f = create_matrix(bs, equation, us, variables[i], variables)
        if matrix is None:
            matrix = np.zeros(matrix_f.shape)
        matrix+=matrix_f
    assert(matrix.shape == (us.shape[0], len(bs))) # should be P, num_bs
    return matrix

if __name__ ==  '__main__':
    fs = ['u_t = u_xx']
    bs = read_bases()
    us = np.load('test_curves.npy')

    res = create_matrices(bs,fs,us)
    
    # us = np.load('test_curves.npy')
    # grad_basis_function(b, us, wrt = 'u', variables = variables)
    #f_deriv = differentiate_f(f, num_derivs, us, variables = variables)

   