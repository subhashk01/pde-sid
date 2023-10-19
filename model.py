import autograd.numpy as np
from autograd import grad, jacobian
from scipy.optimize import minimize
from calculate_G import create_matrix
import copy
from util import read_bases, check_function_integral

    
    
def find_cq(f, bases, check_trivial_bases = True, check_trivial_solutions = True, seed = 0):
    np.random.seed(seed)
    # load data
    print("#### Loading data ####")
    us = np.load('test_curves.npy')

    bases = np.array(bases)
    if check_trivial_bases:
        print('#### Removing Trivial Bases ####')
        trivial = check_trivial(bases)
        if np.sum(trivial) != 0:
            print('Removed: ', bases[trivial==1])
            bases = bases[trivial==0]
            print('Remaining: ', bases)
    
    print("#### Computing bases and gradients #####")
    f_grad_prod = create_matrix(bases, f, us)
    results = svd_and_sparsify(f_grad_prod)
    results['bases'] = bases

    if check_trivial_solutions:
        print('#### Calculating Number of Trivial Conserved Quantities ####')
        for triviality in ['non_sol', 'sol']:
            solutions = threshold_and_format(bases, results[triviality+'_cq_sparse'])
            trivials = check_trivial(solutions)
            if triviality == 'sol':
                print(f'{np.sum(trivials==1)} Trivial '+triviality)
                print(f'{np.sum(trivials==0)} Non Trivial '+triviality)
            results['trivial_'+triviality] = trivials
    return results


def svd_and_sparsify(G, return_non_sol = True, tol_cq=1e-6 ):
    # if return_non_sol, also sparisifies non solutions
    # this parameter is off when checking for triviality

    results = {}

    #### Solving thetas ####
    print("#### Solving thetas ####")
    u, s, v = np.linalg.svd(G)

    u,s,v = u[::-1], s[::-1], v[::-1] # orders everything from smallest to largest SV, since that's what we care about
    results['s_cq_nonorm'] = copy.deepcopy(s)

    s = s/np.sum(s)
    results['s_cq'] = copy.deepcopy(s)

    num_cq = np.sum(s<tol_cq) 
    print('Number of Conserved Quantities: ', num_cq)
    solutions = v[:num_cq]

    print("#### Sparsifying Solution Thetas ####")
    results['sol_cq'] = copy.deepcopy(solutions)
    sparse_sol = np.asarray([[]])
    if len(solutions):
        sparse_sol = sparsify(solutions)
    results['sol_cq_sparse'] = copy.deepcopy(sparse_sol)
    
    if return_non_sol:
        print('#### Sparsifying Non Solution Thetas ####')
        non_solutions = v[num_cq:]
        results['non_sol_cq'] = copy.deepcopy(non_solutions)
        sparse_non_sol = np.asarray([[]])
        if len(non_solutions):
            sparse_non_sol = sparsify(non_solutions)
        results['non_sol_cq_sparse'] = copy.deepcopy(sparse_non_sol)
    
    return results


def sparsify(solutions, tol_dep=1e-4, seed=0, sparse_run=10, sparse_tol=1e-32, max_iter=100):

    n = len(solutions)
    def vector2orth(V):
        V = V.reshape(n, n)
        I = np.eye(n, n)
        A = (V-V.T)/2 # creates skew symmetric matrix
        Q = np.matmul(I-A, np.linalg.inv(I+A)) # cayley transform (makes skew symmetric go to orthogonal)
        return Q

    def L1(V):
        Q = vector2orth(V)
        sp = np.matmul(Q, solutions)
        l1 = np.sum(np.abs(sp))
        return l1

    grad_L1 = grad(L1)
    
    sols = []
    sol_funcs = []
    # try different seeds
    num_seed = sparse_run # how many times we'll do sparisy turn
    for i in range(num_seed):
        print("{}/{}".format(i,num_seed))
        seed = i
        np.random.seed(seed)
        V = np.random.randn(n, n)
        sol = minimize(L1, V, method = "L-BFGS-B", jac=grad_L1, tol=1e-10, options={'maxiter':max_iter})
        sol_funcs.append(sol.fun)
        sols.append(sol.x)

    sol_funcs = np.array(sol_funcs)
    winner = np.argmin(sol_funcs)
    sol_func = sols[winner]
    solutions = np.matmul(vector2orth(sol_func), solutions)

    if len(solutions.shape) == 1:
        solutions = np.array([solutions])
    return solutions


def check_trivial(bases, epsilon = 1e-4):
    # returns an array x of size bases that has a 0 if the base is non trivial and 1 if its trivia;

    # since we want to know if an individual base/solution is degenerate, not if a combination of them is degenerate
    # we don't actually have to use the sparsification code. this is only necessary for 
    x = np.zeros(len(bases))
    us = np.load('test_curves.npy')

    values = []
    for b in bases:
        val = check_function_integral(b, us)
        values.append(val)
    values = np.transpose(np.array(values))

    trivial_svd = svd_and_sparsify(values, return_non_sol = False, tol_cq = 1e-6)
    trivial_sp = trivial_svd['sol_cq_sparse']

    if trivial_sp.shape[1]==0: # no solutions found
        return x 

    for i in range(trivial_sp.shape[0]):
        # we want there to be a 1 in each row and nothing else
        row_pos = np.abs(trivial_sp[i])
        largest_val = np.max(row_pos)
        if np.abs(1-largest_val)<epsilon and np.abs(1-np.sum(row_pos))<epsilon:
            # first condition says that the largest value is almost exactly equal 1
            # second condition forces every other value in row_pos to basically be 0
            x[np.argmax(row_pos)] = 1
    x = x.astype(int) 
    return x




def threshold_and_format(b, a, threshold=1e-1):
    if len(a)==0:
        return ['NO SOLUTIONS']
    assert(a.shape[1] == len(b))
    # Threshold the coefficients
    a[np.abs(a) < threshold] = 0
    
    # Initialize list to hold conserved quantities as strings
    conserved_quantities = []

    # Loop through each row in the thresholded array `a`
    for row in a:
        terms = []
        for coef, basis in zip(row, b):
            # Only include terms with non-zero coefficients
            if coef != 0:
                # check if coef is within threshold of a whole number
                if np.abs(coef - np.round(coef)) < threshold:
                    term = f"{coef:.0f}*({basis})"
                else:
                    term = f"{coef:.2f}*({basis})"
                terms.append(term)
        
        # Join the terms into a single string and append to list
        conserved_quantity = " + ".join(terms)
        
        # Skip empty strings
        if conserved_quantity:
            conserved_quantities.append(conserved_quantity)
    
    return conserved_quantities
    

if __name__ == '__main__':
    f = 'u_xxx-6*u*u_x'
    b = read_bases()
    b = ['-1*u*u_xx-u_x**2', 'u_x**2', '-1*(u)', '0.89*(u**3) + 0.45*(u_x**2)']
    #find_cq(f,b)
    print(check_trivial(b))

    
