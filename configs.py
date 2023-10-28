

class Config:
    x_min, x_max = -10, 10 # range of x values to generate each curve on
    N_p = 1000 # number of points to generate each curve on
    N_g = 5 # number of gaussians to use in the mixture model
    num_derivs = 5 # number of derivatives to take
    num_curves = 200 # number of curves to generate
    max_num_eqs = 10 # max number of equations our system can handle

