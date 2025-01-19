import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# The cost function for the Gromov-Wasserstein distance
def _cost(xsp, xtp, tolog=False):
    """ Returns the GM cost eq (3) in [1] for the Gromov-Wasserstein distance."""
    st = time.time()

    xs = xsp
    xt = xtp

    xs2 = xs * xs
    xs3 = xs2 * xs
    xs4 = xs2 * xs2

    xt2 = xt * xt
    xt3 = xt2 * xt
    xt4 = xt2 * xt2

    X = torch.sum(xs, 0)
    X2 = torch.sum(xs2, 0)
    X3 = torch.sum(xs3, 0)
    X4 = torch.sum(xs4, 0)

    Y = torch.sum(xt, 0)
    Y2 = torch.sum(xt2, 0)
    Y3 = torch.sum(xt3, 0)
    Y4 = torch.sum(xt4, 0)

    xxyy_ = torch.sum((xs2) * (xt2), 0)
    xxy_ = torch.sum((xs2) * (xt), 0)
    xyy_ = torch.sum((xs) * (xt2), 0)
    xy_ = torch.sum((xs) * (xt), 0)

    n = xs.shape[0]

    C2 = 2 * X2 * Y2 + 2 * (n * xxyy_ - 2 * Y * xxy_ - 2 * X * xyy_ + 2 * xy_ * xy_)

    power4_x = 2 * n * X4 - 8 * X3 * X + 6 * X2 * X2
    power4_y = 2 * n * Y4 - 8 * Y3 * Y + 6 * Y2 * Y2

    C = (1 / (n ** 2)) * (power4_x + power4_y - 2 * C2)

    ed = time.time()

    if not tolog:
        return C
    else:
        return C, ed - st


# Gromov-1D function that computes the 1D Gromov Wasserstein distance between sorted projections
def gromov_1d(xs, xt, tolog=False):
    """ Solves the Gromov in 1D (eq (2) in [1] for each proj."""
    
    if tolog:
        log = {}

    st = time.time()
    xs2, i_s = torch.sort(xs, dim=0)

    if tolog:
        xt_asc, i_t = torch.sort(xt, dim=0)  # Sort in ascending order
        xt_desc, i_t = torch.sort(xt, dim=0, descending=True)  # Sort in descending order
        l1, t1 = _cost(xs2, xt_asc, tolog=tolog)
        l2, t2 = _cost(xs2, xt_desc, tolog=tolog)
    else:
        xt_asc, i_t = torch.sort(xt, dim=0)
        xt_desc, i_t = torch.sort(xt, dim=0, descending=True)
        l1 = _cost(xs2, xt_asc, tolog=tolog)
        l2 = _cost(xs2, xt_desc, tolog=tolog)

    toreturn = torch.mean(torch.min(l1, l2))
    ed = time.time()

    if tolog:
        log['g1d'] = ed - st
        log['t1'] = t1
        log['t2'] = t2

    if tolog:
        return toreturn, log
    else:
        return toreturn


# Function that performs the projection using the "sink_" operation (projection of the samples)
def sink_(xs, xt, device, nproj=200, P=None):
    """ Sinks the points of the measure in the lowest dimension onto the highest dimension and applies the projections."""
    dim_d = xs.shape[1]
    dim_p = xt.shape[1]

    # Padding to make dimensions equal
    if dim_d < dim_p:
        random_projection_dim = dim_p
        xs2 = torch.cat((xs, torch.zeros((xs.shape[0], dim_p - dim_d)).to(device)), dim=1)
        xt2 = xt
    else:
        random_projection_dim = dim_d
        xt2 = torch.cat((xt, torch.zeros((xt.shape[0], dim_d - dim_p)).to(device)), dim=1)
        xs2 = xs

    # If P is None, generate a new random projection matrix
    if P is None:
        P = torch.randn(random_projection_dim, nproj)
    p = P / torch.sqrt(torch.sum(P ** 2, 0, True))

    try:
        xsp = torch.matmul(xs2, p.to(device))
        xtp = torch.matmul(xt2, p.to(device))
    except RuntimeError as error:
        print('----------------------------------------')
        print('xs original dim:', xs.shape)
        print('xt original dim:', xt.shape)
        print('dim_p:', dim_p)
        print('dim_d:', dim_d)
        print('random_projection_dim:', random_projection_dim)
        print('projection matrix dim:', p.shape)
        print('xs2 dim:', xs2.shape)
        print('xt2 dim:', xt2.shape)
        print('----------------------------------------')
        print(error)
        raise BadShapeError

    return xsp, xtp


# PowerSpherical class
class PowerSpherical:
    """Class for a PowerSpherical distribution."""
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
    
    def rsample(self):
        # Sampling from the PowerSpherical distribution
        normed = self.loc / torch.sqrt(torch.sum(self.loc ** 2, dim=1, keepdim=True))
        return normed * self.scale.unsqueeze(1)


# RPSGW function for Random Projections Sliced Gromov-Wasserstein
def ebrpsgw_gpu(X, Y, L=10, p=2, device='cuda', kappa=50, nproj=200):
    """Computes the Random Projections Sliced Gromov-Wasserstein distance."""
    dim = X.size(1)
    
    # Compute the random projections (initialization of theta)
    #theta = (X.detach()[np.random.choice(X.shape[0], L, replace=True)] - Y.detach()[np.random.choice(Y.shape[0], L, replace=True)])
    #indices_X = torch.randint(0, X.shape[0], (L,), device=device)
    #indices_Y = torch.randint(0, Y.shape[0], (L,), device=device)
    device = torch.device('cuda') 
    #L = int(L) 
    #indices_X = torch.randperm(X.shape[0], device=device)[:L]
    #indices_Y = torch.randperm(Y.shape[0], device=device)[:L]
    #indices_X = torch.randint(low=0, high=X.shape[0], size=(L,), device=device)
    #indices_Y = torch.randint(low=0, high=Y.shape[0], size=(L,), device=device)
    X = X.to(device)
    linear_projection = nn.Linear(Y.shape[1], X.shape[1]).to(device) 
    Y = Y.to(device)
    # Apply the linear transformation
    Y_projected = linear_projection(Y) 

    sgw = []
    for i in range(L):
        index_X = torch.randint(low=0, high=X.shape[0], size=(1,), device=device)
        index_Y = torch.randint(low=0, high=Y.shape[0], size=(1,), device=device)

        theta = (X[index_X] - Y_projected[index_Y])
        theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))
        #theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))  # Normalize
    
        # PowerSpherical distribution for projections
        ps = PowerSpherical(loc=theta, scale=torch.full((theta.shape[0],), kappa, device=device))
    
        # Sample projections from the distribution
        theta = ps.rsample()
    
        # Project both the source and target data using sink_
        xsp, xtp = sink_(X, Y, device, nproj=nproj)
    
    # Compute the Gromov-Wasserstein distance between the projections
    
        sw = gromov_1d(xsp, xtp, tolog=False)
        sgw.append(sw.item())
    #print(sgw)
    sgw = torch.tensor (sgw, device=device)

    #sgw = torch.stack( sgw, dim=1)
    weights = torch.softmax( sgw , dim = 0)
    sgw = torch.sum(weights*sgw, dim =0).mean()

    return torch.pow(sgw, 1. / p)


# Helper function to create random projections
def rand_projections(dim, L, device):
    """ Generates random projections of a given dimension and number of projections """
    projections = torch.randn(L, dim, device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))  # Normalize
    return projections


# BadShapeError for exception handling during projection
class BadShapeError(Exception):
    pass
