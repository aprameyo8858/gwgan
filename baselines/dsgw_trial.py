import torch
import time
import torch.optim as optim
from power_spherical import PowerSpherical  # Import PowerSpherical

# The cost function as defined in the Gromov 1D code you provided
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


# Dynamically Sliced Gromov-Wasserstein (DSGW) function
def DSGW(X, Y, L=5, kappa=10, p=2, s_lr=0.1, n_lr=2, device='cuda', nproj=200):
    """Computes the Dynamically Sliced Gromov-Wasserstein (DSGW) distance."""
    dim = X.size(1)  # Dimension of the data
    
    # Initialize the projection vectors (epsilon)
    epsilon = torch.randn((1, dim), device=device, requires_grad=True)
    epsilon.data = epsilon.data / torch.sqrt(torch.sum(epsilon.data ** 2, dim=1, keepdim=True))  # Normalize
    
    # Set up the optimizer for the projection vectors
    optimizer = optim.SGD([epsilon], lr=s_lr)
    
    # Ensure X and Y also require gradients
    X_detach = X.detach().requires_grad_(True)  # Detach but keep gradient tracking for backward pass
    Y_detach = Y.detach().requires_grad_(True)

    # Iterate to optimize the projections
    for _ in range(n_lr - 1):
        # Create a PowerSpherical distribution and sample the projections
        vmf = PowerSpherical(epsilon, torch.full((1,), kappa, device=device))
        theta = vmf.rsample((L,))  # Correct the shape issue here
        theta = theta.view(L, -1)  # Reshaping

        # Perform the sink operation (projection) on the data
        xsp, xtp = sink_(X_detach, Y_detach, device, nproj=nproj)
        
        # Use the Gromov-1D function to compute the cost between projections
        negative_sgw = -torch.pow(gromov_1d(xsp, xtp, tolog=False).mean(), 1. / p)
        
        optimizer.zero_grad()
        negative_sgw.backward()  # This should now work since negative_sgw is on the computational graph
        optimizer.step()
        
        # Re-normalize epsilon (projection vectors)
        epsilon.data = epsilon.data / torch.sqrt(torch.sum(epsilon.data ** 2, dim=1, keepdim=True))
    
    # Recompute the Gromov-1D cost after optimization
    vmf = PowerSpherical(epsilon, torch.full((1,), kappa, device=device))
    theta = vmf.rsample((L,))  # Correct the shape issue here
    theta = theta.view(L, -1)  # Reshaping
    
    # Perform the sink operation (projection) on the data
    xsp, xtp = sink_(X, Y, device, nproj=nproj)
    
    # Compute the Sliced Gromov-Wasserstein distance
    sw = gromov_1d(xsp, xtp, tolog=False).mean()
    
    return torch.pow(sw, 1. / p)


# BadShapeError for exception handling during projection
class BadShapeError(Exception):
    pass
