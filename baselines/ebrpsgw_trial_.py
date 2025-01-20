import torch
import time
import numpy as np
from torch import distributions as dist

# For the energy-based spherical distribution
class PowerSpherical(dist.Distribution):
    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.dim = loc.size(1)
        super().__init__(validate_args=validate_args)
    
    def rsample(self):
        normed_loc = self.loc / torch.norm(self.loc, dim=1, keepdim=True)
        noise = torch.randn_like(self.loc)  # Standard Gaussian noise
        noise = noise / torch.norm(noise, dim=1, keepdim=True)
        
        # Transform using the scale
        sample = self.loc + self.scale.unsqueeze(1) * noise
        return sample

# Spherical sampling function (for simplicity in EB-SGW)
def spherical_sample(n, dim, scale=1.0, device='cuda'):
    """Sample points uniformly distributed on a sphere in `dim`-dimensional space."""
    u = torch.randn((n, dim), device=device)
    norm = torch.norm(u, dim=1, keepdim=True)
    u = u / norm  # Normalize to make it a unit vector
    return scale * u  # Apply scale

# The main EB-SGW function
def ebrpsgw_gpu(xs, xt, device, nproj=200, tolog=False, P=None, kappa=1, L=10, p=2):
    """
    Returns Energy-Based Sliced Gromov-Wasserstein (EB-SGW) distance.
    
    Parameters:
    ----------
    xs : tensor, shape (n, p)
        Source samples
    xt : tensor, shape (n, q)
        Target samples
    device : torch.device
        The device on which to perform computation
    nproj : int
        Number of projections. Ignored if P is not None
    P : tensor, shape (max(p,q), n_proj)
        Projection matrix. If None creates a new projection matrix
    tolog : bool
        Whether to return timings or not
    kappa : float
        The scale for the spherical distribution in EBRPSW
    L : int
        Number of projections to generate for theta
    p : int
        Power for Wasserstein distance computation

    Returns:
    -------
    C : tensor, shape (n_proj, 1)
        Energy-based sliced Gromov-Wasserstein cost for each projection
    """
    if tolog:
        log = {}

    # Step 1: Sink the points using the Delta padding operator
    if tolog: 
        st = time.time()
        xsp, xtp = sink_(xs, xt, device, nproj, P)
        ed = time.time()   
        log['time_sink_'] = ed - st
    else:
        xsp, xtp = sink_(xs, xt, device, nproj, P)

    # Step 2: Create transformation distribution (theta) using the energy-based approach
    dim = xs.size(1)
    
    # Sample the transformation from a spherical distribution
    theta = spherical_sample(L, dim, scale=kappa, device=device)
    
    # Step 3: Compute the Gromov-Wasserstein distances with energy-based transformations
    if tolog:
        st = time.time()
        d, log_gw1d = gromov_1d_energy(xsp, xtp, theta, p=p, tolog=True)
        ed = time.time()
        log['time_gw_1D'] = ed - st
        log['gw_1d_details'] = log_gw1d
    else:
        d = gromov_1d_energy(xsp, xtp, theta, p=p, tolog=False)

    # Return either the computed distance or the log details
    if tolog:
        return d, log
    else:
        return d

# Gromov-Wasserstein 1D computation with energy-based transformations
def gromov_1d_energy(xsp, xtp, theta, p=2, tolog=False): 
    """
    Computes Gromov-Wasserstein distance using the energy-based transformation theta.

    Parameters:
    ----------
    xsp : tensor, shape (n, n_proj)
         1D sorted samples for each projection in the source
    xtp : tensor, shape (n, n_proj)
         1D sorted samples for each projection in the target
    theta : tensor, shape (L, n_proj)
           Transformation sampled from the distribution
    p : int
        Power for Wasserstein distance
    tolog : bool
        Whether to return timings or not

    Returns:
    -------
    distance : tensor, shape (n_proj, 1)
        The energy-based sliced Gromov-Wasserstein distance for each projection
    """
    # Apply the energy-based transformation to the distances
    if tolog:
        log = {}
    
    st = time.time()
    xs2, i_s = torch.sort(xsp, dim=0)
    xt_asc, i_t = torch.sort(xtp, dim=0)

    # Calculate pairwise distances using the energy-based transformation
    theta_reshaped = theta.view(-1, 1, theta.shape[-1])  # Ensure correct shape for broadcasting
    dist = compute_energy_based_dist(xs2, xt_asc, theta_reshaped, p)

    toreturn = torch.mean(dist)  # Aggregate distances across projections
    
    ed = time.time()
    
    if tolog:
        log['g1d'] = ed - st
    
    if tolog:
        return toreturn, log
    else:
        return toreturn


# Compute pairwise distances considering the energy-based transformation
def compute_energy_based_dist(xs, xt, theta, p):
    """Computes pairwise distances considering the energy-based transformation"""
    # Compute the transformed distances based on the projections and theta
    theta_reshaped = theta.reshape(12288, 64) 

    xs_transformed = xs + torch.matmul(theta_reshaped, xt.T)
    dist = torch.norm(xs_transformed - xt, p=p, dim=1)
    
    return dist

# Delta padding operator (sink function)
def sink_(xs, xt, device, nproj=200, P=None): 
    """ Sinks the points of the measure in the lowest dimension onto the highest dimension and applies the projections.
    Only implemented with the 0 padding Delta=Delta_pad operator (see [1])
    
    Parameters:
    ----------
    xs : tensor, shape (n, p)
         Source samples
    xt : tensor, shape (n, q)
         Target samples
    device : torch device
    nproj : integer
            Number of projections. Ignored if P is not None
    P : tensor, shape (max(p,q),n_proj)
        Projection matrix
    Returns:
    -------
    xsp : tensor, shape (n,n_proj)
           Projected source samples 
    xtp : tensor, shape (n,n_proj)
           Projected target samples 
    """
    dim_d = xs.shape[1]
    dim_p = xt.shape[1]
    
    if dim_d < dim_p:
        random_projection_dim = dim_p
        xs2 = torch.cat((xs, torch.zeros((xs.shape[0], dim_p - dim_d)).to(device)), dim=1)
        xt2 = xt
    else:
        random_projection_dim = dim_d
        xt2 = torch.cat((xt, torch.zeros((xt.shape[0], dim_d - dim_p)).to(device)), dim=1)
        xs2 = xs
    
    if P is None:
        P = torch.randn(random_projection_dim, nproj)
    p = P / torch.sqrt(torch.sum(P**2, 0, True))
    
    try:
        xsp = torch.matmul(xs2, p.to(device))
        xtp = torch.matmul(xt2, p.to(device))
    except RuntimeError as error:
        print(f'Error: {error}')
        raise BadShapeError
    
    return xsp, xtp
