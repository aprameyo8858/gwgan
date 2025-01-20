#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:39:12 2019

@author: vayer
"""

import torch
import time
import numpy as np
from power_spherical import PowerSpherical
from random import choices


class BadShapeError(Exception):
    pass

def rasgw_gpu(xs, xt, device, L=10, nproj=200, tolog=False, P=None):
    """ Returns SGW between xs and xt eq (4) in [1]. Only implemented with the 0 padding operator Delta.
    Parameters
    ----------
    xs : tensor, shape (n, p)
         Source samples
    xt : tensor, shape (n, q)
         Target samples
    device : torch device
    L : int
         Number of projections (L times)
    nproj : integer
            Number of projections. Ignored if P is provided.
    P : tensor, shape (max(p,q), L)
        Projection matrix. If None, creates a new projection matrix.
    tolog : bool
            Whether to return timings or not.
    Returns
    -------
    C : tensor, shape (L, 1)
           Cost for each projection.
    """
    if tolog:
        log = {}

    # Sink data using projections
    if tolog:
        st = time.time()
        xsp, xtp = sink_(xs, xt, device, L, nproj, P)
        ed = time.time()
        log['time_sink_'] = ed - st
    else:
        xsp, xtp = sink_(xs, xt, device, L, nproj, P)

    # Compute Gromov-Wasserstein cost
    if tolog:
        st = time.time()
        d, log_gw1d = gromov_1d(xsp, xtp, tolog=True)
        ed = time.time()
        log['time_gw_1D'] = ed - st
        log['gw_1d_details'] = log_gw1d
    else:
        d = gromov_1d(xsp, xtp, tolog=False)

    if tolog:
        return d, log
    else:
        return d


def _cost(xsp, xtp, tolog=False):
    """ Returns the GM cost eq (3) in [1].
    Parameters
    ----------
    xsp : tensor, shape (n, L)
         1D sorted samples (after finding sigma opt) for each proj in the source
    xtp : tensor, shape (n, L)
         1D sorted samples (after finding sigma opt) for each proj in the target
    tolog : bool
            Whether to return timings or not
    Returns
    -------
    C : tensor, shape (L, 1)
           Cost for each projection.
    """
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


def gromov_1d(xs, xt, tolog=False):
    """ Solves the Gromov in 1D (eq (2) in [1] for each proj).
    Parameters
    ----------
    xs : tensor, shape (n, L)
         1D sorted samples for each proj in the source
    xt : tensor, shape (n, L)
         1D sorted samples for each proj in the target
    tolog : bool
            Whether to return timings or not
    Returns
    -------
    toreturn : tensor, shape (L, 1)
           The SGW cost for each proj.
    """
    if tolog:
        log = {}

    st = time.time()
    xs2, i_s = torch.sort(xs, dim=0)

    if tolog:
        xt_asc, i_t = torch.sort(xt, dim=0)  # Sort increasing
        xt_desc, i_t = torch.sort(xt, dim=0, descending=True)  # Sort decreasing
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

def sink_(xs, xt, device, L, nproj=None, P=None, kappa=50): 
    """
    Sinks the points of the measure in the lowest dimension onto the highest dimension 
    and applies the projections (through zero padding or a provided projection matrix).
    
    Parameters
    ----------
    xs : tensor, shape (n, p)
        Source samples (n samples, p dimensions).
    xt : tensor, shape (n, q)
        Target samples (n samples, q dimensions).
    device : torch device
        The device on which the tensors will be located (CPU or GPU).
    L : int
        The number of projections (iterations).
    nproj : int, optional
        Number of projections. Ignored if P is provided.
    P : tensor, shape (max(p, q), L), optional
        Projection matrix. If None, random projections are generated.
    kappa : float, optional
        Scaling factor for the spherical distribution (used when generating random projections).

    Returns
    -------
    xsp_batch : tensor, shape (n, L)
        Projected source samples (after applying the projections).
    xtp_batch : tensor, shape (n, L)
        Projected target samples (after applying the projections).
    """
    
    # Get the input dimensionalities of the source and target
    dim_d = xs.shape[1]  # Dimension of the source samples
    dim_p = xt.shape[1]  # Dimension of the target samples
    
    # Zero-padding if the dimensions of the source and target are different
    if dim_d < dim_p:
        # Source samples need padding (zero padding)
        xs2 = torch.cat((xs, torch.zeros((xs.shape[0], dim_p - dim_d)).to(device)), dim=1)
        xt2 = xt
    else:
        # Target samples need padding (zero padding)
        xt2 = torch.cat((xt, torch.zeros((xt.shape[0], dim_d - dim_p)).to(device)), dim=1)
        xs2 = xs
    
    # If no projection matrix is provided, generate random projections
    if P is None:
        # Generate random projections in both source and target spaces
        z_xx = (xs2.detach()[np.random.choice(xs2.shape[0], L, replace=True)] - xs2.detach()[np.random.choice(xs2.shape[0], L, replace=True)])
        z_xx_bar = z_xx / torch.sqrt(torch.sum(z_xx ** 2, dim=1, keepdim=True))  # Normalize

        z_yy = (xt2.detach()[np.random.choice(xt2.shape[0], L, replace=True)] - xt2.detach()[np.random.choice(xt2.shape[0], L, replace=True)])
        z_yy_bar = z_yy / torch.sqrt(torch.sum(z_yy ** 2, dim=1, keepdim=True))  # Normalize

        # Combine the projections
        z_xx_yy = (z_xx_bar + z_yy_bar) / torch.sqrt(torch.sum((z_xx_bar + z_yy_bar) ** 2, dim=1, keepdim=True))
        z_xx_yy_dash = (z_xx_bar - z_yy_bar) / torch.sqrt(torch.sum((z_xx_bar - z_yy_bar) ** 2, dim=1, keepdim=True))

        # Create the final projection matrix (theta) by averaging the two projection directions
        theta = 0.5 * (z_xx_yy + z_xx_yy_dash)
        theta = torch.nan_to_num(theta, nan=0.0)  # Handle NaN values, if any

        # Use a spherical distribution to sample projections
        ps = PowerSpherical(
            loc=theta,
            scale=torch.full((theta.shape[0],), kappa, device=device),
        )
        theta = ps.rsample()

        # Use the transpose of theta to ensure correct dimensions
        P = theta.T

    # Normalize the projection matrix P
    P = P / torch.sqrt(torch.sum(P ** 2, 0, keepdim=True))

    # Apply the projections to the source and target data
    try:
        xsp_batch = torch.matmul(xs2, P.to(device))  # Project the source samples
        xtp_batch = torch.matmul(xt2, P.to(device))  # Project the target samples
    except RuntimeError as error:
        print('----------------------------------------')
        print(f'Error during projection: {error}')
        print(f'xs original dim: {xs.shape}')
        print(f'xt original dim: {xt.shape}')
        print(f'dim_p: {dim_p}, dim_d: {dim_d}')
        print(f'Projection matrix P: {P.shape}')
        print(f'xs2 dim: {xs2.shape}')
        print(f'xt2 dim: {xt2.shape}')
        print('----------------------------------------')
        raise BadShapeError

    return xsp_batch, xtp_batch
