"""
Flow Equations for Many-Body Quantum Systems
S. J. Thomson
Dahlem Centre for Complex Quantum Systems, FU Berlin
steven.thomson@fu-berlin.de
steventhomson.co.uk / @PhysicsSteve
https://orcid.org/0000-0001-9065-9842
---------------------------------------------

This work is licensed under a Creative Commons 
Attribution-NonCommercial-ShareAlike 4.0 International License. This work may
be edited and shared by others, provided the author of this work is credited 
and any future authors also freely share and make their work available. This work
may not be used or modified by authors who do not make any derivative work 
available under the same conditions. This work may not be modified and used for
any commercial purpose, in whole or in part. For further information, see the 
license details at https://creativecommons.org/licenses/by-nc-sa/4.0/.

This work is distributed without any form of warranty or committment to provide technical 
support, nor the guarantee that it is free from bugs or technical incompatibilities
with all possible computer systems. Use, modify and troubleshoot at your own risk.

---------------------------------------------

This file contains all of the code used to construct the RHS of the flow equations using matrix/tensor contractions 
and numerically integrate the flow equation to obtain a diagonal Hamiltonian.

"""

import os,functools,time
try:
    from psutil import cpu_count  # type: ignore
except Exception:
    cpu_count = None  # type: ignore
# Set up threading options for parallel solver
def _safe_num_cores() -> int:
    try:
        if cpu_count is not None:
            n = cpu_count(logical=False)
            if n is None:
                n = cpu_count(logical=True)
            if n:
                return int(n)
    except Exception:
        pass
    return int(os.cpu_count() or 1)

_NCORES = str(_safe_num_cores())
# Respect externally-provided thread settings (e.g. benchmark harness) to avoid oversubscription.
os.environ.setdefault('OMP_NUM_THREADS', _NCORES)       # Set number of OpenMP threads to run in parallel
os.environ.setdefault('MKL_NUM_THREADS', _NCORES)       # Set number of MKL threads to run in parallel
os.environ.setdefault('NUMBA_NUM_THREADS', _NCORES)     # Set number of Numba threads
os.environ['JAX_ENABLE_X64'] = 'false'  
import jax
import jax.numpy as jnp
from jax import jit
from jax import make_jaxpr
from jax.lax import fori_loop
import numpy as np
# from diffrax import diffeqsolve, ODETerm, Dopri5
# JAX compatibility: host_callback was removed in newer JAX versions.
try:
    from jax.experimental.host_callback import id_print  # type: ignore
except Exception:
    try:
        from jax import debug as _jax_debug  # type: ignore

        def id_print(x, what=None, **kwargs):  # type: ignore
            _jax_debug.print("{w}{x}", w=(("" if what is None else str(what) + ": ")), x=x)
            return x

    except Exception:

        def id_print(x, what=None, **kwargs):  # type: ignore
            return x
import jax.random as jr
# from jax.lax import dynamic_slice as slice
# from jax.config import config
# config.update("jax_enable_x64", True)
from datetime import datetime
# Dynamics (QuSpin) is optional. Import lazily so flow-only runs don't require QuSpin.
try:
    from ..dynamics import dyn_con, dyn_exact, dyn_itc  # type: ignore
except Exception:
    dyn_con = None
    dyn_exact = None
    dyn_itc = None
# from numba import jit,prange
import gc,copy
from ..contract import contract, contractNO, no_helper
from ..utility import nstate, state_spinless, indices
from  jax.experimental.ode import odeint as ode
from scipy.integrate import ode as ode_np
#import matplotlib.pyplot as plt
from jax.numpy.linalg import norm as frn
from ..memlog import memlog

# Optional memory logging (disabled by default).
def _mem_every() -> int:
    try:
        return int(os.environ.get("PYFLOW_MEMLOG_EVERY", "0"))
    except Exception:
        return 0


def _memlog(tag: str, step: int | None = None, l: float | None = None, **fields):
    every = _mem_every()
    if every <= 0:
        return
    if step is not None and (step % every != 0):
        return
    memlog(tag, step=step, l=l, **fields)

#
# ==============================================================================
#  Optional approximate speed knobs (opt-in via env vars)
# ------------------------------------------------------------------------------
#  Goal: speed up large-L runs when accuracy is not important.
#  Default behavior is unchanged unless you set these env vars.
#
#  - PYFLOW_H4_UPDATE_EVERY = k (default 1)
#      Recompute expensive dH4/dl only every k steps; reuse cached dH4 in between.
#
#  - PYFLOW_SKIP_SMALL_TERMS = 1 (default 0) with PYFLOW_SKIP_EPS
#      If max(|H4|) < eps, skip dH4 computation entirely (set dH4=0).
#
#  Notes:
#  - These are approximate / non-ODE-solver updates and WILL change results.
#  - We only enable the approximate stepper when norm=False and Hflow=True.
# ==============================================================================
def _approx_h4_update_every() -> int:
    try:
        k = int(os.environ.get("PYFLOW_H4_UPDATE_EVERY", "1"))
        return max(1, k)
    except Exception:
        return 1


def _approx_skip_small_terms() -> bool:
    return os.environ.get("PYFLOW_SKIP_SMALL_TERMS", "0") in ("1", "true", "True")


def _approx_skip_eps(default: float = 0.0) -> float:
    try:
        return float(os.environ.get("PYFLOW_SKIP_EPS", str(default)))
    except Exception:
        return default


def _approx_enabled() -> bool:
    return (_approx_h4_update_every() > 1) or _approx_skip_small_terms()

def _force_steps() -> int | None:
    """
    If set, force the forward integration to run exactly this many steps
    (ignoring cutoff/accuracy). Use with care: results will generally be unphysical.
    """
    v = os.environ.get("PYFLOW_FORCE_STEPS", "").strip()
    if not v:
        return None
    try:
        n = int(v)
        return n if n > 0 else None
    except Exception:
        return None


def _approx_dh2_only(H2, method: str):
    """Cheap dH2/dl using only quadratic terms (no H4 contractions)."""
    H2_0 = jnp.diag(jnp.diag(H2))
    V0 = H2 - H2_0
    eta0 = contract(H2_0, V0, method=method, eta=True)
    return contract(eta0, H2, method=method)


def _approx_step_h2_h4(
    H2,
    H4,
    l0: float,
    l1: float,
    step_idx: int,
    method: str,
    *,
    cache: dict,
    norm: bool,
    Hflow: bool,
):
    """
    Approximate single-step update (explicit Euler):
      H(l1) = H(l0) + dl * dH/dl|_{l0}

    - dH2 is always recomputed cheaply from H2 only.
    - dH4 is recomputed every k steps; otherwise reuses cached dH4.
    - Optional skip: if max(|H4|) < eps, set dH4=0.
    """
    dl = float(l1 - l0)
    dH2 = _approx_dh2_only(H2, method=method)

    # Only enable approximations for the typical fast path
    if norm or (not Hflow):
        # Fallback to exact integrator call for this step
        # Keep ODE tolerances consistent across modes (original / ckpt / recursive / hybrid)
        _rtol = float(os.environ.get('PYFLOW_ODE_RTOL', '1e-6'))
        _atol = float(os.environ.get('PYFLOW_ODE_ATOL', '1e-6'))
        steps = np.linspace(l0, l1, num=2, endpoint=True)
        soln = ode(int_ode, [H2, H4], steps, rtol=_rtol, atol=_atol)
        return soln[0][-1], soln[1][-1]

    k_every = _approx_h4_update_every()
    do_skip = _approx_skip_small_terms()
    eps = _approx_skip_eps(default=0.0)

    need_dh4 = ("dH4" not in cache) or (k_every <= 1) or (step_idx % k_every == 0)

    if do_skip and eps > 0.0:
        # If H4 is effectively zero, skip the expensive dH4 contraction.
        # This O(n^4) reduction is typically much cheaper than the contractions.
        if float(jnp.max(jnp.abs(H4))) < eps:
            cache["dH4"] = jnp.zeros_like(H4)
            need_dh4 = False

    if need_dh4:
        # Recompute full RHS and cache only dH4; dH2 is overwritten by the cheap one above.
        _dH2_full, dH4 = int_ode([H2, H4], l0, method=method, norm=False, Hflow=True)
        cache["dH4"] = dH4
    else:
        dH4 = cache["dH4"]

    return H2 + dl * dH2, H4 + dl * dH4


# ------------------------------------------------------------------------------
# JIT-compiled block runner for approximate stepping (removes Python-per-step overhead)
# ------------------------------------------------------------------------------
_approx_block_cache = {}


def _approx_run_block(
    H2,
    H4,
    dl_list,
    t_start_idx: int,
    t_end_idx: int,
    *,
    method: str,
):
    """
    Run approximate stepping for k in [t_start_idx, t_end_idx) and return trajectories:
      traj2.shape = (block_len+1, n, n)
      traj4.shape = (block_len+1, n, n, n, n)

    Only used when _approx_enabled() and norm=False (call-site enforces).
    """
    block_len = int(t_end_idx - t_start_idx)
    if block_len <= 0:
        return H2[None, ...], H4[None, ...]

    k_every = _approx_h4_update_every()
    do_skip = _approx_skip_small_terms()
    eps = _approx_skip_eps(default=0.0)

    n = int(H2.shape[0])
    cache_key = (n, method, int(k_every), bool(do_skip), float(eps))
    fn = _approx_block_cache.get(cache_key)

    if fn is None:
        from functools import partial
        from jax.lax import scan as _scan

        @partial(jit, static_argnames=("method", "k_every", "do_skip", "eps"))
        def _run(H2_0, H4_0, t0s, t1s, start_idx, *, method: str, k_every: int, do_skip: bool, eps: float):
            def body(carry, tpair):
                H2c, H4c, dH4_cache, has_cache, idx = carry
                t0, t1 = tpair
                # Keep everything in the same dtype as the carried state to satisfy lax.scan invariants.
                dl = jnp.asarray(t1 - t0, dtype=H2c.dtype)
                dH2 = jnp.asarray(_approx_dh2_only(H2c, method=method), dtype=H2c.dtype)

                need = jnp.logical_or(jnp.logical_not(has_cache), jnp.logical_or(k_every <= 1, (idx % k_every) == 0))

                # `do_skip` and `eps` are static (compile-time constants); keep this as Python branch.
                if do_skip and eps > 0.0:
                    small = jnp.max(jnp.abs(H4c)) < eps
                    dH4_new = jnp.where(small, jnp.zeros_like(H4c), dH4_cache)
                    has_new = jnp.logical_or(has_cache, small)
                    need = jnp.logical_and(need, jnp.logical_not(small))
                else:
                    dH4_new = dH4_cache
                    has_new = has_cache

                def recompute(_):
                    _dH2_full, dH4 = int_ode([H2c, H4c], t0, method=method, norm=False, Hflow=True)
                    # Ensure dtype matches cached branch to satisfy lax.cond requirements
                    return jnp.asarray(dH4, dtype=H4c.dtype)

                dH4 = jax.lax.cond(need, recompute, lambda _: dH4_new, operand=None)
                has_new = jnp.logical_or(has_new, need)

                H2n = jnp.asarray(H2c + dl * dH2, dtype=H2c.dtype)
                H4n = jnp.asarray(H4c + dl * dH4, dtype=H4c.dtype)
                return (H2n, H4n, dH4, has_new, idx + 1), (H2n, H4n)

            init = (H2_0, H4_0, jnp.zeros_like(H4_0), jnp.array(False), jnp.array(start_idx))
            (H2f, H4f, _, _, _), (H2s, H4s) = _scan(body, init, (t0s, t1s))
            traj2 = jnp.concatenate([H2_0[None, ...], H2s], axis=0)
            traj4 = jnp.concatenate([H4_0[None, ...], H4s], axis=0)
            return traj2, traj4

        fn = _run
        _approx_block_cache[cache_key] = fn

    t0s = jnp.array(dl_list[t_start_idx:t_end_idx])
    t1s = jnp.array(dl_list[t_start_idx + 1:t_end_idx + 1])
    traj2, traj4 = fn(H2, H4, t0s, t1s, t_start_idx, method=method, k_every=k_every, do_skip=do_skip, eps=eps)
    return traj2, traj4


#------------------------------------------------------------------------------ 

#def frn(mat):
#    mat = jnp.array(mat)
#    n = len(mat.flatten())
#    return frn2(mat)/n

# @jit(nopython=True,parallel=True,fastmath=True,cache=True)
def cut(y,n,cutoff,indices):
    """ Checks if ALL quadratic off-diagonal parts have decayed below cutoff*10e-3 and TYPICAL (median) off-diag quartic term have decayed below cutoff. """
    mat2 = y[:n**2].reshape(n,n)
    mat2_od = mat2-jnp.diag(jnp.diag(mat2))

    if jnp.max(jnp.abs(mat2_od)) < cutoff*10**(-3):
        mat4 = y[n**2:n**2+n**4]
        mat4_od = jnp.zeros(n**4)
        for i in indices:               
            mat4_od = mat4_od.at[i].set(mat4[i])
        mat4_od = mat4_od[mat4_od != 0]
        if jnp.median(jnp.abs(mat4_od)) < cutoff:
            return 0 
        else:
            return 1
    else:
        return 1

def nonint_ode(H,l,method='einsum'):
    """ Generate the flow equation for non-interacting systems.

        e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

        Parameters
        ----------
        l : float
            The (fictitious) flow time l which parameterises the unitary transform.
        y : array
            Array of size n**2 containing all coefficients of the running Hamiltonian at flow time l.
        n : integer
            Linear system size.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.

            The first two are built-in NumPy methods, while the latter two are custom coded for speed.

        Returns
        -------
        sol : RHS of the flow equation

    """

    H0 = jnp.diag(jnp.diag(H))
    V0 = H - H0
    eta = contract(H0,V0,method=method,eta=True)
    sol = contract(eta,H,method=method,eta=False)

    return sol

#------------------------------------------------------------------------------
# Build the generator eta at each flow time step
def eta_con(y,n,method='jit',norm=False):
    """ Generates the generator at each flow time step. 
    
        Parameters
        ----------

        y : array
            Running Hamiltonian used to build generator.
        n : integer
            Linear system size
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.

    """

    # Extract quadratic parts of Hamiltonian from array y
    H = y[0:n**2]
    H = H.reshape(n,n)
    H0 = jnp.diag(jnp.diag(H))
    V0 = H - H0

    # Extract quartic parts of Hamiltonian from array y
    Hint = y[n**2::]
    Hint = Hint.reshape(n,n,n,n)
    Hint0 = jnp.zeros((n,n,n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                Hint0[i,i,j,j] = Hint[i,i,j,j]
                Hint0[i,j,j,i] = Hint[i,j,j,i]
    Vint = Hint-Hint0

    # Compute quadratic part of generator
    eta2 = contract(H0,V0,method=method,eta=True)

    # Compute quartic part of generator
    eta4 = contract(Hint0,V0,method=method,eta=True) + contract(H0,Vint,method=method,eta=True)

    # Add normal-ordering corrections into generator eta, if norm == True
    if norm == True:
        state=nstate(n,0.5)

        eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H0,Vint,method=method,eta=True,state=state)
        eta2 += eta_no2
        eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
        eta4 += eta_no4

    # Combine into array
    eta = jnp.zeros(n**2+n**4)
    eta[:n**2] = eta2.reshape(n**2)
    eta[n**2:] = eta4.reshape(n**4)

    return eta

# @functools.lru_cache(maxsize=None)
def ex_helper(n):
    test = np.zeros((n,n,n,n),dtype=np.int8)
    for i in range(n):
        for j in range(n):
            test[i,i,j,j] = 1
            test[i,j,j,i] = 1

    return jnp.array(test.reshape(n**4))

# @functools.lru_cache(maxsize=None)
def ex_helper2(n):
    test = np.zeros((n,n,n,n,n,n),dtype=np.int8)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                test[i,i,j,j,k,k] = 1
                test[i,i,j,k,k,j] = 1
                test[i,j,j,i,k,k] = 1
                test[i,j,k,i,j,k] = 1
                test[i,j,j,k,k,i] = 1
                test[i,j,k,k,j,i] = 1

    return jnp.array(test.reshape(n**6))

# @functools.lru_cache(maxsize=None)
def ah_helper(n):
    test = np.zeros((n,n,n,n),dtype=np.int8)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for q in range(n):
                    if test[i,j,k,q] == 0:
                        test[i,j,k,q] =  1
                        test[q,k,j,i] = -1

    return jnp.array(test.reshape(n**4))

# def aH(A):
#     n,_,_,_ = A.shape
#     mask =ah_helper(n)
#     Ah = jnp.multiply(mask,A.reshape(n**4))
#     return Ah.reshape(n,n,n,n)

def extract_diag(H2,Hint):

    n,_ = H2.shape
    H2_0 = jnp.diag(jnp.diag(H2))       # Define diagonal quadratic part H0
    V0 = H2 - H2_0                      # Define off-diagonal quadratic part

    test = ex_helper(n)

    Hint0 = jnp.multiply(test,Hint.reshape(n**4))
    Hint0 = Hint0.reshape(n,n,n,n)
    Vint = Hint-Hint0

    return H2_0,V0,Hint0,Vint

def extract_diag2(H2,Hint,H6):

    n,_ = H2.shape
    H2_0 = jnp.diag(jnp.diag(H2))       # Define diagonal quadratic part H0
    V0 = H2 - H2_0                      # Define off-diagonal quadratic part

    test = ex_helper(n)
    test2 = ex_helper2(n)

    Hint0 = jnp.multiply(test,Hint.reshape(n**4))
    Hint0 = Hint0.reshape(n,n,n,n)
    Vint = Hint-Hint0

    H6_0 = jnp.multiply(test2,H6.reshape(n**6))
    H6_0 = H6_0.reshape(n,n,n,n,n,n)
    V6 = H6-H6_0

    return H2_0,V0,Hint0,Vint,H6_0,V6

#------------------------------------------------------------------------------

# def extract_diag(A,norm=False):
#     B = jnp.zeros(A.shape)
#     n,_,_,_ = A.shape
#     for i in range(n): 
#         for j in range(n):
#             if i != j:
#                 if norm == True:
#                     # Symmetrise (for normal-ordering wrt inhomogeneous states)
#                     A[i,i,j,j] += -A[i,j,j,i]
#                     A[i,j,j,i] = 0.
#             if i != j:
#                 if norm == True:
#                     # Symmetrise (for normal-ordering wrt inhomogeneous states)
#                     A[i,i,j,j] += A[j,j,i,i]
#                     A[i,i,j,j] *= 0.5
#                 # Load new array with diagonal values
#                 B[i,i,j,j] = A[i,i,j,j]
#     return B,A


#------------------------------------------------------------------------------
# JIT-accelerated ODE function (used when USE_JIT_FLOW=1)
#------------------------------------------------------------------------------

# Check if JIT acceleration is enabled
_USE_JIT_FLOW = os.environ.get("USE_JIT_FLOW", "0") in ("1", "true", "True")

# Cache for JIT-compiled ODE functions (keyed by system size n)
_jit_ode_cache = {}

def _get_jit_ode(n):
    """Get or create a JIT-compiled ODE function for system size n."""
    if n not in _jit_ode_cache:
        # Pre-compute masks for this system size
        _ex_mask = ex_helper(n)
        _no_mask = no_helper(n)
        
        @jit
        def _int_ode_jit_inner(y, l):
            """JIT-compiled flow equation (norm=False, Hflow=True case)."""
            H2 = y[0]
            Hint = y[1]
            
            # Extract diagonal and off-diagonal parts
            H2_0 = jnp.diag(jnp.diag(H2))
            V0 = H2 - H2_0
            
            Hint0 = jnp.multiply(_ex_mask, Hint.reshape(n**4)).reshape(n, n, n, n)
            Vint = Hint - Hint0
            
            # Compute generator eta = [H0, V] (Wegner generator)
            # [H2_0, V0] = H2_0 @ V0 - V0 @ H2_0
            eta0 = jnp.einsum('ij,jk->ik', H2_0, V0, optimize=True) - jnp.einsum('ki,ij->kj', V0, H2_0, optimize=True)
            
            # [Hint0, V0] - rank-4 with rank-2 contraction
            eta_int_1 = (jnp.einsum('abcd,df->abcf', Hint0, V0, optimize=True)
                       - jnp.einsum('abcd,ec->abed', Hint0, V0, optimize=True)
                       + jnp.einsum('abcd,bf->afcd', Hint0, V0, optimize=True)
                       - jnp.einsum('abcd,ea->ebcd', Hint0, V0, optimize=True))
            
            # [H2_0, Vint] = -[Vint, H2_0]
            eta_int_2 = -(jnp.einsum('abcd,df->abcf', Vint, H2_0, optimize=True)
                        - jnp.einsum('abcd,ec->abed', Vint, H2_0, optimize=True)
                        + jnp.einsum('abcd,bf->afcd', Vint, H2_0, optimize=True)
                        - jnp.einsum('abcd,ea->ebcd', Vint, H2_0, optimize=True))
            
            eta_int = eta_int_1 + eta_int_2
            eta_int = jnp.multiply(_no_mask, eta_int.reshape(n**4)).reshape(n, n, n, n)
            
            # Compute flow equation RHS: dH/dl = [eta, H]
            # [eta0, H2]
            sol = jnp.einsum('ij,jk->ik', eta0, H2, optimize=True) - jnp.einsum('ki,ij->kj', H2, eta0, optimize=True)
            
            # [eta_int, H2]
            sol2_1 = (jnp.einsum('abcd,df->abcf', eta_int, H2, optimize=True)
                    - jnp.einsum('abcd,ec->abed', eta_int, H2, optimize=True)
                    + jnp.einsum('abcd,bf->afcd', eta_int, H2, optimize=True)
                    - jnp.einsum('abcd,ea->ebcd', eta_int, H2, optimize=True))
            
            # [eta0, Hint]
            sol2_2 = -(jnp.einsum('abcd,df->abcf', Hint, eta0, optimize=True)
                     - jnp.einsum('abcd,ec->abed', Hint, eta0, optimize=True)
                     + jnp.einsum('abcd,bf->afcd', Hint, eta0, optimize=True)
                     - jnp.einsum('abcd,ea->ebcd', Hint, eta0, optimize=True))
            
            sol2 = sol2_1 + sol2_2
            sol2 = jnp.multiply(_no_mask, sol2.reshape(n**4)).reshape(n, n, n, n)
            
            return [sol, sol2]
        
        _jit_ode_cache[n] = _int_ode_jit_inner
    
    return _jit_ode_cache[n]


def int_ode(y,l,eta=[],method='einsum',norm=False,Hflow=True):
        """ Generate the flow equation for the interacting systems.

        e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

        Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
        the ijnput array eta will be used to specify the generator at this flow time step. The latter option will result 
        in a huge speed increase, at the potential cost of accuracy. This is because the SciPy routine used to 
        integrate the ODEs will sometimes add intermediate steps: recomputing eta on the fly will result in these 
        steps being computed accurately, while fixing eta will avoid having to recompute the generator every time an 
        interpolation step is added (leading to a speed increase), however it will mean that the generator evaluated at 
        these intermediate steps has errors of order <dl (where dl is the flow time step). For sufficiently small dl, 
        the benefits from the speed increase likely outweigh the decrease in accuracy.

        Parameters
        ----------
        l : float
            The (fictitious) flow time l which parameterises the unitary transform.
        y : array
            Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
        n : integer
            Linear system size.
        eta : array, optional
            Provide a pre-computed generator, if desired.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        sol0 : RHS of the flow equation for interacting system.

        """
        # Use JIT-accelerated version if enabled and conditions match
        if _USE_JIT_FLOW and norm == False and Hflow == True:
            n = y[0].shape[0]
            jit_ode = _get_jit_ode(n)
            return jit_ode(y, l)
        
        # Original implementation
        # print('y shape', y.shape)
        # Extract various components of the Hamiltonian from the ijnput array 'y'
        # id_print(y)
        # id_print(l)
        H2 = y[0]                           # Define quadratic part of Hamiltonian
        n,_ = H2.shape
        # H2_0 = jnp.diag(jnp.diag(H2))       # Define diagonal quadratic part H0
        # V0 = H2 - H2_0                      # Define off-diagonal quadratic part

        Hint = y[1]                         # Define quartic part of Hamiltonian
        # Hint0 = jnp.zeros((n,n,n,n))        # Define diagonal quartic part 
        # for i in range(n):                  # Load Hint0 with values
        #     for j in range(n):
        #             Hint0 = Hint0.at[i,i,j,j].set(Hint[i,i,j,j])
        #             Hint0 = Hint0.at[i,j,j,i].set(Hint[i,j,j,i])
        # Vint = Hint-Hint0
        # id_print(H2)

        H2_0,V0,Hint0,Vint = extract_diag(H2,Hint)

        if norm == True:
            state = state_spinless(H2)

        if Hflow == True:
            # Compute the generator eta
            eta0 = contract(H2_0,V0,method=method,eta=True)
            eta_int = contract(Hint0,V0,method=method,eta=True) + contract(H2_0,Vint,method=method,eta=True)

            # Add normal-ordering corrections into generator eta, if norm == True
            if norm == True:

                eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H2_0,Vint,method=method,eta=True,state=state)
                eta0 += eta_no2

                eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
                eta_int += eta_no4
        else:
            eta0 = (eta[:n**2]).reshape(n,n)
            eta_int = (eta[n**2:]).reshape(n,n,n,n)

        # id_print(eta0)
   
        # Compute the RHS of the flow equation dH/dl = [\eta,H]
        sol = contract(eta0,H2,method=method)
        sol2 = contract(eta_int,H2,method=method) + contract(eta0,Hint,method=method)

        # Add normal-ordering corrections into flow equation, if norm == True
        if norm == True:
            sol_no = contractNO(eta_int,H2,method=method,eta=False,state=state) + contractNO(eta0,Hint,method=method,eta=False,state=state)
            sol4_no = contractNO(eta_int,Hint,method=method,eta=False,state=state)
            sol+=sol_no
            sol2 += sol4_no

        # id_print([sol,sol2])
        return [sol,sol2]


def int_ode_ITC(y,l,eta=[],method='einsum',norm=False,Hflow=True):
        """ Generate the flow equation for the interacting systems.

        e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

        Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
        the ijnput array eta will be used to specify the generator at this flow time step. The latter option will result 
        in a huge speed increase, at the potential cost of accuracy. This is because the SciPy routine used to 
        integrate the ODEs will sometimes add intermediate steps: recomputing eta on the fly will result in these 
        steps being computed accurately, while fixing eta will avoid having to recompute the generator every time an 
        interpolation step is added (leading to a speed increase), however it will mean that the generator evaluated at 
        these intermediate steps has errors of order <dl (where dl is the flow time step). For sufficiently small dl, 
        the benefits from the speed increase likely outweigh the decrease in accuracy.

        Parameters
        ----------
        l : float
            The (fictitious) flow time l which parameterises the unitary transform.
        y : array
            Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
        n : integer
            Linear system size.
        eta : array, optional
            Provide a pre-computed generator, if desired.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        sol0 : RHS of the flow equation for interacting system.

        """

        H2 = y[0]                           # Define quadratic part of Hamiltonian
        n,_ = H2.shape
        Hint = y[1]                         # Define quartic part of Hamiltonian
        H2_0,V0,Hint0,Vint = extract_diag(H2,Hint)
        n2 = y[2]
        n4 = y[3]

        if norm == True:
            state = state_spinless(H2)

        if Hflow == True:
            # Compute the generator eta
            eta0 = contract(H2_0,V0,method=method,eta=True)
            eta_int = contract(Hint0,V0,method=method,eta=True) + contract(H2_0,Vint,method=method,eta=True)

            # Add normal-ordering corrections into generator eta, if norm == True
            if norm == True:

                eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H2_0,Vint,method=method,eta=True,state=state)
                eta0 += eta_no2

                eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
                eta_int += eta_no4
        else:
            eta0 = (eta[:n**2]).reshape(n,n)
            eta_int = (eta[n**2:]).reshape(n,n,n,n)

        # id_print(eta0)
   
        # Compute the RHS of the flow equation dH/dl = [\eta,H]
        sol = contract(eta0,H2,method=method)
        sol2 = contract(eta_int,H2,method=method) + contract(eta0,Hint,method=method)

        # Compute the RHS of the flow equation dH/dl = [\eta,H]
        sol_n2 = contract(eta0,n2,method=method)
        sol_n4 = contract(eta_int,n2,method=method) + contract(eta0,n4,method=method)

        # Add normal-ordering corrections into flow equation, if norm == True
        if norm == True:
            sol_no = contractNO(eta_int,H2,method=method,eta=False,state=state) + contractNO(eta0,Hint,method=method,eta=False,state=state)
            sol4_no = contractNO(eta_int,Hint,method=method,eta=False,state=state)
            sol+=sol_no
            sol2 += sol4_no

            sol_n2_no = contractNO(eta_int,n2,method=method,eta=False,state=state) + contractNO(eta0,n4,method=method,eta=False,state=state)
            sol_n4_no = contractNO(eta_int,n4,method=method,eta=False,state=state)
            sol_n2+=sol_n2_no
            sol_n4 += sol_n4_no

        # id_print([sol,sol2])
        return [sol,sol2,sol_n2,sol_n4]

def res_test(i,j,a,b,c,res_cut,epsilon=0.0):
    rand = np.random.uniform(-0.1,0.1)
    rand = jnp.array(rand)
    return jnp.where(jnp.abs(a)>jnp.abs(res_cut*(b-c)),jnp.sign(i-j)*a*(1+epsilon*rand),0)

def res_test_int(i,j,k,q,a,b,c,res_cut,epsilon=0.0):
    rand = np.random.uniform(-0.1,0.1)
    rand = jnp.array(rand)
    return jnp.where(jnp.abs(a)>jnp.abs(res_cut*(b-c)),jnp.sign(i-j+k-q)*a*(1+epsilon*rand),0)

def sign_helper(n):
    test = np.zeros((n,n,n,n),dtype=np.int8)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for q in range(n):
                    if i != k and j != q and (i != j or k != q):
                        test[i,j,k,q] = np.sign(i-j+k-q)
    return jnp.array(test).reshape(n**4)

def aH(A):
    n,_,_,_ = A.shape
    mask = sign_helper(n)
    eta = jnp.multiply(mask,A.reshape(n**4))
    return eta.reshape(n,n,n,n)

def int_ode_random_O4(y,l,res_cut=0.75,method='einsum',norm=False):
    """ Generate the flow equation for the interacting systems.

    e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

    Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
    the input array eta will be used to specify the generator at this flow time step. The latter option will result 
    in a huge speed increase, at the potential cost of accuracy. This is because the SciPy routine used to 
    integrate the ODEs will sometimes add intermediate steps: recomputing eta on the fly will result in these 
    steps being computed accurately, while fixing eta will avoid having to recompute the generator every time an 
    interpolation step is added (leading to a speed increase), however it will mean that the generator evaluated at 
    these intermediate steps has errors of order <dl (where dl is the flow time step). For sufficiently small dl, 
    the benefits from the speed increase likely outweigh the decrease in accuracy.

    Parameters
    ----------
    l : float
        The (fictitious) flow time l which parameterises the unitary transform.
    y : array
        Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
    n : integer
        Linear system size.
    eta : array, optional
        Provide a pre-computed generator, if desired.
    method : string, optional
        Specify which method to use to generate the RHS of the flow equations.
        Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
        The first two are built-in NumPy methods, while the latter two are custom coded for speed.
    norm : bool, optional
        Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
        This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
        ensure that use of normal-ordering is warranted and that the contractions are computed with 
        respect to an appropriate state.
    Hflow : bool, optional
        Choose whether to use pre-computed generator or re-compute eta on the fly.

    Returns
    -------
    sol0 : RHS of the flow equation for interacting system.

    """

    H2 = y[0]                           # Define quadratic part of Hamiltonian
    n,_ = H2.shape
    Hint = y[1]                         # Define quartic part of Hamiltonian
    c1 = y[2]
    c3 = y[3]
    #_,_,_,Vint = extract_diag(H2,Hint)

    eta0 = jnp.zeros((n,n))
    # eta_int = jnp.zeros((n,n,n,n))
    # eta_int = aH(Vint)

    for i in range(n):
        for j in range(n):
            if i > j:
                result = res_test(i,j,H2[i,j],H2[i,i],H2[j,j],res_cut)
                eta0 = eta0.at[i,j].set(result)
                eta0 = eta0.at[j,i].set(-1*result)
 
            # for k in range(n):
            #     for q in range(n):
            #         if i != k and j != q and (i != j or k != q):
            #             result = res_test_int(i,j,k,q,Hint[i,j,k,q],Hint[i,i,k,k],Hint[j,j,q,q],.5)
            #             eta_int = eta_int.at[i,j,k,q].set(result)
                        # eta_int = eta_int.at[q,k,j,i].set(-1*result)
    
    # Compute the RHS of the flow equation dH/dl = [\eta,H]
    sol = contract(eta0,H2,method=method)
    # sol2 = contract(eta_int,H2,method=method) + contract(eta0,Hint,method=method)
    sol2 = contract(eta0,Hint,method=method)

    # Compute the RHS of the flow equation dc/dl = [\eta,c]
    sol_c1 = contract(eta0,c1,method=method)
    # sol_c3 = contract(eta_int,c1,method=method) + contract(eta0,c3,method=method)
    sol_c3 = contract(eta0,c3,method=method)
    
    return [sol,sol2,sol_c1,sol_c3]

def int_ode_random_O6(y,l,res_cut=0.75,method='einsum',norm=False):
    """ Generate the flow equation for the interacting systems.

    e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

    Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
    the input array eta will be used to specify the generator at this flow time step. The latter option will result 
    in a huge speed increase, at the potential cost of accuracy. This is because the SciPy routine used to 
    integrate the ODEs will sometimes add intermediate steps: recomputing eta on the fly will result in these 
    steps being computed accurately, while fixing eta will avoid having to recompute the generator every time an 
    interpolation step is added (leading to a speed increase), however it will mean that the generator evaluated at 
    these intermediate steps has errors of order <dl (where dl is the flow time step). For sufficiently small dl, 
    the benefits from the speed increase likely outweigh the decrease in accuracy.

    Parameters
    ----------
    l : float
        The (fictitious) flow time l which parameterises the unitary transform.
    y : array
        Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
    n : integer
        Linear system size.
    eta : array, optional
        Provide a pre-computed generator, if desired.
    method : string, optional
        Specify which method to use to generate the RHS of the flow equations.
        Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
        The first two are built-in NumPy methods, while the latter two are custom coded for speed.
    norm : bool, optional
        Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
        This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
        ensure that use of normal-ordering is warranted and that the contractions are computed with 
        respect to an appropriate state.
    Hflow : bool, optional
        Choose whether to use pre-computed generator or re-compute eta on the fly.

    Returns
    -------
    sol0 : RHS of the flow equation for interacting system.

    """

    H2 = y[0]                           # Define quadratic part of Hamiltonian
    n,_ = H2.shape
    Hint = y[1]                         # Define quartic part of Hamiltonian
    c1 = y[2]
    c3 = y[3]
    H6 = y[4]
    c5 = y[5]

    eta0 = jnp.zeros((n,n))
    for i in range(n):
        for j in range(i):
            result = res_test(i,j,H2[i,j],H2[i,i],H2[j,j],res_cut)
            eta0 = eta0.at[i,j].set(result)
            eta0 = eta0.at[j,i].set(-1*result)
    #eta_int = jnp.zeros((n,n,n,n))

    # _,_,_,Vint = extract_diag(H2,Hint)
    # eta_int = aH(Vint)

    # Compute the RHS of the flow equation dH/dl = [\eta,H]
    sol =  contract(eta0,H2,method=method)
    sol2 = contract(eta0,Hint,method=method)  #+ contract(eta_int,H2,method=method) 
    sol3 = contract(eta0,H6)

    # Compute the RHS of the flow equation dH/dl = [\eta,H]
    sol_c1 = contract(eta0,c1,method=method)
    sol_c3 = contract(eta0,c3,method=method) #+ contract(eta_int,c1,method=method)
    sol_c5 = contract(eta0,c5,method=method)
    
    return [sol,sol2,sol_c1,sol_c3,sol3,sol_c5]

def int_ode_toda(l,y,eta=[],method='einsum',norm=False,order=4):
    """ Generate the flow equation for the interacting systems.

    e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

    Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
    the ijnput array eta will be used to specify the generator at this flow time step. The latter option will result 
    in a huge speed increase, at the potential cost of accuracy. This is because the SciPy routine used to 
    integrate the ODEs will sometimes add intermediate steps: recomputing eta on the fly will result in these 
    steps being computed accurately, while fixing eta will avoid having to recompute the generator every time an 
    interpolation step is added (leading to a speed increase), however it will mean that the generator evaluated at 
    these intermediate steps has errors of order <dl (where dl is the flow time step). For sufficiently small dl, 
    the benefits from the speed increase likely outweigh the decrease in accuracy.

    Parameters
    ----------
    l : float
        The (fictitious) flow time l which parameterises the unitary transform.
    y : array
        Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
    n : integer
        Linear system size.
    eta : array, optional
        Provide a pre-computed generator, if desired.
    method : string, optional
        Specify which method to use to generate the RHS of the flow equations.
        Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
        The first two are built-in NumPy methods, while the latter two are custom coded for speed.
    norm : bool, optional
        Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
        This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
        ensure that use of normal-ordering is warranted and that the contractions are computed with 
        respect to an appropriate state.
    Hflow : bool, optional
        Choose whether to use pre-computed generator or re-compute eta on the fly.

    Returns
    -------
    sol0 : RHS of the flow equation for interacting system.

    """

    H2 = y                           # Define quadratic part of Hamiltonian
    n = int(np.sqrt(len(H2)//2))
    H2 = H2[0:n**2].reshape(n,n)
 
    eta0 = np.zeros((n,n))
    eta1 = np.zeros((n,n))
    reslist = res(n,H2)

    epsilon = 0.0
    for r in reslist:
        print(r)
        eta0[r[0],r[1]] = np.sign(r[0]-r[1])*r[2] #*(1+epsilon*np.random.uniform(-0.1,0.1))
        eta0[r[1],r[0]] = -eta0[r[0],r[1]]
    eta0 = np.array(eta0)

    # Compute the RHS of the flow equation dH/dl = [\eta,H]
    sol = contract(eta0,H2,method=method)

    output = np.zeros(2*n**2)
    output[0:n**2] = sol.reshape(n**2)
    output[n**2:] = eta0.reshape(n**2)
    return output

def int_ode_ladder(y,l,eta=[],method='einsum',norm=False,Hflow=True,order=4):
    """ Generate the flow equation for the interacting systems.

    e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

    Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
    the ijnput array eta will be used to specify the generator at this flow time step. The latter option will result 
    in a huge speed increase, at the potential cost of accuracy. This is because the SciPy routine used to 
    integrate the ODEs will sometimes add intermediate steps: recomputing eta on the fly will result in these 
    steps being computed accurately, while fixing eta will avoid having to recompute the generator every time an 
    interpolation step is added (leading to a speed increase), however it will mean that the generator evaluated at 
    these intermediate steps has errors of order <dl (where dl is the flow time step). For sufficiently small dl, 
    the benefits from the speed increase likely outweigh the decrease in accuracy.

    Parameters
    ----------
    l : float
        The (fictitious) flow time l which parameterises the unitary transform.
    y : array
        Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
    n : integer
        Linear system size.
    eta : array, optional
        Provide a pre-computed generator, if desired.
    method : string, optional
        Specify which method to use to generate the RHS of the flow equations.
        Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
        The first two are built-in NumPy methods, while the latter two are custom coded for speed.
    norm : bool, optional
        Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
        This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
        ensure that use of normal-ordering is warranted and that the contractions are computed with 
        respect to an appropriate state.
    Hflow : bool, optional
        Choose whether to use pre-computed generator or re-compute eta on the fly.

    Returns
    -------
    sol0 : RHS of the flow equation for interacting system.

    """
    if len(y) == 6:
        order = 4
    elif len(y) == 8:
        order = 6

    H2 = y[0]                           # Define quadratic part of Hamiltonian
    n,_ = H2.shape
    Hint = y[1]                         # Define quartic part of Hamiltonian

    if order == 4:
        H2_0,V0,Hint0,Vint = extract_diag(H2,Hint)
        c1 = y[2]
        c3 = y[3]
    elif order == 6:
        H6 = y[2]
        H2_0,V0,Hint0,Vint,H6_0,V6 = extract_diag2(H2,Hint,H6)
        c1 = y[3]
        c3 = y[4]
        c5 = y[5]

    if norm == True:
        state = state_spinless(H2)

    if Hflow == True:
        # Compute the generator eta
        eta0 = contract(H2_0,V0,method=method,eta=True)
        eta_int = contract(Hint0,V0,method=method,eta=True) + contract(H2_0,Vint,method=method,eta=True)

        # Add normal-ordering corrections into generator eta, if norm == True
        if norm == True:

            eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H2_0,Vint,method=method,eta=True,state=state)
            eta0 += eta_no2

            eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
            eta_int += eta_no4

        if order == 6:
            eta_6 =  contract(Hint0,Vint,method=method,eta=True) 
            eta_6 += contract(H2_0,V6)
            eta_6 += contract(H6_0,V0)

    else:
        eta0 = (eta[:n**2]).reshape(n,n)
        eta_int = (eta[n**2:]).reshape(n,n,n,n)

    # for i in range(n):
    #     for j in range(n):
    #         for k in range(n):
    #             for q in range(n):
    #                 for l in range(n):
    #                     for m in range(n):
    #                             id_print(100000000000000000)
    #                             id_print(eta_6[i,j,k,q,l,m])
    #                             id_print(eta_6[m,l,q,k,j,i])

    # Compute the RHS of the flow equation dH/dl = [\eta,H]
    sol = contract(eta0,H2,method=method)
    sol2 = contract(eta_int,H2,method=method) + contract(eta0,Hint,method=method)
    if order == 6:
        sol3 = contract(eta_int,Hint,method=method,eta=False) 
        sol3 += contract(eta_6,H2,method=method,eta=False)
        sol3 += contract(eta0,H6,method=method,eta=False)

    # Compute the RHS of the flow equation dH/dl = [\eta,H]
    sol_c1 = contract(eta0,c1,method=method)
    sol_c3 = contract(eta_int,c1,method=method) + contract(eta0,c3,method=method)

    if order == 6:
        sol_c5 = contract(eta0,c5,method=method) 
        sol_c5 = contract(eta_6,c1,method=method)
        sol_c5 += contract(eta_int,c3,method=method)

    # Add normal-ordering corrections into flow equation, if norm == True
    if norm == True:
        sol_no = contractNO(eta_int,H2,method=method,eta=False,state=state) + contractNO(eta0,Hint,method=method,eta=False,state=state)
        sol4_no = contractNO(eta_int,Hint,method=method,eta=False,state=state)
        sol+=sol_no
        sol2 += sol4_no

        sol_c1_no = contractNO(eta_int,c1,method=method,eta=False,state=state) + contractNO(eta0,c3,method=method,eta=False,state=state)
        sol_c3_no = contractNO(eta_int,c3,method=method,eta=False,state=state)
        sol_c1 += sol_c1_no
        sol_c3 += sol_c3_no

    if order == 4:
        n1 = frn(eta0)*frn(H2)+frn(eta0)*frn(Hint)+frn(eta_int)*frn(H2)
        n2 = frn(eta_int)*frn(Hint)
    elif order == 6:
        n1 = frn(eta0)*frn(H2)+frn(eta0)*frn(Hint)+frn(eta_int)*frn(H2)+frn(eta_int)*frn(Hint)+frn(eta_6)*frn(H2)+frn(eta0)*frn(H6)
        n2 = frn(eta_int)*frn(H6) + frn(eta_6)*frn(Hint) + frn(eta_6)*frn(H6)

    if order == 4:
        return [sol,sol2,sol_c1,sol_c3,n1,n2]
    elif order == 6:
        return [sol,sol2,sol3,sol_c1,sol_c3,sol_c5,n1,n2]

def int_ode2(y,l,eta=[],method='einsum',norm=False,Hflow=True):
        """ Generate the flow equation for the interacting systems.

        e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

        Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
        the ijnput array eta will be used to specify the generator at this flow time step. The latter option will result 
        in a huge speed increase, at the potential cost of accuracy. This is because the SciPy routine used to 
        integrate the ODEs will sometimes add intermediate steps: recomputing eta on the fly will result in these 
        steps being computed accurately, while fixing eta will avoid having to recompute the generator every time an 
        interpolation step is added (leading to a speed increase), however it will mean that the generator evaluated at 
        these intermediate steps has errors of order <dl (where dl is the flow time step). For sufficiently small dl, 
        the benefits from the speed increase likely outweigh the decrease in accuracy.

        Parameters
        ----------
        l : float
            The (fictitious) flow time l which parameterises the unitary transform.
        y : array
            Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
        n : integer
            Linear system size.
        eta : array, optional
            Provide a pre-computed generator, if desired.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        sol0 : RHS of the flow equation for interacting system.

        """
        # print('y shape', y.shape)
        # Extract various components of the Hamiltonian from the ijnput array 'y'
        # id_print(y)
        # id_print(l)
        H2 = y[0]                           # Define quadratic part of Hamiltonian
        n,_,_,_ = H2.shape
        H2 = H2[::,::,0,0]
        # H2_0 = jnp.diag(jnp.diag(H2))       # Define diagonal quadratic part H0
        # V0 = H2 - H2_0                      # Define off-diagonal quadratic part

        Hint = y[1]                         # Define quartic part of Hamiltonian
        # Hint0 = jnp.zeros((n,n,n,n))        # Define diagonal quartic part 
        # for i in range(n):                  # Load Hint0 with values
        #     for j in range(n):
        #             Hint0 = Hint0.at[i,i,j,j].set(Hint[i,i,j,j])
        #             Hint0 = Hint0.at[i,j,j,i].set(Hint[i,j,j,i])
        # Vint = Hint-Hint0
        # id_print(H2)

        H2_0,V0,Hint0,Vint = extract_diag(H2,Hint)

        if norm == True:
            state = state_spinless(H2)

        if Hflow == True:
            # Compute the generator eta
            eta0 = contract(H2_0,V0,method=method,eta=True)
            eta_int = contract(Hint0,V0,method=method,eta=True) + contract(H2_0,Vint,method=method,eta=True)

            # Add normal-ordering corrections into generator eta, if norm == True
            if norm == True:

                eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H2_0,Vint,method=method,eta=True,state=state)
                eta0 += eta_no2

                eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
                eta_int += eta_no4
        else:
            eta0 = (eta[:n**2]).reshape(n,n)
            eta_int = (eta[n**2:]).reshape(n,n,n,n)

        # id_print(eta0)
   
        # Compute the RHS of the flow equation dH/dl = [\eta,H]
        sol = contract(eta0,H2,method=method)
        sol2 = contract(eta_int,H2,method=method) + contract(eta0,Hint,method=method)

        # Add normal-ordering corrections into flow equation, if norm == True
        if norm == True:
            sol_no = contractNO(eta_int,H2,method=method,eta=False,state=state) + contractNO(eta0,Hint,method=method,eta=False,state=state)
            sol4_no = contractNO(eta_int,Hint,method=method,eta=False,state=state)
            sol+=sol_no
            sol2 += sol4_no

        # id_print([sol,sol2])
        return [sol,sol2]

# @jit
def update(n2,n4,H2,Hint,steps,method='einsum'):

    n,_ = H2.shape
    H0,V0,Hint0,Vint = extract_diag(H2,Hint)

    eta2 = contract(H0,V0,method=method,comp=False,eta=True)
    eta4 = contract(Hint0,V0,method=method,comp=False,eta=True) + contract(H0,Vint,method=method,comp=False,eta=True)

    dl = steps[-1]-steps[0]
    # id_print(dl)
    dn2 = contract(eta2,n2,method=method,comp=False)
    dn4 = contract(eta4,n2,method=method,comp=False) + contract(eta2,n4,method=method,comp=False)
    n2 += dl*dn2
    n4 += dl*dn4

    return n2,n4

def liom_ode(y,l,n,array,method='tensordot',comp=False,Hflow=True,norm=False,bck=True):
    """ Generate the flow equation for density operators of the interacting systems.

        e.g. compute the RHS of dn/dl = [\eta,n] which will be used later to integrate n(l) -> n(l + dl)

        Note that this can be used to integrate density operators either 'forward' (from l=0 to l -> infinity) or
        also 'backward' (from l -> infinity to l=0), as the flow equations are the same either way. The only changes
        are the initial condition and the sign of the timestep dl.

        Parameters
        ----------
        l : float
            The (fictitious) flow time l which parameterises the unitary transform.
        y : array
            Array of size n**2 + n**4 containing all coefficients of the running density operator at flow time l.
        H : array
            Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
        n : integer
            Linear system size.
        eta : array, optional
            Provide a pre-computed generator, if desired.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        comp : bool, optional
            Specify whether the density operator is complex, e.g. for use in time evolution.
            Triggers the 'contract' function with 'jit' method to use efficient complex conjugation routine.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        sol0 : RHS of the flow equation for the density operator of the interacting system.


    """

    if Hflow == True:
        # Extract various components of the Hamiltonian from the ijnput array 'y'
        H2 = array[0]                       # Define quadratic part of Hamiltonian
        print('h2shape',H2.shape)
        # H0 = jnp.diag(jnp.diag(H2))           # Define diagonal quadratic part H0
        # V0 = H2 - H0                        # Define off-diagonal quadratic part B
        
        if len(array)>1:
            Hint = array[1]            # Define quartic part of Hamiltonian
            # Hint0 = jnp.zeros((n,n,n,n))     # Define diagonal quartic part 
            # for i in range(n):              # Load Hint0 with values
            #     for j in range(n):
            #         # if i != j:
            #             # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
            #             Hint0 = Hint0.at[i,i,j,j].set(Hint[i,i,j,j])
            #             Hint0 = Hint0.at[i,j,j,i].set(Hint[i,j,j,i])
            # Vint = Hint-Hint0

            H0,V0,Hint0,Vint = extract_diag(H2,Hint)

        # Compute the quadratic generator eta2
        eta2 = contract(H0,V0,method=method,comp=False,eta=True)
        # id_print(eta2)

        if len(array) > 1:
            eta4 = contract(Hint0,V0,method=method,comp=comp,eta=True) + contract(H0,Vint,method=method,comp=comp,eta=True)

        # Add normal-ordering corrections into generator eta, if norm == True
        # if norm == True:
        #     state=state_spinless(H2)
        #     eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H0,Vint,method=method,eta=True,state=state)
        #     eta2 += eta_no2

        #     eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
        #     eta4 += eta_no4

    # else:
    #     eta2 = (array[0:]).reshape(n,n)
    #     eta4 = (array[n**2::]).reshape(n,n,n,n)

    # Extract components of the density operator from ijnput array 'y'
    n2 = y[0]                           # Define quadratic part of density operator
    if len(y)>1:                     # If interacting system...
        n4 = y[1]                  #...then define quartic part of density operator
                    
    # Compute the quadratic terms in the RHS of the flow equation
    sol2 = contract(eta2,n2,method=method,comp=comp)

    # Compute quartic terms, if interacting system
    if len(y) > 1:
        sol4 = contract(eta4,n2,method=method,comp=comp) + contract(eta2,n4,method=method,comp=comp)

    # Add normal-ordering corrections into flow equation, if norm == True
    # if norm == True:
    #     sol_no = contractNO(eta4,n2,method=method,eta=False,state=state) + contractNO(eta2,n4,method=method,eta=False,state=state)
    #     sol+=sol_no
    #     if len(y) > n**2:
    #         sol4_no = contractNO(eta4,n4,method=method,eta=False,state=state)
    #         sol2 += sol4_no

    if bck == True:
        return [-1*sol2,-1*sol4]
    elif bck == False:
        return [sol2,sol4]

def liom_ode_int(y,l,n,array,bck=True,method='tensordot',comp=False,Hflow=True,norm=False):
    """ Generate the flow equation for density operators of the interacting systems.

        e.g. compute the RHS of dn/dl = [\eta,n] which will be used later to integrate n(l) -> n(l + dl)

        Note that this can be used to integrate density operators either 'forward' (from l=0 to l -> infinity) or
        also 'backward' (from l -> infinity to l=0), as the flow equations are the same either way. The only changes
        are the initial condition and the sign of the timestep dl.

        Parameters
        ----------
        l : float
            The (fictitious) flow time l which parameterises the unitary transform.
        y : array
            Array of size n**2 + n**4 containing all coefficients of the running density operator at flow time l.
        H : array
            Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
        n : integer
            Linear system size.
        eta : array, optional
            Provide a pre-computed generator, if desired.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        comp : bool, optional
            Specify whether the density operator is complex, e.g. for use in time evolution.
            Triggers the 'contract' function with 'jit' method to use efficient complex conjugation routine.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        sol0 : RHS of the flow equation for the density operator of the interacting system.


    """

    if Hflow == True:
        # Extract various components of the Hamiltonian from the ijnput array 'y'
        H2 = array[0]                       # Define quadratic part of Hamiltonian
        # H0 = jnp.diag(jnp.diag(H2))           # Define diagonal quadratic part H0
        # V0 = H2 - H0                        # Define off-diagonal quadratic part B

        # if len(array)>1:
        m,_ = y[0].shape
        Hint = array[1]                         # Define quartic part of Hamiltonian
        # Hint0 = jnp.zeros((m,m,m,m))        # Define diagonal quartic part 
        # for i in range(m):                  # Load Hint0 with values
        #     for j in range(m):
        #             Hint0 = Hint0.at[i,i,j,j].set(Hint[i,i,j,j])
        #             Hint0 = Hint0.at[i,j,j,i].set(Hint[i,j,j,i])
        # Vint = Hint-Hint0

        H0,V0,Hint0,Vint = extract_diag(H2,Hint)
        # id_print(H2)
        # Compute the quadratic generator eta2
        eta2 = contract(H0,V0,method=method,comp=False,eta=True)
        eta4 = contract(Hint0,V0,method=method,comp=comp,eta=True) + contract(H0,Vint,method=method,comp=comp,eta=True)
        # id_print(eta2)
        # Add normal-ordering corrections into generator eta, if norm == True
        # if norm == True:
        #     state=state_spinless(H2)
        #     eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H0,Vint,method=method,eta=True,state=state)
        #     eta2 += eta_no2

        #     eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
        #     eta4 += eta_no4

    # Extract components of the density operator from ijnput array 'y'
    n2 = y[0]                           # Define quadratic part of density operator
    n4 = y[1]                           #...then define quartic part of density operator
                    
    # Compute the quadratic terms in the RHS of the flow equation
    sol2 = -1*(contract(eta2,n2,method=method,comp=comp))

    # Compute quartic terms, if interacting system
    sol4 = -1*(contract(eta4,n2,method=method,comp=comp) + contract(eta2,n4,method=method,comp=comp))

    # Add normal-ordering corrections into flow equation, if norm == True
    # if norm == True:
    #     sol_no = contractNO(eta4,n2,method=method,eta=False,state=state) + contractNO(eta2,n4,method=method,eta=False,state=state)
    #     sol+=sol_no
    #     if len(y) > n**2:
    #         sol4_no = contractNO(eta4,n4,method=method,eta=False,state=state)
    #         sol2 += sol4_no

    return [sol2,sol4]

def liom_ode_int_fwd(y,l,n,array,bck=False,method='einsum',comp=False,Hflow=True,norm=False):
    """ Generate the flow equation for density operators of the interacting systems.

        e.g. compute the RHS of dn/dl = [\eta,n] which will be used later to integrate n(l) -> n(l + dl)

        Note that this can be used to integrate density operators either 'forward' (from l=0 to l -> infinity) or
        also 'backward' (from l -> infinity to l=0), as the flow equations are the same either way. The only changes
        are the initial condition and the sign of the timestep dl.

        Parameters
        ----------
        l : float
            The (fictitious) flow time l which parameterises the unitary transform.
        y : array
            Array of size n**2 + n**4 containing all coefficients of the running density operator at flow time l.
        H : array
            Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
        n : integer
            Linear system size.
        eta : array, optional
            Provide a pre-computed generator, if desired.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        comp : bool, optional
            Specify whether the density operator is complex, e.g. for use in time evolution.
            Triggers the 'contract' function with 'jit' method to use efficient complex conjugation routine.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        sol0 : RHS of the flow equation for the density operator of the interacting system.


    """

    if Hflow == True:
        # Extract various components of the Hamiltonian from the ijnput array 'y'
        H2 = array[0]                       # Define quadratic part of Hamiltonian
        # H0 = jnp.diag(jnp.diag(H2))           # Define diagonal quadratic part H0
        # V0 = H2 - H0                        # Define off-diagonal quadratic part B

        # if len(array)>1:
        m,_ = array[0].shape
        Hint = array[1]                         # Define quartic part of Hamiltonian
        # Hint0 = jnp.zeros((m,m,m,m))        # Define diagonal quartic part 
        # for i in range(m):                  # Load Hint0 with values
        #     for j in range(m):
        #             Hint0 = Hint0.at[i,i,j,j].set(Hint[i,i,j,j])
        #             Hint0 = Hint0.at[i,j,j,i].set(Hint[i,j,j,i])
        # Vint = Hint-Hint0

        H0,V0,Hint0,Vint = extract_diag(H2,Hint)

        # Compute the quadratic generator eta2
        eta2 = contract(H0,V0,method=method,comp=False,eta=True)
        eta4 = contract(Hint0,V0,method=method,comp=comp,eta=True) + contract(H0,Vint,method=method,comp=comp,eta=True)

        # Add normal-ordering corrections into generator eta, if norm == True
        # if norm == True:
        #     state=state_spinless(H2)
        #     eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H0,Vint,method=method,eta=True,state=state)
        #     eta2 += eta_no2

        #     eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
        #     eta4 += eta_no4

    # Extract components of the density operator from ijnput array 'y'
    n2 = y[0]                           # Define quadratic part of density operator
    n4 = y[1]                           #...then define quartic part of density operator
                    
    # Compute the quadratic terms in the RHS of the flow equation
    sol2 = contract(eta2,n2,method=method,comp=comp)

    # Compute quartic terms, if interacting system
    sol4 = contract(eta4,n2,method=method,comp=comp) + contract(eta2,n4,method=method,comp=comp)

    # Add normal-ordering corrections into flow equation, if norm == True
    # if norm == True:
    #     sol_no = contractNO(eta4,n2,method=method,eta=False,state=state) + contractNO(eta2,n4,method=method,eta=False,state=state)
    #     sol+=sol_no
    #     if len(y) > n**2:
    #         sol4_no = contractNO(eta4,n4,method=method,eta=False,state=state)
    #         sol2 += sol4_no

    if bck == True:
        return [-1*sol2,-1*sol4]
    elif bck == False:
        return [sol2,sol4]


def int_ode_fwd(l,y0,n,eta=[],method='jit',norm=False,Hflow=False,comp=False):
        """ Generate the flow equation for the interacting systems.

        e.g. compute the RHS of dH/dl = [\eta,H] which will be used later to integrate H(l) -> H(l + dl)

        Note that with the parameter Hflow = True, the generator will be recomputed as required. Using Hflow=False,
        the ijnput array eta will be used to specify the generator at this flow time step. The latter option will result 
        in a huge speed increase, at the potential cost of accuracy. This is because the SciPi routine used to 
        integrate the ODEs will sometimes add intermediate steps: recomputing eta on the fly will result in these 
        steps being computed accurately, while fixing eta will avoid having to recompute the generator every time an 
        interpolation step is added (leading to a speed increase), however it will mean that the generator evaluated at 
        these intermediate steps has errors of order <dl (where dl is the flow time step). For sufficiently small dl, 
        the benefits from the speed increase likely outweigh the decrease in accuracy.

        Parameters
        ----------
        l : float
            The (fictitious) flow time l which parameterises the unitary transform.
        y : array
            Array of size n**2 + n**4 containing all coefficients of the running Hamiltonian at flow time l.
        n : integer
            Linear system size.
        eta : array, optional
            Provide a pre-computed generator, if desired.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vectorize'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        sol0 : RHS of the flow equation for interacting system.

        """
        y = y0[:n**2+n**4]
        nlist = y0[n**2+n**4::]

        # Extract various components of the Hamiltonian from the ijnput array 'y'
        H = y[0:n**2]                   # Define quadratic part of Hamiltonian
        H = H.reshape(n,n)              # Reshape into matrix
        H0 = jnp.diag(jnp.diag(H))        # Define diagonal quadratic part H0
        V0 = H - H0                     # Define off-diagonal quadratic part B

        Hint = y[n**2:]                 # Define quartic part of Hamiltonian
        Hint = Hint.reshape(n,n,n,n)    # Reshape into rank-4 tensor
        Hint0 = jnp.zeros((n,n,n,n))     # Define diagonal quartic part 
        for i in range(n):              # Load Hint0 with values
            for j in range(n):
                # if i != j:
                    # Load dHint_diag with diagonal values (n_i n_j or c^dag_i c_j c^dag_j c_i)
                    Hint0[i,i,j,j] = Hint[i,i,j,j]
                    Hint0[i,j,j,i] = Hint[i,j,j,i]
        Vint = Hint-Hint0

        # Extract components of the density operator from ijnput array 'y'
        n2 = nlist[0:n**2]                  # Define quadratic part of density operator
        n2 = n2.reshape(n,n)            # Reshape into matrix
        if len(nlist)>n**2:                 # If interacting system...
            n4 = nlist[n**2::]              #...then define quartic part of density operator
            n4 = n4.reshape(n,n,n,n)    # Reshape into tensor
        
        if norm == True:
            state=state_spinless(H)

        if Hflow == True:
            # Compute the generator eta
            eta0 = contract(H0,V0,method=method,eta=True)
            eta_int = contract(Hint0,V0,method=method,eta=True) + contract(H0,Vint,method=method,eta=True)

            # Add normal-ordering corrections into generator eta, if norm == True
            if norm == True:

                eta_no2 = contractNO(Hint,V0,method=method,eta=True,state=state) + contractNO(H0,Vint,method=method,eta=True,state=state)
                eta0 += eta_no2

                eta_no4 = contractNO(Hint0,Vint,method=method,eta=True,state=state)
                eta_int += eta_no4
        else:
            eta0 = (eta[:n**2]).reshape(n,n)
            eta_int = (eta[n**2:]).reshape(n,n,n,n)
   
        # Compute the RHS of the flow equation dH/dl = [\eta,H]
        sol = contract(eta0,H0+V0,method=method)
        sol2 = contract(eta_int,H0+V0,method=method) + contract(eta0,Hint,method=method)

        nsol = contract(eta0,n2,method=method,comp=comp)
        if len(y) > n**2:
            nsol2 = contract(eta_int,n2,method=method,comp=comp) + contract(eta0,n4,method=method,comp=comp)


        # Add normal-ordering corrections into flow equation, if norm == True
        if norm == True:
            sol_no = contractNO(eta_int,H0+V0,method=method,eta=False,state=state) + contractNO(eta0,Hint,method=method,eta=False,state=state)
            sol4_no = contractNO(eta_int,Hint,method=method,eta=False,state=state)
            sol+= sol_no
            sol2 += sol4_no
        
        # Define and load output list sol0
        sol0 = jnp.zeros(2*(n**2+n**4))
        sol0[:n**2] = sol.reshape(n**2)
        sol0[n**2:n**2+n**4] = sol2.reshape(n**4)
        sol0[n**2+n**4:2*n**2+n**4] = nsol.reshape(n**2)
        sol0[2*n**2+n**4:] = nsol2.reshape(n**4)

        return sol0
#------------------------------------------------------------------------------  

def flow_static(n,hamiltonian,dl_list,qmax,cutoff,method='jit',store_flow=True):
    """
    Diagonalise an initial non-interacting Hamiltonian and compute the integrals of motion.

    Note that this function does not use the trick of fixing eta: as non-interacting systems are
    quadratic in terms of fermion operators, there are no high-order tensor contractions and so 
    fixing eta is not necessary here as the performance gain is not expected to be significant.

        Parameters
        ----------
        n : integer
            Linear system size.
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        dl_list : array, float
            List of flow times to use for the numerical integration.
        qmax : integer
            Maximum number of flow time steps.
        cutoff : float
            Threshold value below which off-diagonal elements are set to zero.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.

        Returns
        -------
        output : dict
            Dictionary containing diagonal Hamiltonian ("H0_diag") and LIOM on central site ("LIOM").
    
    """
    H2 = hamiltonian.H2_spinless
    H2 = H2.astype(jnp.float64)

    # Define integrator
    sol = ode(nonint_ode,(H2),dl_list)
    print(jnp.sort(jnp.diag(sol[-1])))

    # Initialise a density operator in the diagonal basis on the central site
    init_liom = jnp.zeros((n,n))
    init_liom = init_liom.at[n//2,n//2].set(1.0)

    # Reverse list of flow times in order to conduct backwards integration
    dl_list = -1*dl_list[::-1]
    sol = sol[::-1]

    # Do backwards integration
    k0=0
    for k0 in range(len(dl_list)-1):
        liom = ode(liom_ode,init_liom,dl_list[k0:k0+2],n,sol[k0])
        init_liom = liom[-1]

    # Take final value for the transformed density operator and reshape to a matrix
    central = (liom[-1,:n**2]).reshape(n,n)

    # Build output dictionary
    output = {"H0_diag":sol[0].reshape(n,n),"LIOM":central}
    if store_flow == True:
        output["flow"] = sol[::-1]
        output["dl_list"] = dl_list[::-1]

    return output
    
# @jit(nopython=True,parallel=True,fastmath=True)
def proc(mat,cutoff):
    """ Test function to zero all matrix elements below a cutoff. """
    for i in range(len(mat)):
        if mat[i] < cutoff:
            mat[i] = 0.
    return mat

def flow_static_int(n,hamiltonian,dl_list,qmax,cutoff,method='jit',norm=True,Hflow=False,store_flow=False):
    """
    Diagonalise an initial interacting Hamiltonian and compute the integrals of motion.

    Parameters
        ----------
        n : integer
            Linear system size.
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        Hint : array, float
            Diagonal component of Hamiltonian
        Vint : array, float
            Off-diagonal component of Hamiltonian.
        dl_list : array, float
            List of flow times to use for the numerical integration.
        qmax : integer
            Maximum number of flow time steps.
        cutoff : float
            Threshold value below which off-diagonal elements are set to zero.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        output : dict
            Dictionary containing diagonal Hamiltonian ("H0_diag","Hint"), LIOM interaction coefficient ("LIOM Interactions"),
            the LIOM on central site ("LIOM") and the value of the second invariant of the flow ("Invariant").
    
    """
    H2,Hint = hamiltonian.H2_spinless,hamiltonian.H4_spinless

    # Number of intermediate setps specified for the solver to use
    # It will in any case insert others as needed
    increment = 2

    # Optional: force a fixed number of steps (ignore cutoff/accuracy).
    forced = _force_steps()
    if forced is not None:
        # Need forced+1 time points to make forced steps.
        dl_list = dl_list[: min(len(dl_list), forced + 1)]
        print(f"        [FORCE_STEPS] Running exactly {len(dl_list)-1} steps (ignoring cutoff)")

    print('dl_list',len(dl_list),dl_list[0],dl_list[-1])

    # Define integrator
    # sol = ode(int_ode,[H2,Hint],dl_list)
    mem_tot = (len(dl_list)*(n**2+n**4)*4)/1e9
    chunk = int(np.ceil(mem_tot/6))
    #print('MANUALLY SET TO 20 CHUNKS FOR DEBUG PURPOSES')
    # chunk =int(2)
    if chunk > 1:
        chunk_size = len(dl_list)//chunk
        print('Memory',mem_tot,chunk,chunk_size)
    # chunk =int(2)

    if chunk <= 1:
        # Integration with hard-coded event handling
        k=1
        sol2 = jnp.zeros((len(dl_list),n,n))
        sol4 = jnp.zeros((len(dl_list),n,n,n,n))
        sol2 = sol2.at[0].set(H2)
        sol4 = sol4.at[0].set(Hint)
        J0 = 1
        # Ensure peak RSS is captured right after allocating the dominant buffers
        memlog("flow:alloc", step=0, mode="original", kind="sol2/sol4", steps=len(dl_list), n=n)
        last_h2 = sol2[0]
        last_h4 = sol4[0]

        # term = ODETerm(int_ode)
        # solver = Dopri5()

        sol2_test=jnp.zeros((len(dl_list),n,n,n,n))
        sol2_test=sol2_test.at[::,::,::,0,0].set(sol2)

        # print('jax list')
        # print(make_jaxpr(int_ode)([jnp.zeros((n,n)),jnp.zeros((n,n,n,n))],0.1))
        # print('*****************************************************')
        # print('*****************************************************')
        # print('*****************************************************')
        # print('*****************************************************')
        # print('jax array')
        # print(make_jaxpr(int_ode2)(jnp.array([jnp.zeros((n,n,n,n)),jnp.zeros((n,n,n,n))]),0.1))

        # ODE tolerances - configurable via environment for performance tuning
        # Looser tolerances (1e-5) can be 2-5x faster with minimal accuracy loss
        _rtol = float(os.environ.get('PYFLOW_ODE_RTOL', '1e-6'))
        _atol = float(os.environ.get('PYFLOW_ODE_ATOL', '1e-6'))
        print(f"        Flow integration: max_steps={len(dl_list)}, cutoff={cutoff:.2e}, rtol={_rtol:.0e}, atol={_atol:.0e}")
        last_progress = 0
        approx = _approx_enabled()
        _approx_cache: dict = {}
        while k <len(dl_list) and ((forced is not None) or (J0 > cutoff)):
            if approx and (not norm):
                h2_next, h4_next = _approx_step_h2_h4(
                    sol2[k-1],
                    sol4[k-1],
                    float(dl_list[k-1]),
                    float(dl_list[k]),
                    step_idx=k,
                    method=method,
                    cache=_approx_cache,
                    norm=norm,
                    Hflow=True,
                )
                sol2 = sol2.at[k].set(h2_next)
                sol4 = sol4.at[k].set(h4_next)
                J0 = jnp.max(jnp.abs(h2_next - jnp.diag(jnp.diag(h2_next))))
                last_h2 = h2_next
                last_h4 = h4_next
            else:
                steps = np.linspace(dl_list[k-1],dl_list[k],num=increment,endpoint=True)
                soln = ode(int_ode,[sol2[k-1],sol4[k-1]],steps,rtol=_rtol,atol=_atol)
                sol2 = sol2.at[k].set(soln[0][-1])
                sol4 = sol4.at[k].set(soln[1][-1])
                J0 = jnp.max(jnp.abs(soln[0][-1] - jnp.diag(jnp.diag(soln[0][-1]))))
                last_h2 = soln[0][-1]
                last_h4 = soln[1][-1]

            # Progress printing every 10% or every 50 steps
            progress = (k * 100) // len(dl_list)
            if progress >= last_progress + 10 or k % 50 == 0:
                print(f"        Step {k}/{len(dl_list)} ({progress}%) | l={dl_list[k]:.4f} | off-diag={J0:.2e}", flush=True)
                last_progress = progress
            
            # Add memlog record
            if k % 10 == 0:
                memlog("flow:step", step=k, mode="original")
            k += 1
        print(f"        Converged at step {k-1}, final off-diag={J0:.2e}")

    else:

        # Initialise arrays
        # Note: the memory required for these arrays is *not* pre-allocated. The reason is twofold: partly, it
        # is likely that the integration will finish before the max value of dl_list is encountered, therefore 
        # it's a waste to allocate all the memory. Secondly, in a later step we create a shortened copy of the array
        # of the form sol2[0:k], which returns a new array of length k that requires separate memory allocation, so if we 
        # max out the memory allocation here, there's no space left for to allocate the shortened array later.
        # (Modifying the array in-place would be better, but I don't know how to do that...)

        k=1
        sol2 = np.zeros((len(dl_list),n,n),dtype=np.float32)
        # sol2.fill(0.)
        sol4 = np.zeros((len(dl_list),n,n,n,n),dtype=np.float32)
        # sol4.fill(0.)
        sol2_gpu = jnp.zeros((chunk_size,n,n))
        sol4_gpu = jnp.zeros((chunk_size,n,n,n,n))
        sol2_gpu = sol2_gpu.at[0].set(H2)
        sol4_gpu = sol4_gpu.at[0].set(Hint)
        J0 = 1
        memlog("flow:alloc", step=0, mode="original", kind="sol2/sol4_chunked", steps=len(dl_list), n=n, chunk_size=chunk_size)
        last_h2 = sol2_gpu[0]
        last_h4 = sol4_gpu[0]
    
        # Integration with hard-coded event handling
        # ODE tolerances - configurable via environment for performance tuning
        _rtol = float(os.environ.get('PYFLOW_ODE_RTOL', '1e-6'))
        _atol = float(os.environ.get('PYFLOW_ODE_ATOL', '1e-6'))
        print(f"        Flow integration (chunked): max_steps={len(dl_list)}, cutoff={cutoff:.2e}, chunk_size={chunk_size}, rtol={_rtol:.0e}")
        last_progress = 0
        approx = _approx_enabled()
        _approx_cache: dict = {}
        while k <len(dl_list) and ((forced is not None) or (J0 > cutoff)):
            if approx and (not norm):
                h2_prev = sol2_gpu[k%chunk_size-1]
                h4_prev = sol4_gpu[k%chunk_size-1]
                h2_next, h4_next = _approx_step_h2_h4(
                    h2_prev,
                    h4_prev,
                    float(dl_list[k-1]),
                    float(dl_list[k]),
                    step_idx=k,
                    method=method,
                    cache=_approx_cache,
                    norm=norm,
                    Hflow=True,
                )
                sol2_gpu = sol2_gpu.at[k%chunk_size].set(h2_next)
                sol4_gpu = sol4_gpu.at[k%chunk_size].set(h4_next)
                J0 = jnp.max(jnp.abs(h2_next - jnp.diag(jnp.diag(h2_next))))
                last_h2 = h2_next
                last_h4 = h4_next
            else:
                steps = np.linspace(dl_list[k-1],dl_list[k],num=increment,endpoint=True)
                soln = ode(int_ode,[sol2_gpu[k%chunk_size-1],sol4_gpu[k%chunk_size-1]],steps,rtol=_rtol,atol=_atol)
                sol2_gpu = sol2_gpu.at[k%chunk_size].set(soln[0][-1])
                sol4_gpu = sol4_gpu.at[k%chunk_size].set(soln[1][-1])
                J0 = jnp.max(jnp.abs(soln[0][-1] - jnp.diag(jnp.diag(soln[0][-1]))))
                last_h2 = soln[0][-1]
                last_h4 = soln[1][-1]

            # Progress printing every 10% or every 50 steps
            progress = (k * 100) // len(dl_list)
            if progress >= last_progress + 10 or k % 50 == 0:
                print(f"        Step {k}/{len(dl_list)} ({progress}%) | l={dl_list[k]:.4f} | off-diag={J0:.2e}", flush=True)
                last_progress = progress

            if k%chunk_size==0:
                count = int(k/chunk_size)
                if (sol2[(count-1)*chunk_size:(count)*chunk_size]).shape==np.array(sol2_gpu).shape:
                    sol2[(count-1)*chunk_size:(count)*chunk_size] = np.array(sol2_gpu)
                    sol4[(count-1)*chunk_size:(count)*chunk_size] = np.array(sol4_gpu)
            elif k == len(dl_list)-1 or J0 <= cutoff:
                remainder = len(sol2_gpu[0:k%chunk_size])
                sol2[(count)*chunk_size:k] = np.array(sol2_gpu[0:remainder])
                sol4[(count)*chunk_size:k] = np.array(sol4_gpu[0:remainder])
            k += 1
        print(f"        Converged at step {k-1}, final off-diag={J0:.2e}")
        

    print('dl_list',len(dl_list),dl_list[0],dl_list[-1])
    print(k,J0,dl_list[k-1])
    if k != len(dl_list):
        dl_list = dl_list[0:k]
        sol2 = sol2[0:k]
        sol4 = sol4[0:k]
        

    steps = np.zeros(len(dl_list)-1)
    for i in range(len(dl_list)-1):
        steps[i] = dl_list[i+1]-dl_list[i]
    print(np.max(steps),np.min(steps))

    # Resize chunks
    if chunk > 1:
        mem_tot = (len(dl_list)*(n**2+n**4)*4)/1e9
        chunk = int(np.ceil(mem_tot/6))
        # chunk = 2
        chunk_size = len(dl_list)//chunk
        print('NEW CHUNK SIZE',chunk_size)
        if int(chunk_size*chunk) != int(len(dl_list)):
            print('Chunk size error - LIOMs may not be reliable.')

        del sol2_gpu
        del sol4_gpu 

        sol2_gpu = jnp.zeros((chunk_size,n,n))
        sol4_gpu = jnp.zeros((chunk_size,n,n,n,n))

    # Store initial interaction value and trace of initial H^2 for later error estimation
    delta = jnp.max(Hint)
    e1 = jnp.trace(jnp.dot(H2,H2))
    
    # Define final quadratic/quartic Hamiltonian from the last computed step.
    # (When approximate stepping is enabled, `soln` is not defined.)
    H0_diag = last_h2.reshape(n, n)
    print(jnp.sort(jnp.diag(H0_diag)))
    print('Max |V|: ',jnp.max(jnp.abs(H0_diag-jnp.diag(jnp.diag(H0_diag)))))
    # Define final diagonal quartic Hamiltonian
    Hint2 = last_h4.reshape(n, n, n, n)   
    # Extract only the density-density terms of the final quartic Hamiltonian, as a matrix                     
    HFint = jnp.zeros(n**2).reshape(n,n)
    for i in range(n):
        for j in range(n):
            HFint = HFint.at[i,j].set(Hint2[i,i,j,j]-Hint2[i,j,j,i])

    # Compute the difference in the second invariant of the flow at start and end
    # This acts as a measure of the unitarity of the transform
    Hflat = HFint.reshape(n**2)
    inv = 0.
    for i in range(n**2):
        inv += 2*Hflat[i]**2
    e2 = jnp.trace(jnp.dot(H0_diag,H0_diag))
    inv2 = jnp.abs(e1 - e2 + ((2*delta)**2)*(n-1) - inv)/jnp.abs(e2+((2*delta)**2)*(n-1))

    # Compute the l-bit interactions from the density-density terms in the final Hamiltonian
    lbits = jnp.zeros(n-1)
    for q in range(1,n):
        lbits = lbits.at[q-1].set(jnp.median(jnp.log10(jnp.abs(jnp.diag(HFint,q)+jnp.diag(HFint,-q))/2.)))

    # Initialise a density operator in the microscopic basis on the central site
    init_liom2 = jnp.zeros((n,n))
    init_liom4 = jnp.zeros((n,n,n,n))
    init_liom2 = init_liom2.at[n//2,n//2].set(1.0)

    # Do forwards integration
    k0=0
    jit_update = jit(update)
    if chunk <= 1:
        for k0 in range(len(dl_list)-1):
            steps = np.linspace(dl_list[k0],dl_list[k0+1],num=increment,endpoint=True)
            init_liom2,init_liom4  = jit_update(init_liom2,init_liom4,sol2[k0],sol4[k0],steps)

    else:
        for k0 in range(len(dl_list)-1):
            # print(k0,int(k0/chunk_size),k0%chunk_size)
            if k0%chunk_size==0:
                # print('load mem')
                count = int(k0/chunk_size)
                if jnp.array(sol2[count*chunk_size:(count+1)*chunk_size]).shape == sol2_gpu.shape:
                    # print('load1')
                    sol2_gpu = sol2_gpu.at[:,:].set(jnp.array(sol2[count*chunk_size:(count+1)*chunk_size]))
                    sol4_gpu = sol4_gpu.at[:,:].set(jnp.array(sol4[count*chunk_size:(count+1)*chunk_size]))
                else:
                    # print('load2')
                    sol2_gpu = jnp.array(sol2[count*chunk_size:(count+1)*chunk_size])
                    sol4_gpu = jnp.array(sol4[count*chunk_size:(count+1)*chunk_size])

            # print('****')
            # print(k0)
            # print(sol2_gpu[k0%chunk_size])
            # print(sol2[k0])
            steps = np.linspace(dl_list[k0],dl_list[k0+1],num=increment,endpoint=True)

            init_liom2,init_liom4  = jit_update(init_liom2,init_liom4,sol2_gpu[k0%chunk_size],sol4_gpu[k0%chunk_size],steps)

    liom_fwd2 = np.array(init_liom2)
    liom_fwd4 = np.array(init_liom4)
    

    if chunk > 1:
        del sol2_gpu
        del sol4_gpu
        sol2_gpu = jnp.zeros((chunk_size,n,n))
        sol4_gpu = jnp.zeros((chunk_size,n,n,n,n))

    # Preserve the (forward, truncated) flow-time grid for output before reversing for backward integration.
    dl_list_fwd = np.array(dl_list)

    # Reverse list of flow times in order to conduct backwards integration
    dl_list = dl_list[::-1]
    # sol2=sol2[::-1]
    # sol4 = sol4[::-1]

    # Initialise a density operator in the diagonal basis on the central site
    init_liom2 = jnp.zeros((n,n))
    init_liom4 = jnp.zeros((n,n,n,n))
    init_liom2 = init_liom2.at[n//2,n//2].set(1.0)

    # Do backwards integration
    k0=0
    if chunk <= 1:
        for k0 in range(len(dl_list)-1):
            steps = np.linspace(dl_list[k0],dl_list[k0+1],num=increment,endpoint=True)
            init_liom2,init_liom4  = jit_update(init_liom2,init_liom4,sol2[-k0+1],sol4[-k0+1],steps)
    else:
        for k0 in range(len(dl_list)-1):
            if k0%chunk_size==0:
                count = int(k0/chunk_size)

                if count == 0 and ((sol2[-1*((count+1)*chunk_size)::]).shape == sol2_gpu.shape):
                    sol2_gpu = sol2_gpu.at[:,:].set(jnp.array(sol2[-1*((count+1)*chunk_size)::]))
                    sol4_gpu = sol4_gpu.at[:,:].set(jnp.array(sol4[-1*((count+1)*chunk_size)::]))
                elif count > 0 and (sol2[-1*((count+1)*chunk_size):-((count)*chunk_size)]).shape == sol2_gpu.shape:
                    sol2_gpu = sol2_gpu.at[:,:].set(jnp.array(sol2[-1*((count+1)*chunk_size):-((count)*chunk_size)]))
                    sol4_gpu = sol4_gpu.at[:,:].set(jnp.array(sol4[-1*((count+1)*chunk_size):-((count)*chunk_size)]))
                else:
                    sol2_gpu = jnp.array(sol2[0:-1*((count)*chunk_size)])
                    sol4_gpu = jnp.array(sol4[0:-1*((count)*chunk_size)])

            steps = np.linspace(dl_list[k0],dl_list[k0+1],num=increment,endpoint=True)
            # print('****')
            # print(k0)
            # print(sol2_gpu[-(k0)%chunk_size-1])
            # print(sol2[-k0-1])
            init_liom2,init_liom4  = jit_update(init_liom2,init_liom4,sol2_gpu[-(k0)%chunk_size-1],sol4_gpu[-(k0)%chunk_size-1],steps)
    
    # Reverse again to get these lists the right way around
    # dl_list = -1*dl_list[::-1]
    # sol2=sol2[::-1]
    # sol4 = sol4[::-1]

    # import matplotlib.pyplot as plt
    # plt.plot(jnp.log10(jnp.abs(jnp.diag(init_liom2.reshape(n,n)))))
    # plt.plot(jnp.log10(jnp.abs(jnp.diag(liom_fwd2.reshape(n,n)))),'--')

    output = {
        "H0_diag": np.array(H0_diag),
        "Hint": np.array(Hint2),
        "LIOM Interactions": lbits,
        "LIOM2": init_liom2,
        "LIOM4": init_liom4,
        "LIOM2_FWD": liom_fwd2,
        "LIOM4_FWD": liom_fwd4,
        "Invariant": inv2,
        # Keep output schema consistent with other flow routines so main scripts can always export.
        "dl_list": dl_list_fwd,
        "truncation_err": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    }
    if store_flow == True:
        output["flow2"] = np.array(sol2)
        output["flow4"] = np.array(sol4)
        # For store_flow we keep the (reversed) dl_list used in backward integration.
        output["dl_list"] = dl_list

    # Free up some memory
    # del sol2,sol4
    # gc.collect()

    return output


#------------------------------------------------------------------------------
# Optimized flow integration using JAX lax.while_loop
#------------------------------------------------------------------------------

def _create_int_ode_jit(n_size):
    """Create a JIT-compatible ODE function for flow equations.
    
    Args:
        n_size: System size (needed for pre-computing masks)
    """
    # Pre-compute masks outside JIT for efficiency
    _ex_mask = ex_helper(n_size)
    _no_mask = no_helper(n_size)
    
    @jit
    def int_ode_jit(y, l):
        """JIT-compiled flow equation for interacting systems.
        
        Implements dH/dl = [eta, H] where eta is the Wegner generator.
        """
        H2 = y[0]
        Hint = y[1]
        n = n_size
        
        # Extract diagonal and off-diagonal parts
        H2_0 = jnp.diag(jnp.diag(H2))
        V0 = H2 - H2_0
        
        # For Hint, use the ex_helper mask
        Hint0 = jnp.multiply(_ex_mask, Hint.reshape(n**4)).reshape(n, n, n, n)
        Vint = Hint - Hint0
        
        # Compute generator eta = [H0, V] (Wegner generator)
        # eta0 = [H2_0, V0] = H2_0 @ V0 - V0 @ H2_0
        eta0 = jnp.einsum('ij,jk->ik', H2_0, V0, optimize=True) - jnp.einsum('ki,ij->kj', V0, H2_0, optimize=True)
        
        # eta_int from [Hint0, V0] and [H2_0, Vint]
        # [Hint0, V0] contribution - rank-4 with rank-2 contraction (con42 formula)
        eta_int_1 = (jnp.einsum('abcd,df->abcf', Hint0, V0, optimize=True)
                   - jnp.einsum('abcd,ec->abed', Hint0, V0, optimize=True)
                   + jnp.einsum('abcd,bf->afcd', Hint0, V0, optimize=True)
                   - jnp.einsum('abcd,ea->ebcd', Hint0, V0, optimize=True))
        
        # [H2_0, Vint] contribution - this is con24 = -con42(Vint, H2_0)
        eta_int_2 = -(jnp.einsum('abcd,df->abcf', Vint, H2_0, optimize=True)
                    - jnp.einsum('abcd,ec->abed', Vint, H2_0, optimize=True)
                    + jnp.einsum('abcd,bf->afcd', Vint, H2_0, optimize=True)
                    - jnp.einsum('abcd,ea->ebcd', Vint, H2_0, optimize=True))
        
        eta_int = eta_int_1 + eta_int_2
        
        # Apply mask to eta_int
        eta_int = jnp.multiply(_no_mask, eta_int.reshape(n**4)).reshape(n, n, n, n)
        
        # Compute flow equation RHS: dH/dl = [eta, H]
        # sol = [eta0, H2] (rank-2 commutator)
        sol = jnp.einsum('ij,jk->ik', eta0, H2, optimize=True) - jnp.einsum('ki,ij->kj', H2, eta0, optimize=True)
        
        # sol2 = [eta_int, H2] + [eta0, Hint]
        # [eta_int, H2] - rank-4 with rank-2 contraction (con42)
        sol2_1 = (jnp.einsum('abcd,df->abcf', eta_int, H2, optimize=True)
                - jnp.einsum('abcd,ec->abed', eta_int, H2, optimize=True)
                + jnp.einsum('abcd,bf->afcd', eta_int, H2, optimize=True)
                - jnp.einsum('abcd,ea->ebcd', eta_int, H2, optimize=True))
        
        # [eta0, Hint] - rank-2 with rank-4 contraction (con24 = -con42(Hint, eta0))
        sol2_2 = -(jnp.einsum('abcd,df->abcf', Hint, eta0, optimize=True)
                 - jnp.einsum('abcd,ec->abed', Hint, eta0, optimize=True)
                 + jnp.einsum('abcd,bf->afcd', Hint, eta0, optimize=True)
                 - jnp.einsum('abcd,ea->ebcd', Hint, eta0, optimize=True))
        
        sol2 = sol2_1 + sol2_2
        
        # Apply mask to sol2
        sol2 = jnp.multiply(_no_mask, sol2.reshape(n**4)).reshape(n, n, n, n)
        
        return [sol, sol2]
    
    return int_ode_jit


def flow_static_int_jit(n, hamiltonian, dl_list, qmax, cutoff, method='tensordot', norm=False, Hflow=True, store_flow=False):
    """
    JIT-optimized version of flow_static_int using lax.while_loop.
    
    This version compiles the entire integration loop with XLA, providing
    significant speedup by eliminating Python overhead between steps.
    
    Use environment variable USE_JIT_FLOW=1 to enable this version.
    
    Note: This version does not support norm=True (normal ordering corrections)
    or backward LIOM integration. Use standard flow_static_int for those features.
    """
    from jax import lax
    
    if norm:
        print("        [JIT Flow] Warning: norm=True not supported in JIT mode, using norm=False")
    
    H2, Hint = hamiltonian.H2_spinless, hamiltonian.H4_spinless
    
    # ODE tolerances
    _rtol = float(os.environ.get('PYFLOW_ODE_RTOL', '1e-6'))
    _atol = float(os.environ.get('PYFLOW_ODE_ATOL', '1e-6'))
    
    print(f"        [JIT Flow] max_steps={len(dl_list)}, cutoff={cutoff:.2e}, rtol={_rtol:.0e}, atol={_atol:.0e}")
    
    # Convert to JAX arrays
    dl_arr = jnp.array(dl_list)
    H2_init = jnp.array(H2)
    Hint_init = jnp.array(Hint)
    max_steps = len(dl_list)
    cutoff_jnp = jnp.float32(cutoff)
    
    # Create the JIT-compiled ODE function
    int_ode_jit = _create_int_ode_jit(n)
    
    # Define single integration step
    def single_step(H2_curr, Hint_curr, dl_start, dl_end):
        """Integrate one flow step from dl_start to dl_end."""
        steps = jnp.linspace(dl_start, dl_end, num=2, endpoint=True)
        soln = ode(int_ode_jit, [H2_curr, Hint_curr], steps, rtol=_rtol, atol=_atol)
        return soln[0][-1], soln[1][-1]
    
    # Define the loop body for lax.while_loop
    def cond_fn(state):
        """Continue while k < max_steps and J0 > cutoff."""
        k, J0, _, _ = state
        return jnp.logical_and(k < max_steps, J0 > cutoff_jnp)
    
    def body_fn(state):
        """Execute one integration step."""
        k, J0, H2_curr, Hint_curr = state
        
        # Get flow time bounds for this step
        dl_start = dl_arr[k - 1]
        dl_end = dl_arr[k]
        
        # Integrate one step
        H2_new, Hint_new = single_step(H2_curr, Hint_curr, dl_start, dl_end)
        
        # Compute off-diagonal magnitude
        J0_new = jnp.max(jnp.abs(H2_new - jnp.diag(jnp.diag(H2_new))))
        
        return (k + 1, J0_new, H2_new, Hint_new)
    
    # JIT compile the entire loop
    @jit
    def run_flow_loop(H2_init, Hint_init):
        init_state = (1, jnp.float32(1.0), H2_init, Hint_init)
        final_state = lax.while_loop(cond_fn, body_fn, init_state)
        return final_state
    
    # Run the compiled flow
    print("        [JIT Flow] Compiling (first run may be slow)...", flush=True)
    compile_start = time.time()
    
    final_k, final_J0, H2_final, Hint_final = run_flow_loop(H2_init, Hint_init)
    
    # Block until computation completes
    final_k = int(final_k)
    final_J0 = float(final_J0)
    
    elapsed = time.time() - compile_start
    print(f"        [JIT Flow] Converged at step {final_k-1}, off-diag={final_J0:.2e}, time={elapsed:.2f}s")
    
    # Build output (same format as flow_static_int)
    H0_diag = H2_final
    Hint2 = Hint_final
    
    # Extract l-bit interactions
    HFint = jnp.zeros((n, n))
    for i in range(n):
        for j in range(n):
            HFint = HFint.at[i, j].set(Hint2[i, i, j, j] - Hint2[i, j, j, i])
    
    lbits = jnp.zeros(n - 1)
    for q in range(1, n):
        diag_sum = jnp.abs(jnp.diag(HFint, q)) + jnp.abs(jnp.diag(HFint, -q))
        lbits = lbits.at[q - 1].set(jnp.median(jnp.log10(diag_sum / 2.0 + 1e-20)))
    
    # LIOM placeholder (backward integration not implemented in JIT version)
    init_liom2 = jnp.zeros((n, n))
    init_liom2 = init_liom2.at[n // 2, n // 2].set(1.0)
    init_liom4 = jnp.zeros((n, n, n, n))
    
    output = {
        "H0_diag": np.array(H0_diag),
        "Hint": np.array(Hint2),
        "lbits": np.array(lbits),
        "LIOM2": np.array(init_liom2),
        "LIOM4": np.array(init_liom4),
        "LIOM2_FWD": np.array(init_liom2),
        "LIOM4_FWD": np.array(init_liom4),
        "dl_list": dl_list[:final_k],
        "truncation_err": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    }
    
    return output


def flow_int_ITC(n,hamiltonian,dl_list,qmax,cutoff,tlist,method='jit',norm=True,Hflow=False,store_flow=False):
    """
    Diagonalise an initial interacting Hamiltonian, compute the forward integrals of motion and 
    also compute the infinite-temperature autocorrelation function.

    Parameters
        ----------
        n : integer
            Linear system size.
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        Hint : array, float
            Diagonal component of Hamiltonian
        Vint : array, float
            Off-diagonal component of Hamiltonian.
        dl_list : array, float
            List of flow times to use for the numerical integration.
        qmax : integer
            Maximum number of flow time steps.
        cutoff : float
            Threshold value below which off-diagonal elements are set to zero.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        output : dict
            Dictionary containing diagonal Hamiltonian ("H0_diag","Hint"), LIOM interaction coefficient ("LIOM Interactions"),
            the LIOM on central site ("LIOM") and the value of the second invariant of the flow ("Invariant").
    
    """
    H2,Hint = hamiltonian.H2_spinless,hamiltonian.H4_spinless

    # Integration with hard-coded event handling
    k=1
    sol2 = H2
    sol4 = Hint
    J0 = 1

    # Initialise a density operator in the microscopic basis on the central site
    n2 = jnp.zeros((n,n))
    n4 = jnp.zeros((n,n,n,n))
    n2 = n2.at[n//2,n//2].set(1.0)

    soln = ode(int_ode_ITC,[sol2,sol4,n2,n4],dl_list,rtol=1e-8,atol=1e-8)

    liom_fwd2 = n2
    liom_fwd4 = n4

    # Define final diagonal quadratic Hamiltonian
    H0_diag = soln[0][-1].reshape(n,n)
    print(jnp.sort(jnp.diag(H0_diag)))
    print('Max |V|: ',jnp.max(jnp.abs(H0_diag-jnp.diag(jnp.diag(H0_diag)))))
    # Define final diagonal quartic Hamiltonian
    Hint2 = soln[1][-1].reshape(n,n,n,n)   
    liom_fwd2 = soln[2][-1].reshape(n,n)
    liom_fwd4 = soln[3][-1].reshape(n,n,n,n)

    dyn,corr=dyn_itc(n,tlist,np.array(liom_fwd2,dtype=np.complex128),np.array(H0_diag),np.array(liom_fwd4,dtype=np.complex128),np.array(Hint2))

    print(liom_fwd2)

    # Optional debug plot (safe if matplotlib is installed).
    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.plot(tlist, corr, "rx--")
        # plt.show()
        # plt.close()
    except Exception:
        pass

    output = {"H0_diag":np.array(H0_diag), "Hint":np.array(Hint2),"LIOM2_FWD":liom_fwd2,"LIOM4_FWD":liom_fwd4,"dyn": dyn, "corr":corr}

    return output

def res(n,H2,res_cut,cutoff=1e-8):
    reslist = jnp.zeros(n*(n-1))
    count = 0
    for i in range(n):
        for j in range(i):
            reslist = reslist.at[count].set(res_test(i,j,H2[i,j],H2[i,i],H2[j,j],res_cut))
            # if reslist[count] != 0:
            #     print(i,j,H2[i,j],H2[i,j]/res_cut,np.abs(H2[i,i]-H2[j,j]))
            count += 1

    reslist = reslist[reslist > cutoff]
    return reslist

def flow_int_fl(n,hamiltonian,dl_list,qmax,cutoff,tlist,method='jit',norm=True,Hflow=False,store_flow=False,order=4,d=1.0,dim=1):
    """
    Diagonalise an initial interacting Hamiltonian, compute the forward integrals of motion and 
    also compute the infinite-temperature autocorrelation function.

    Parameters
        ----------
        n : integer
            Linear system size.
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        Hint : array, float
            Diagonal component of Hamiltonian
        Vint : array, float
            Off-diagonal component of Hamiltonian.
        dl_list : array, float
            List of flow times to use for the numerical integration.
        qmax : integer
            Maximum number of flow time steps.
        cutoff : float
            Threshold value below which off-diagonal elements are set to zero.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        output : dict
            Dictionary containing diagonal Hamiltonian ("H0_diag","Hint"), LIOM interaction coefficient ("LIOM Interactions"),
            the LIOM on central site ("LIOM") and the value of the second invariant of the flow ("Invariant").
    
    """
    H2,Hint = hamiltonian.H2_spinless,hamiltonian.H4_spinless

    k=1
    sol2 = jnp.array(H2,dtype=jnp.float64)
    sol4 = jnp.array(Hint,dtype=jnp.float64)
    if order == 6:
        sol6 = jnp.zeros((n,n,n,n,n,n),dtype=jnp.float64)
    J0 = 1

    # Initialise a density operator in the microscopic basis on the central site
    c1 = jnp.zeros((n),dtype=jnp.float64)
    c3 = jnp.zeros((n,n,n),dtype=jnp.float64)
    if dim == 2 and n%2 == 0:
        c1 = c1.at[n//2+int(jnp.sqrt(n))//2].set(1.0)
        print(c1)
    else:
        c1 = c1.at[n//2].set(1.0)
    if order == 6:
        c5 = jnp.zeros((n,n,n,n,n),dtype=jnp.float64)

    # Scrambling uses additional odeint calls and can dominate runtime for small n.
    # Keep default behavior (enabled), but allow disabling for profiling/memory tests.
    random_unitary = os.environ.get("PYFLOW_SCRAMBLE", "1") in ("1", "true", "True")
    res_cut = 0.5
    sctime = 0.

    # Optional tuning: odeint tolerances (defaults keep original behavior).
    try:
        _rtol = float(os.environ.get("PYFLOW_ODE_RTOL", "1e-8"))
    except Exception:
        _rtol = 1e-8
    try:
        _atol = float(os.environ.get("PYFLOW_ODE_ATOL", "1e-8"))
    except Exception:
        _atol = 1e-8

    # Optional timing summary
    timelog = os.environ.get("PYFLOW_TIMELOG", "0") in ("1", "true", "True")
    t_scramble = 0.0
    t_flow = 0.0
    t_scramble_extra = 0.0

    J0list = jnp.zeros(50,dtype=jnp.float32)
    J2list = jnp.zeros(50,dtype=jnp.float32)

    if random_unitary == True:
        _t0 = time.perf_counter()
        count = 0
        _,V0,_,Vint = extract_diag(sol2,sol4)
        J00 = jnp.max(jnp.abs(V0))
        
        reslist = res(n,sol2,res_cut,cutoff=cutoff)
        # print(reslist)
        while len(reslist)>0 and J0 > J00:
            if jnp.max(jnp.abs(sol4))>0.25:
                print('HALT SCRAMBLING')
                break
            dm_list = jnp.linspace(0,1.,5,endpoint=True)
            if order == 4:
                soln = ode(int_ode_random_O4,[sol2,sol4,c1,c3],np.array([dm_list[0],dm_list[-1]]),res_cut,rtol=_rtol,atol=_atol)
                sol2,sol4,c1,c3 = soln[0][-1],soln[1][-1],soln[2][-1],soln[3][-1]
                if jnp.max(jnp.abs(sol4))>0.25:
                    print('POTENTIAL CONVERGENCE ERROR')
                    break
            elif order == 6:
                soln = ode(int_ode_random_O6,[sol2,sol4,c1,c3,sol6,c5],np.array([dm_list[0],dm_list[-1]]),res_cut,rtol=_rtol,atol=_atol)
                sol2,sol4,c1,c3,sol6,c5 = soln[0][-1],soln[1][-1],soln[2][-1],soln[3][-1],soln[4][-1],soln[5][-1]
                if jnp.max(jnp.abs(sol4))>0.25:
                    print('POTENTIAL CONVERGENCE ERROR')
                    break
            count += 1
            # if count > 0:
            #     res_cut = 0.75
            reslist = res(n,sol2,res_cut,cutoff=cutoff)

        _,V0,_,Vint = extract_diag(sol2,sol4)
        J0 = jnp.max(jnp.abs(V0))
        J2 = jnp.max(jnp.abs(Vint))
        J0list = J0list.at[k%len(J0list)].set(J0)
        J2list = J2list.at[k%len(J2list)].set(jnp.max(jnp.abs(Vint)))
        # print(count,J0,J2,jnp.max(jnp.abs(sol4)),jnp.max(jnp.abs(c1)))
        print(c1)

        if count > 0:
            print('*****', k,jnp.max(jnp.abs(V0)),jnp.max(jnp.abs(Vint)),jnp.median(jnp.abs(Vint[Vint != 0.])))
        print('Number of random unitaries applied: ', count)
        if order == 4:
            print('Max LIOM terms: ',jnp.max(jnp.abs(c1)),jnp.max(jnp.abs(c3)))
        elif order == 6:
            print('Max LIOM terms: ',jnp.max(jnp.abs(c1)),jnp.max(jnp.abs(c3)),jnp.max(jnp.abs(c5)))
        t_scramble += time.perf_counter() - _t0

    #print(np.round(np.array(sol2),3))
    #print('SOL4',np.sort(sol4.reshape(n**4))[-10::])

    J0list = jnp.zeros(50)
    J2list = jnp.zeros(50,dtype=jnp.float32)
    scramble = 0
    trunc_err = 0.
    sol_norm = 0.
    rel_err = 0.
    n1 = 0.
    n2 = 0.
    sctime = 0

    # Ensure J0/J2 are defined even when scrambling is disabled.
    # (Previously they were only set inside the random_unitary branch.)
    _, V0, _, Vint = extract_diag(sol2, sol4)
    J0 = jnp.max(jnp.abs(V0))
    J2 = jnp.max(jnp.abs(Vint))
    J0list = J0list.at[k % len(J0list)].set(J0)
    J2list = J2list.at[k % len(J2list)].set(J2)

    _memlog("flow_int_fl:init", step=0, l=0.0, n=int(n), qmax=int(qmax), order=int(order), dim=int(dim))
    while k < len(dl_list)-1 and (J0 > cutoff or J2 > 1e-3):
        # Periodic memory sampling (RSS) to build a memory-vs-step trace.
        try:
            l_now = float(np.array(dl_list[k]))
        except Exception:
            l_now = None
        _memlog("flow_int_fl:loop", step=int(k), l=l_now, J0=float(J0), J2=float(J2))

        # Break loop if a situation is encountered where the quadratic part has decayed, but the 
        # quartic off-diagonal terms are unchanging. This likely reflects some sort of 2-particle
        # degeneracy that can't be removed by the transform, and may cause the sixth-order terms 
        # to diverge if the flow continues. (This is not physical, purely an artifact of the method.)

        if k>len(J2list) and J0 < cutoff and int(jnp.round(jnp.mean(J2list)*1e4)) == int(jnp.round(J2*1e4)):
            break
        if J0<1e-6 and jnp.median(jnp.abs(V0[V0 != 0]))<1e-8:
            print('V0 break',J0,jnp.max(jnp.abs(V0)),jnp.median(jnp.abs(V0[V0 != 0])))
            break

        _t0 = time.perf_counter()
        if order == 4:
            soln = ode(int_ode_ladder,[sol2,sol4,c1,c3,0.,0.],dl_list[k-1:k+1],rtol=_rtol,atol=_atol)
            sol2,sol4,c1,c3,n1,n2 = soln[0][-1],soln[1][-1],soln[2][-1],soln[3][-1],soln[4][-1],soln[5][-1]
        elif order == 6:
            soln = ode(int_ode_ladder,[sol2,sol4,sol6,c1,c3,c5,0.,0.],dl_list[k-1:k+1],rtol=_rtol,atol=_atol)
            sol2,sol4,sol6,c1,c3,c5,n1,n2 = soln[0][-1],soln[1][-1],soln[2][-1],soln[3][-1],soln[4][-1],soln[5][-1],soln[6][-1],soln[7][-1]
        t_flow += time.perf_counter() - _t0

        # print(np.where(np.abs(np.array(c3)) == np.abs(np.array(c3)).max()))
        # i1,j1,k1 = np.where(np.abs(np.array(c3)) == np.abs(np.array(c3)).max())
        # print(np.array(c3)[i1,j1,k1])
        # if k%100 == 0 and (jnp.max(jnp.abs(sol4))>0.1 or jnp.max(jnp.abs(c3))>0.1):
        #     print('POTENTIAL CONVERGENCE ERROR',k,jnp.max(jnp.abs(sol4)),jnp.max(jnp.abs(c3)))
        #     # break
        #     reslist = res(n,sol2,res_cut,cutoff=cutoff)
        #     while len(reslist)>0:
        #         print('*** SCRAMBLING STEP ***',reslist,res_cut)
        #         print(jnp.max(jnp.abs(sol2-np.diag(np.diag(sol2)))),jnp.max(jnp.abs(sol4)),jnp.max(jnp.abs(c3)))
        #         dm_list = jnp.linspace(0,1.,5,endpoint=True)
        #         if order == 4:
        #             soln2 = ode(int_ode_random_O4,[sol2,sol4,c1,c3],np.array([dm_list[0],dm_list[-1]]),res_cut,rtol=1e-8,atol=1e-8)
        #             sol2,sol4,c1,c3 = soln2[0][-1],soln2[1][-1],soln2[2][-1],soln2[3][-1]
        #         elif order == 6:
        #             soln2 = ode(int_ode_random_O6,[sol2,sol4,c1,c3,sol6,c5],np.array([dm_list[0],dm_list[-1]]),res_cut,rtol=1e-8,atol=1e-8)
        #             sol2,sol4,c1,c3,sol6,c5 = soln2[0][-1],soln2[1][-1],soln2[2][-1],soln2[3][-1],soln2[4][-1],soln2[5][-1]
        #         reslist = res(n,sol2,res_cut,cutoff=cutoff)


        _,V0,_,Vint = extract_diag(sol2,sol4)
        J0 = jnp.max(jnp.abs(V0))
        J2 = jnp.max(jnp.abs(Vint))
        J0list = J0list.at[k%len(J0list)].set(J0)
        J2list = J2list.at[k%len(J2list)].set(jnp.max(jnp.abs(Vint)))
        #print(k,J0,jnp.mean(J0list),J2,jnp.max(jnp.abs(sol4)),jnp.max(jnp.abs(c3)))
        if jnp.max(jnp.abs(sol4))>0.25 or jnp.max(jnp.abs(c3))>0.5:
            break

        # print(np.where(np.abs(np.array(V0)) == np.abs(np.array(V0)).max()))
        # i1,j1 = np.where(np.abs(np.array(V0)) == np.abs(np.array(V0)).max())
        # print(np.array(sol2)[i1,j1])
        #===========================================================================================
        # Add additional scrambling steps if flow becomes too slow.
        # (Triggers if J0 remains unchanged for a while, and is above some cutoff.)
        # An alternative would be to trigger if new resonances appear, but the code to scan for resonances
        # is extremely slow, and so it's better to only call it if the flow noticably slows down

        # print(k,int(jnp.round(jnp.mean(J0list)*1e4)),int(jnp.round(J0*1e4)),int(jnp.round(jnp.mean(J0list)*1e4)) == int(jnp.round(J0*1e4)))
        #sctime = 0
        if random_unitary and ((k > len(dl_list)//2 and k%50==0 and int(jnp.round(jnp.mean(J0list)*1e4)) == int(jnp.round(J0*1e4)) and J0 > 1e-3) or (jnp.max(jnp.abs(c3))>0.15 and J0 > 1e-3)):
            # if k > len(dl_list)//2:
            res_cut = 0.5
            reslist = res(n,sol2,res_cut,cutoff=cutoff)
            if len(reslist) == 0:
                reslist = res(n,sol2,res_cut,cutoff=0.25*cutoff)
                sctime += 1
            while len(reslist)>0:
                _t1 = time.perf_counter()
                print('*** SCRAMBLING STEP ***',reslist,res_cut)
                print(jnp.max(jnp.abs(sol2-np.diag(np.diag(sol2)))),jnp.max(jnp.abs(sol4)),jnp.max(jnp.abs(c3)))
                dm_list = jnp.linspace(0,5.,5,endpoint=True)
                if order == 4:
                    soln2 = ode(int_ode_random_O4,[sol2,sol4,c1,c3],np.array([dm_list[0],dm_list[-1]]),res_cut,rtol=_rtol,atol=_atol)
                    sol2,sol4,c1,c3 = soln2[0][-1],soln2[1][-1],soln2[2][-1],soln2[3][-1]
                elif order == 6:
                    soln2 = ode(int_ode_random_O6,[sol2,sol4,c1,c3,sol6,c5],np.array([dm_list[0],dm_list[-1]]),res_cut,rtol=_rtol,atol=_atol)
                    sol2,sol4,c1,c3,sol6,c5 = soln2[0][-1],soln2[1][-1],soln2[2][-1],soln2[3][-1],soln2[4][-1],soln2[5][-1]
                reslist = res(n,sol2,res_cut,cutoff=cutoff)
                scramble += 1
                t_scramble_extra += time.perf_counter() - _t1

                # Breaks if flow is slow, but no more resonances can be detected for an extended period
                # Indicates potential degeneracies that cannot be removed 
                if sctime > 25:
                    print('SCRAMBLE BREAK')
                    break
        #===========================================================================================
        
        trunc_err += (dl_list[k]-dl_list[k-1])*n2
        sol_norm += (dl_list[k]-dl_list[k-1])*n1
        rel_err += (dl_list[k]-dl_list[k-1])*(n2/n1)

        k += 1
    print(dl_list[k],k,J0,J2)
    dl_list = dl_list[0:k]
    print('No. of scrambling steps: ', scramble)
    print('* TRUNCATION ERROR *',trunc_err,sol_norm,trunc_err/sol_norm)
    if timelog:
        print(
            "TIMING(s): scramble_init=%.3f flow_ode=%.3f scramble_extra=%.3f total=%.3f"
            % (t_scramble, t_flow, t_scramble_extra, (t_scramble + t_flow + t_scramble_extra))
        )
    if order == 4:
        print('Max LIOM terms: ',jnp.max(jnp.abs(c1)),jnp.max(jnp.abs(c3)))
    elif order == 6:
        print('Max LIOM terms: ',jnp.max(jnp.abs(c1)),jnp.max(jnp.abs(c3)),jnp.max(jnp.abs(c5)))

    # Define final diagonal quadratic Hamiltonian
    H0_diag = soln[0][-1].reshape(n,n)
    print(jnp.sort(jnp.diag(H0_diag)))
    print('Max |V|: ',jnp.max(jnp.abs(H0_diag-jnp.diag(jnp.diag(H0_diag)))))

    # Define final diagonal quartic Hamiltonian
    Hint2 = soln[1][-1].reshape(n,n,n,n)   

    #Define LIOMs
    if order == 4:
        liom_fwd1 = soln[2][-1].reshape(n)
        liom_fwd3 = soln[3][-1].reshape(n,n,n)
    elif order == 6:
        liom_fwd1 = soln[3][-1].reshape(n)
        liom_fwd3 = soln[4][-1].reshape(n,n,n)
        liom_fwd5 = soln[5][-1].reshape(n,n,n,n,n)
        Hint6 = soln[2][-1].reshape(n,n,n,n,n,n)

    liom_fwd3 = np.array(liom_fwd3)
    for i in range(n):
        liom_fwd3[i,i,:] = np.zeros(n)

    liom_fwd1 = np.array(liom_fwd1,dtype=np.float64)
    liom_fwd3 = np.array(liom_fwd3,dtype=np.float64)

    # Build LIOM number operator from creation and annihilation operators
    # NB: All operators are real at this stage, as no time evolution has been performed
    l2 = np.einsum('a,b->ab',liom_fwd1,np.transpose(liom_fwd1))

    # Note the order of the contractions here, required to put the operators in the right order
    # The number operators are defined as c^{\dagger} c c^{\dagger} c
    # whereas the creation/annihilation operators are c^{\dagger} c^{\dagger} c / c^{\dagger} c c
    l4 = -np.einsum('a,bcd->acbd',liom_fwd1,np.transpose(liom_fwd3)) - np.einsum('abc,d->acbd',liom_fwd3,np.transpose(liom_fwd1))
    l4 += -np.einsum('abc,cde->adbe',liom_fwd3,np.transpose(liom_fwd3))
    # NOTE: l6 is an n^6 tensor. Even at n=25, float64 l6 is ~1.9 GiB and can OOM-kill the process.
    # Default: do NOT compute l6 unless explicitly requested.
    compute_l6 = os.environ.get("PYFLOW_COMPUTE_L6", "0") in ("1", "true", "True")
    if compute_l6 and n <= 12:
        l6 = np.einsum("abc,def->afbcde", liom_fwd3, np.transpose(liom_fwd3))
        if order == 6:
            liom_fwd5 = np.array(liom_fwd5, dtype=np.float64)
            l6 += -np.einsum("abcde,f->adbecf", liom_fwd5, jnp.transpose(liom_fwd1)) + -1 * np.einsum(
                "a,bcdef->adbecf", liom_fwd1, np.transpose(liom_fwd5)
            )
    else:
        l6 = jnp.array([0.])

    #print('MAX NUMBER OPERATOR TERMS',jnp.max(jnp.abs(l2)),jnp.max(jnp.abs(l4)),jnp.max(jnp.abs(l6)))
    # Hermitian check: disable for performance
    if n < 12:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for q in range(n):
                        if l4[i,j,k,q] != l4[q,k,j,i]:
                            print('L4 HERMITIAN ERROR',l4[i,j,k,q],l4[q,k,j,i])
                        if order == 6:
                            for l in range(n):
                                for m in range(n):
                                    if l6[i,j,k,q,l,m] != l6[m,l,q,k,j,i]:
                                        print('L6 HERMITIAN ERROR',l6[i,j,k,q,l,m], l6[m,l,q,k,j,i])
    
    H0_diag = np.array(H0_diag,dtype=np.float64)
    Hint2 = np.array(Hint2,dtype=np.float64)

    print(np.where(np.abs(np.array(l4)) == np.abs(np.array(l4)).max()))
    i1,j1,k1,q1 = np.where(np.abs(np.array(l4)) == np.abs(np.array(l4)).max())
    print(np.array(l4)[i1,j1,k1,q1])

    for i in range(n):
        l4[i,:,i,:] = 0
        l4[:,i,:,i] = 0
        if compute_l6 and n <= 12:
            l6[i,:,i,:,:,:] = 0
            l6[i,:,:,:,i,:] = 0
            l6[:,:,i,:,i,:] = 0
            l6[:,i,:,i,:,:] = 0
            l6[:,i,:,:,:,i] = 0
            l6[:,:,:,i,:,i] = 0

    # print('l2 diag BEFORE',np.diag(l2))
    # print('l2 trace BEFORE', np.sum(np.diag(l2)))
    for i in range(n):
        for j in range(n):
            # if i != j:
                l4[i,i,j,j] += -l4[i,j,j,i]
                l4[i,j,j,i] = 0.
    # print('l2 diag AFTER',np.diag(l2))
    # print('l2 trace AFTER', np.sum(np.diag(l2)))
    # l2 *= 1/np.sum(np.diag(l2))

    print('*****')
    print(np.where(np.abs(np.array(l4)) == np.abs(np.array(l4)).max()))
    i1,j1,k1,q1 = np.where(np.abs(np.array(l4)) == np.abs(np.array(l4)).max())
    print(np.array(l4)[i1,j1,k1,q1],np.array(l4)[q1,k1,j1,i1],np.array(l4)[i1,q1,k1,j1])

    HF = np.zeros((n,n))
    Hint2_diag = np.zeros((n,n,n,n))
    for i in range(n):
        for j in range(n):
            Hint2_diag[i,i,j,j] += Hint2[i,i,j,j]
            # if i != j:
            # If i == j, normal-ordering corrections kill this term, so letting the sum run over
            # all values of i,j here takes care of this for us
            Hint2_diag[i,i,j,j] += -Hint2[i,j,j,i]
            # H0_diag[i,i] += -Hint2[i,j,j,i]
            Hint2[i,j,j,i] = 0.
            Hint2[i,i,j,j] = 0.
            HF[i,j] = Hint2_diag[i,i,j,j]

    print(np.round(HF,3))

    Vint = Hint2.reshape(n**4)
    print('Max off-diagonal interaction terms', np.sort(np.abs(Vint))[-10::])

    lbits = []
    for q in range(1,n):
        lbits += [np.mean(np.abs(np.diag(HF,q)))]

    _memlog("flow_int_fl:postprocess_done", step=int(k), l=float(np.array(dl_list[min(k, len(dl_list)-1)])))

    # itc requires the (optional) dynamics module (QuSpin). For flow-only runs, return zeros.
    if dyn_itc is None:
        itc = np.zeros(len(tlist), dtype=np.float64)
        itc2 = np.zeros(len(tlist), dtype=np.float64)
    else:
        #itc = dyn_itc(n,tlist,l2,H0_diag,Hint=np.zeros((n,n,n,n)),num4=np.zeros((n,n,n,n)),num6=np.zeros((n,n,n,n,n,n)),int=True)
        itc, itc2 = dyn_itc(
            n,
            np.array(tlist, dtype=np.float64),
            l2,
            H0_diag,
            Hint=Hint2_diag,
            num4=l4,
            num6=l6,
            int=True,
        )

    output = {"H0_diag":np.array(H0_diag), "Hint":np.array(Hint2+Hint2_diag),"LIOM1_FWD":liom_fwd1,"LIOM3_FWD":liom_fwd3,"l2":l2,"l4":l4,"lbits":lbits}
    # output["nt_list"] = nt_list
    output["itc"] = itc
    output["itc_nonint"] = itc2
    output["truncation_err"] = [trunc_err,sol_norm,trunc_err/sol_norm,rel_err]
    output["dl_list"] = dl_list
    if order == 6 and n <= 36:
        output["H6"] = np.array(Hint6)
    return output

def flow_static_int_fwd(n,hamiltonian,dl_list,qmax,cutoff,method='jit',norm=True,Hflow=False,store_flow=False):
    """
    Diagonalise an initial interacting Hamiltonian and compute the integrals of motion.

    Note: this function does not compute the LIOMs in the conventional way. Rather, it starts with a local 
    operator in the initial basis and transforms it into the diagonal basis, essentially the inverse of 
    the process used to produce LIOMs conventionally. This bypasses the requirement to store the full 
    unitary transform in memory, meaning that only a single tensor of order O(L^4) needs to be stored at 
    each flow time step, dramatically increasing the accessible system sizes. However, use this with care
    as it is *not* a conventional LIOM, despite displaying essentially the same features, and should be 
    understood as such.

    Parameters
        ----------
        n : integer
            Linear system size.
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        Hint : array, float
            Diagonal component of Hamiltonian
        Vint : array, float
            Off-diagonal component of Hamiltonian.
        dl_list : array, float
            List of flow times to use for the numerical integration.
        qmax : integer
            Maximum number of flow time steps.
        cutoff : float
            Threshold value below which off-diagonal elements are set to zero.
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.
        norm : bool, optional
            Specify whether to use non-perturbative normal-ordering corrections (True) or not (False).
            This may take a lot longer to run, but typically improves accuracy. Care must be taken to 
            ensure that use of normal-ordering is warranted and that the contractions are computed with 
            respect to an appropriate state.
        Hflow : bool, optional
            Choose whether to use pre-computed generator or re-compute eta on the fly.

        Returns
        -------
        output : dict
            Dictionary containing diagonal Hamiltonian ("H0_diag","Hint"), LIOM interaction coefficient ("LIOM Interactions"),
            the LIOM on central site ("LIOM") and the value of the second invariant of the flow ("Invariant").
    
    """

    H2,Hint = hamiltonian.H2_spinless,hamiltonian.H4_spinless

    if store_flow == True:
        # Initialise array to hold solution at all flow times
        flow_list = jnp.zeros((qmax,2*(n**2+n**4)))

    # print('Memory64 required: MB', sol_int.nbytes/10**6)

    # Store initial interaction value and trace of initial H^2 for later error estimation
    delta = jnp.max(Hint)
    e1 = jnp.trace(jnp.dot(H2,H2))
    
    # Define integrator
    r_int = ode(int_ode_fwd).set_integrator('dopri5',nsteps=100,rtol=10**(-6),atol=10**(-6))
    
    # Set initial conditions
    init = jnp.zeros(2*(n**2+n**4),dtype=jnp.float32)
    init[:n**2] = ((H2)).reshape(n**2)
    init[n**2:n**2+n**4] = (Hint).reshape(n**4)

    # Initialise a density operator in the diagonal basis on the central site
    init_liom = jnp.zeros(n**2+n**4)
    init_liom2 = jnp.zeros((n,n))
    init_liom2[n//2,n//2] = 1.0
    init_liom[:n**2] = init_liom2.reshape(n**2)
    init[n**2+n**4:] = init_liom
    if store_flow == True:
        flow_list[0] = init

    r_int.set_initial_value(init,dl_list[0])
    r_int.set_f_params(n,[],method,norm,Hflow)

    # Numerically integrate the flow equations
    k = 1                       # Flow timestep index
    J0 = 10.                    # Seed value for largest off-diagonal term
    decay = 1
    index_list = indices(n)
    last_progress = 0
    print(f"        Flow integration (fwd): max_steps={qmax}, cutoff={cutoff:.2e}")
    # Integration continues until qmax is reached or all off-diagonal elements decay below cutoff
    while r_int.successful() and k < qmax-1 and decay == 1:
        if Hflow == True:
            r_int.integrate(dl_list[k])
            step = r_int.y
            if store_flow == True:
               flow_list[k] = step

        decay = cut(step,n,cutoff,index_list)

        mat = step[:n**2].reshape(n,n)
        off_diag = mat-jnp.diag(jnp.diag(mat))
        J0 = max(off_diag.reshape(n**2))
        
        # Progress printing every 10% or every 50 steps
        progress = (k * 100) // qmax
        if progress >= last_progress + 10 or k % 50 == 0:
            print(f"        Step {k}/{qmax} ({progress}%) | l={dl_list[k]:.4f} | off-diag={J0:.2e}", flush=True)
            last_progress = progress
        k += 1 
    print(f"        Converged at step {k-1}, final off-diag={J0:.2e}")

    # Truncate solution list and flow time list to max timestep reached
    dl_list = dl_list[:k-1]
    if store_flow == True:
        flow_list = flow_list[:k-1]
    
    liom = step[n**2+n**4::]
    step = step[:n**2+n**4]

    # Define final diagonal quadratic Hamiltonian
    H0_diag = step[:n**2].reshape(n,n)
    # Define final diagonal quartic Hamiltonian
    Hint2 = step[n**2::].reshape(n,n,n,n)   
    # Extract only the density-density terms of the final quartic Hamiltonian, as a matrix                     
    HFint = jnp.zeros(n**2).reshape(n,n)
    for i in range(n):
        for j in range(n):
            HFint[i,j] = Hint2[i,i,j,j]
            HFint[i,j] += -Hint2[i,j,j,i]

    # Compute the difference in the second invariant of the flow at start and end
    # This acts as a measure of the unitarity of the transform
    Hflat = HFint.reshape(n**2)
    inv = 2*jnp.sum([d**2 for d in Hflat])
    e2 = jnp.trace(jnp.dot(H0_diag,H0_diag))
    inv2 = jnp.abs(e1 - e2 + ((2*delta)**2)*(n-1) - inv)/jnp.abs(e2+((2*delta)**2)*(n-1))

    # Compute the l-bit interactions from the density-density terms in the final Hamiltonian
    lbits = jnp.zeros(n-1)
    for q in range(1,n):
        lbits[q-1] = jnp.median(jnp.log10(jnp.abs(jnp.diag(HFint,q)+jnp.diag(HFint,-q))/2.))

    # liom_all = jnp.sum([j**2 for j in liom])
    f2 = jnp.sum([j**2 for j in liom[0:n**2]])
    f4 = jnp.sum([j**2 for j in liom[n**2::]])
    print('LIOM',f2,f4)
    print('Hint max',jnp.max(jnp.abs(Hint2)))

    output = {"H0_diag":H0_diag, "Hint":Hint2,"LIOM Interactions":lbits,"LIOM":liom,"Invariant":inv2}
    if store_flow == True:
        output["flow"] = flow_list
        output["dl_list"] = dl_list

        # Free up some memory
        del flow_list
        gc.collect()

    return output

def flow_dyn(n,hamiltonian,num,dl_list,qmax,cutoff,tlist,method='jit',store_flow=False):
    """
    Diagonalise an initial non-interacting Hamiltonian and compute the quench dynamics.

        Parameters
        ----------
        n : integer
            Linear system size.
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        Hint : array, float
            Diagonal component of Hamiltonian
        Vint : array, float
            Off-diagonal component of Hamiltonian.
        num : array, float
            Density operator n_i(t=0) to be time-evolved.
        dl_list : array, float
            List of flow times to use for the numerical integration.
        qmax : integer
            Maximum number of flow time steps.
        cutoff : float
            Threshold value below which off-diagonal elements are set to zero.
        tlist : array
            List of timesteps to return time-evolved operator n_i(t).
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.

        Returns
        -------
        sol : array, float
            Final (diagonal) Hamiltonian
        central : array, float
            Local integral of motion (LIOM) computed on the central lattice site of the chain
    
    """
    H2 = hamiltonian.H2_spinless

    # Initialise array to hold solution at all flow times
    sol = jnp.zeros((qmax,n**2))
    sol[0] = (H2).reshape(n**2)

    # Define integrator
    r = ode(nonint_ode).set_integrator('dopri5', nsteps=1000)
    r.set_initial_value((H2).reshape(n**2),dl_list[0])
    r.set_f_params(n,method)
    
    # Numerically integrate the flow equations
    k = 1                       # Flow timestep index
    J0 = 10.                    # Seed value for largest off-diagonal term
    # Integration continues until qmax is reached or all off-diagonal elements decay below cutoff
    while r.successful() and k < qmax-1 and J0 > cutoff:
        r.integrate(dl_list[k])
        sol[k] = r.y
        mat = sol[k].reshape(n,n)
        off_diag = mat-jnp.diag(jnp.diag(mat))
        J0 = max(jnp.abs(off_diag.reshape(n**2)))
        k += 1
    print(k,J0)
    sol=sol[0:k-1]
    dl_list= dl_list[0:k-1]

    # Initialise a density operator in the diagonal basis on the central site
    # liom = jnp.zeros((qmax,n**2))
    init_liom = jnp.zeros((n,n))
    init_liom[n//2,n//2] = 1.0
    # liom[0,:n**2] = init_liom.reshape(n**2)
    
    # Reverse list of flow times in order to conduct backwards integration
    dl_list = dl_list[::-1]

    # Define integrator for density operator
    n_int = ode(liom_ode).set_integrator('dopri5', nsteps=100)
    n_int.set_initial_value(init_liom.reshape(n**2),dl_list[0])

    # Numerically integrate the flow equations for the density operator 
    # Integral goes from l -> infinity to l=0 (i.e. from diagonal basis to original basis)
    k0=1
    while n_int.successful() and k0 < len(dl_list[:k]):
        n_int.set_f_params(n,sol[-k0],method)
        n_int.integrate(dl_list[k0])
        liom = n_int.y
        k0 += 1
    
    # Take final value for the transformed density operator and reshape to a matrix
    # central = (liom.reshape(n,n))
    
    # Invert dl again back to original
    dl_list = dl_list[::-1] 

    # Define integrator for density operator again
    # This time we integrate from l=0 to l -> infinity
    num_int = ode(liom_ode).set_integrator('dopri5', nsteps=100)
    num_int.set_initial_value(num.reshape(n**2),dl_list[0])
    k0=1
    num=jnp.zeros((k,n**2))
    while num_int.successful() and k0 < k-1:
        num_int.set_f_params(n,sol[k0],method)
        num_int.integrate(dl_list[k0])
        num[k0] = num_int.y
        k0 += 1
    num = num[:k0-1]

    # Run non-equilibrium dynamics following a quench from CDW state
    # Returns answer *** in LIOM basis ***
    evolist = dyn_con(n,num[-1],sol[-1],tlist,method=method)
    print(evolist)

    # For each timestep, integrate back from l -> infinity to l=0
    # i.e. from LIOM basis back to original microscopic basis
    num_t_list = jnp.zeros((len(tlist),n**2))
    dl_list = dl_list[::-1] # Reverse dl for backwards flow
    for t0 in range(len(tlist)):
        num_int = ode(liom_ode).set_integrator('dopri5', nsteps=100)
        num_int.set_initial_value(evolist[t0],dl_list[0])

        k0=1
        while num_int.successful() and k0 < k-1:
            num_int.set_f_params(n,sol[-k0],method)
            num_int.integrate(dl_list[k0])
            k0 += 1
        num_t_list[t0] = num_int.y
        
    # Initialise a list to store the expectation value of time-evolved density operator at each timestep
    nlist = jnp.zeros(len(tlist))

    # Set up initial state as a CDW
    list1 = jnp.array([1. for i in range(n//2)])
    list2 = jnp.array([0. for i in range(n//2)])
    state = jnp.array([val for pair in zip(list1,list2) for val in pair])
    
    # Compute the expectation value <n_i(t)> for each timestep t
    n2list = num_t_list[::,:n**2]
    for t0 in range(len(tlist)):
        mat = n2list[t0].reshape(n,n)
        for i in range(n):
            nlist[t0] += (mat[i,i]*state[i]**2).real

    output = {"H0_diag":sol[-1].reshape(n,n),"LIOM":liom,"Invariant":0,"Density Dynamics":nlist}
    if store_flow == True:
        output["flow"] = sol
        output["dl_list"] = dl_list[::-1]

    return output

     
def flow_dyn_int_singlesite(n,hamiltonian,num,num_int,dl_list,qmax,cutoff,tlist,method='jit',store_flow=False):
    """
    Diagonalise an initial interacting Hamiltonian and compute the quench dynamics.

    This function will return a time-evolved number operator.

        Parameters
        ----------
        n : integer
            Linear system size.
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        Hint : array, float
            Diagonal component of Hamiltonian
        Vint : array, float
            Off-diagonal component of Hamiltonian.
        num : array, float
            Density operator n_i(t=0) to be time-evolved.
        dl_list : array, float
            List of flow times to use for the numerical integration.
        qmax : integer
            Maximum number of flow time steps.
        cutoff : float
            Threshold value below which off-diagonal elements are set to zero.
        tlist : array
            List of timesteps to return time-evolved operator n_i(t).
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.

        Returns
        -------
        sol : array, float
            Final (diagonal) Hamiltonian
        central : array, float
            Local integral of motion (LIOM) computed on the central lattice site of the chain
    
    """
    H2 = hamiltonian.H2_spinless
    H4=hamiltonian.H4_spinless

    # Initialise array to hold solution at all flow times
    sol_int = jnp.zeros((qmax,n**2+n**4),dtype=jnp.float32)
    # print('Memory64 required: MB', sol_int.nbytes/10**6)
    
    # Initialise the first flow timestep
    init = jnp.zeros(n**2+n**4,dtype=jnp.float32)
    init[:n**2] = (H2).reshape(n**2)
    init[n**2:] = (H4).reshape(n**4)
    sol_int[0] = init

    # Define integrator
    r_int = ode(int_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
    r_int.set_initial_value(init,dl_list[0])
    r_int.set_f_params(n,[],method)
    
    
    # Numerically integrate the flow equations
    k = 1                       # Flow timestep index
    J0 = 10.                    # Seed value for largest off-diagonal term
    # Integration continues until qmax is reached or all off-diagonal elements decay below cutoff
    while r_int.successful() and k < qmax-1 and J0 > cutoff:
        r_int.integrate(dl_list[k])
        sol_int[k] = r_int.y
        mat = sol_int[k,0:n**2].reshape(n,n)
        off_diag = mat-jnp.diag(jnp.diag(mat))
        J0 = max(off_diag.reshape(n**2))
        k += 1

    # Truncate solution list and flow time list to max timestep reached
    sol_int=sol_int[:k-1]
    dl_list=dl_list[:k-1]

    # Define final Hamiltonian, for function return
    H0final,Hintfinal = sol_int[-1,:n**2].reshape(n,n),sol_int[-1,n**2::].reshape(n,n,n,n)
    
    # Define final diagonal quadratic Hamiltonian
    H0_diag = sol_int[-1,:n**2].reshape(n,n)
    # Define final diagonal quartic Hamiltonian
    Hint = sol_int[-1,n**2::].reshape(n,n,n,n)   
    # Extract only the density-density terms of the final quartic Hamiltonian, as a matrix                     
    HFint = jnp.zeros(n**2).reshape(n,n)
    for i in range(n):
        for j in range(n):
            HFint[i,j] = Hint[i,i,j,j]
            HFint[i,j] += -Hint[i,j,j,i]

    # Compute the l-bit interactions from the density-density terms in the final Hamiltonian
    lbits = jnp.zeros(n-1)
    for q in range(1,n):
        lbits[q-1] = jnp.median(jnp.log10(jnp.abs(jnp.diag(HFint,q)+jnp.diag(HFint,-q))/2.))

    # Initialise a density operator in the diagonal basis on the central site
    liom = jnp.zeros((k,n**2+n**4),dtype=jnp.float32)
    init_liom = jnp.zeros((n,n))
    init_liom[n//2,n//2] = 1.0
    liom[0,:n**2] = init_liom.reshape(n**2)
    
    # Reverse list of flow times in order to conduct backwards integration
    dl_list = dl_list[::-1]

    # Define integrator for density operator
    n_int = ode(liom_ode).set_integrator('dopri5', nsteps=50)
    n_int.set_initial_value(liom[0],dl_list[0])

    # Numerically integrate the flow equations for the density operator 
    # Integral goes from l -> infinity to l=0 (i.e. from diagonal basis to original basis)
    k0=1
    while n_int.successful() and k0 < k-1:
        n_int.set_f_params(n,sol_int[-k0],method)
        n_int.integrate(dl_list[k0])
        liom[k0] = n_int.y
        k0 += 1

    # Take final value for the transformed density operator and reshape quadratic part to a matrix
    central = (liom[k0-1,:n**2]).reshape(n,n)

    # Invert dl again back to original
    dl_list = dl_list[::-1] 

    # Define integrator for density operator again
    # This time we integrate from l=0 to l -> infinity
    num = jnp.zeros((k,n**2+n**4),dtype=jnp.float32)
    num_init = jnp.zeros((n,n))
    num_init[n//2,n//2] = 1.0
    num[0,0:n**2] = num_init.reshape(n**2)
    num_int = ode(liom_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
    num_int.set_initial_value(num[0],dl_list[0])

    # Integrate the density operator
    k0=1
    while num_int.successful() and k0 < k-1:
        num_int.set_f_params(n,sol_int[k0],method)
        num_int.integrate(dl_list[k0])
        k0 += 1
    num = num_int.y

    # Run non-equilibrium dynamics following a quench from CDW state
    # Returns answer *** in LIOM basis ***
    evolist2 = dyn_exact(n,num,sol_int[-1],tlist)
    # evolist2 = dyn_con(n,num,sol_int[-1],tlist)
    
    # For each timestep, integrate back from l -> infinity to l=0
    # i.e. from LIOM basis back to original microscopic basis
    num_t_list2 = jnp.zeros((len(tlist),n**2+n**4),dtype=jnp.complex128)
    dl_list = dl_list[::-1] # Reverse dl for backwards flow
    for t0 in range(len(tlist)):
        
        num_int = ode(liom_ode).set_integrator('dopri5',nsteps=100,atol=10**(-8),rtol=10**(-8))
        num_int.set_initial_value(evolist2[t0],dl_list[0])
        k0=1
        while num_int.successful() and k0 < k-1:
            num_int.set_f_params(n,sol_int[-k0],method,True)
            num_int.integrate(dl_list[k0])
            k0 += 1
        num_t_list2[t0] = (num_int.y).real

    # Initialise a list to store the expectation value of time-evolved density operator at each timestep
    nlist2 = jnp.zeros(len(tlist))

    # Set up initial state as a CDW
    list1 = jnp.array([1. for i in range(n//2)])
    list2 = jnp.array([0. for i in range(n//2)])
    state = jnp.array([val for pair in zip(list1,list2) for val in pair])
    
    # Compute the expectation value <n_i(t)> for each timestep t
    n2list = num_t_list2[::,:n**2]
    n4list = num_t_list2[::,n**2:]
    for t0 in range(len(tlist)):
        mat = n2list[t0].reshape(n,n)
        mat4 = n4list[t0].reshape(n,n,n,n)
        for i in range(n):
            nlist2[t0] += (mat[i,i]*state[i]).real
            for j in range(n):
                if i != j:
                    nlist2[t0] += (mat4[i,i,j,j]*state[i]*state[j]).real
                    nlist2[t0] += -(mat4[i,j,j,i]*state[i]*state[j]).real
    print(nlist2)

    output = {"H0_diag":H0_diag,"Hint":Hintfinal,"LIOM Interactions":lbits,"LIOM":liom,"Invariant":0,"Density Dynamics":nlist2}
    if store_flow == True:
        output["flow"] = sol_int
        output["dl_list"] = dl_list[::-1]

    return output
 
    
def flow_dyn_int_imb(n,hamiltonian,num,num_int,dl_list,qmax,cutoff,tlist,method='jit',store_flow=False):
    """
    Diagonalise an initial interacting Hamiltonian and compute the quench dynamics.

    This function will return the imbalance following a quench, which involves computing the 
    non-equilibrium dynamics of the densiy operator on every single lattice site.

        Parameters
        ----------
        n : integer
            Linear system size.
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        Hint : array, float
            Diagonal component of Hamiltonian
        Vint : array, float
            Off-diagonal component of Hamiltonian.
        num : array, float
            Density operator n_i(t=0) to be time-evolved.
        dl_list : array, float
            List of flow times to use for the numerical integration.
        qmax : integer
            Maximum number of flow time steps.
        cutoff : float
            Threshold value below which off-diagonal elements are set to zero.
        tlist : array
            List of timesteps to return time-evolved operator n_i(t).
        method : string, optional
            Specify which method to use to generate the RHS of the flow equations.
            Method choices are 'einsum', 'tensordot', 'jit' and 'vec'.
            The first two are built-in NumPy methods, while the latter two are custom coded for speed.

        Returns
        -------
        sol : array, float
            Final (diagonal) Hamiltonian
        central : array, float
            Local integral of motion (LIOM) computed on the central lattice site of the chain
    
    """

    H2 = hamiltonian.H2_spinless
    H4 = hamiltonian.H4_spinless

    # Initialise array to hold solution at all flow times
    sol_int = jnp.zeros((qmax,n**2+n**4),dtype=jnp.float32)
    # print('Memory64 required: MB', sol_int.nbytes/10**6)
    
    # Initialise the first flow timestep
    init = jnp.zeros(n**2+n**4,dtype=jnp.float32)
    init[:n**2] = (H2).reshape(n**2)
    init[n**2:] = (H4).reshape(n**4)
    sol_int[0] = init

    # Define integrator
    r_int = ode(int_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
    r_int.set_initial_value(init,dl_list[0])
    r_int.set_f_params(n,[],method)
    
    
    # Numerically integrate the flow equations
    k = 1                       # Flow timestep index
    J0 = 10.                    # Seed value for largest off-diagonal term
    # Integration continues until qmax is reached or all off-diagonal elements decay below cutoff
    while r_int.successful() and k < qmax-1 and J0 > cutoff:
        r_int.integrate(dl_list[k])
        sol_int[k] = r_int.y
        mat = sol_int[k,0:n**2].reshape(n,n)
        off_diag = mat-jnp.diag(jnp.diag(mat))
        J0 = max(off_diag.reshape(n**2))
        k += 1

    # Truncate solution list and flow time list to max timestep reached
    sol_int=sol_int[:k-1]
    dl_list=dl_list[:k-1]

    # Define final Hamiltonian, for function return
    H0final,Hintfinal = sol_int[-1,:n**2].reshape(n,n),sol_int[-1,n**2::].reshape(n,n,n,n)
    
    # Define final diagonal quadratic Hamiltonian
    H0_diag = sol_int[-1,:n**2].reshape(n,n)
    # Define final diagonal quartic Hamiltonian
    Hint = sol_int[-1,n**2::].reshape(n,n,n,n)   
    # Extract only the density-density terms of the final quartic Hamiltonian, as a matrix                     
    HFint = jnp.zeros(n**2).reshape(n,n)
    for i in range(n):
        for j in range(n):
            HFint[i,j] = Hint[i,i,j,j]
            HFint[i,j] += -Hint[i,j,j,i]

    # Compute the l-bit interactions from the density-density terms in the final Hamiltonian
    lbits = jnp.zeros(n-1)
    for q in range(1,n):
        lbits[q-1] = jnp.median(jnp.log10(jnp.abs(jnp.diag(HFint,q)+jnp.diag(HFint,-q))/2.))

    # Initialise a density operator in the diagonal basis on the central site
    liom = jnp.zeros((k,n**2+n**4),dtype=jnp.float32)
    init_liom = jnp.zeros((n,n))
    init_liom[n//2,n//2] = 1.0
    liom[0,:n**2] = init_liom.reshape(n**2)
    
    # Reverse list of flow times in order to conduct backwards integration
    dl_list = dl_list[::-1]
    
    # Set up initial state as a CDW
    list1 = jnp.array([1. for i in range(n//2)])
    list2 = jnp.array([0. for i in range(n//2)])
    state = jnp.array([val for pair in zip(list1,list2) for val in pair])
    
    # Define lists to store the time-evolved density operators on each lattice site
    # 'imblist' will include interaction effects
    # 'imblist2' includes only single-particle effects
    # Both are kept to check for diverging interaction terms
    imblist = jnp.zeros((n,len(tlist)))
    imblist2 = jnp.zeros((n,len(tlist)))

    # Compute the time-evolution of the number operator on every site
    for site in range(n):
        # Initialise operator to be time-evolved
        num = jnp.zeros((k,n**2+n**4))
        num_init = jnp.zeros((n,n),dtype=jnp.float32)
        num_init[site,site] = 1.0

        num[0,0:n**2] = num_init.reshape(n**2)
        
            # Invert dl again back to original
        dl_list = dl_list[::-1]

        # Define integrator for density operator again
        # This time we integrate from l=0 to l -> infinity
        num_int = ode(liom_ode).set_integrator('dopri5', nsteps=50,atol=10**(-6),rtol=10**(-3))
        num_int.set_initial_value(num[0],dl_list[0])
        k0=1
        while num_int.successful() and k0 < k-1:
            num_int.set_f_params(n,sol_int[k0],method)
            num_int.integrate(dl_list[k0])
            # liom[k0] = num_int.y
            k0 += 1
        num = num_int.y
        
        # Run non-equilibrium dynamics following a quench from CDW state
        # Returns answer *** in LIOM basis ***
        evolist2 = dyn_exact(n,num,sol_int[-1],tlist)
        dl_list = dl_list[::-1] # Reverse the flow
        
        num_t_list2 = jnp.zeros((len(tlist),n**2+n**4))
        # For each timestep, integrate back from l -> infinity to l=0
        # i.e. from LIOM basis back to original microscopic basis
        for t0 in range(len(tlist)):
            
            num_int = ode(liom_ode).set_integrator('dopri5',nsteps=50,atol=10**(-8),rtol=10**(-8))
            num_int.set_initial_value(evolist2[t0],dl_list[0])
            k0=1
            while num_int.successful() and k0 < k-1:
                num_int.set_f_params(n,sol_int[-k0],method,True)
                num_int.integrate(dl_list[k0])
                k0 += 1
            num_t_list2[t0] = num_int.y
        
        # Initialise lists to store the expectation value of time-evolved density operator at each timestep
        nlist = jnp.zeros(len(tlist))
        nlist2 = jnp.zeros(len(tlist))
        
        # Compute the expectation value <n_i(t)> for each timestep t
        n2list = num_t_list2[::,:n**2]
        n4list = num_t_list2[::,n**2:]
        for t0 in range(len(tlist)):
            mat = n2list[t0].reshape(n,n)
            mat4 = n4list[t0].reshape(n,n,n,n)
            # phaseMF = 0.
            for i in range(n):
                # nlist[t0] += (mat[i,i]*state[i]**2).real
                nlist[t0] += (mat[i,i]*state[i]).real
                nlist2[t0] += (mat[i,i]*state[i]).real
                for j in range(n):
                    if i != j:
                        nlist[t0] += (mat4[i,i,j,j]*state[i]*state[j]).real
                        nlist[t0] += -(mat4[i,j,j,i]*state[i]*state[j]).real
                        
        imblist[site] = ((-1)**site)*nlist/n
        imblist2[site] = ((-1)**site)*nlist2/n

    # Compute the imbalance over the entire system
    # Note that the (-1)^i factors are included in imblist already
    imblist = 2*jnp.sum(imblist,axis=0)
    imblist2 = 2*jnp.sum(imblist2,axis=0)

    output = {"H0_diag":H0_diag,"Hint":Hintfinal,"LIOM Interactions":lbits,"LIOM":liom,"Invariant":0,"Imbalance":imblist}
    if store_flow == True:
        output["flow"] = sol_int
        output["dl_list"] = dl_list[::-1]

    return output

#------------------------------------------------------------------------------
# Function for benchmarking the non-interacting system using 'einsum'
def flow_einsum_nonint(H0,V0,dl):
    """ Benchmarking function to diagonalise H for a non-interacting system using NumPy's einsum function.

        This function is used to test the routines included in contract.py by explicitly calling 
        the 'einsum' function, which is a slow but very transparent way to do the matrix/tensor contractions.

        Parameters
        ----------
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        dl : float
            Size of step in flow time (dl << 1)
    
    """
      
    startTime = datetime.now()
    q = 0
    while jnp.max(jnp.abs(V0))>10**(-2):
    
        # Non-interacting generator
        eta0 = jnp.einsum('ij,jk->ik',H0,V0) - jnp.einsum('ki,ij->kj',V0,H0,optimize=True)
        
        # Flow of non-interacting terms
        dH0 = jnp.einsum('ij,jk->ik',eta0,(H0+V0)) - jnp.einsum('ki,ij->kj',(H0+V0),eta0,optimize=True)
    
        # Update non-interacting terms
        H0 = H0+dl*jnp.diag(jnp.diag(dH0))
        V0 = V0 + dl*(dH0-jnp.diag(jnp.diag(dH0)))

        q += 1

    print('***********')
    print('FE time - einsum',datetime.now()-startTime)
    print('Max off diagonal element: ', jnp.max(jnp.abs(V0)))
    print(jnp.sort(jnp.diag(H0)))

#------------------------------------------------------------------------------  
# Function for benchmarking the non-interacting system using 'tensordot'
def flow_tensordot_nonint(H0,V0,dl):  

    """ Benchmarking function to diagonalise H for a non-interacting system using NumPy's tensordot function.

        This function is used to test the routines included in contract.py by explicitly calling 
        the 'tensordot' function, which is slightly faster than einsum but also less transparent.

        Parameters
        ----------
        H0 : array, float
            Diagonal component of Hamiltonian
        V0 : array, float
            Off-diagonal component of Hamiltonian.
        dl : float
            Size of step in flow time (dl << 1)
    
    """   

    startTime = datetime.now()
    q = 0
    while jnp.max(jnp.abs(V0))>10**(-3):
    
        # Non-interacting generator
        eta = jnp.tensordot(H0,V0,axes=1) - jnp.tensordot(V0,H0,axes=1)
        
        # Flow of non-interacting terms
        dH0 = jnp.tensordot(eta,H0+V0,axes=1) - jnp.tensordot(H0+V0,eta,axes=1)
    
        # Update non-interacting terms
        H0 = H0+dl*jnp.diag(jnp.diag(dH0))
        V0 = V0 + dl*(dH0-jnp.diag(jnp.diag(dH0)))

        q += 1
        
    print('***********')
    print('FE time - Tensordot',datetime.now()-startTime)
    print('Max off diagonal element: ', jnp.max(jnp.abs(V0)))
    print(jnp.sort(jnp.diag(H0)))

def flow_static_int_ckpt(n,hamiltonian,dl_list,qmax,cutoff,method='tensordot',norm=False,Hflow=False,store_flow=False):
    forced = _force_steps()
    if forced is not None:
        dl_list = dl_list[: min(len(dl_list), forced + 1)]
        print(f"        [FORCE_STEPS] Running exactly {len(dl_list)-1} steps (ignoring cutoff)")
    """
    针对大系统优化的版本：使用 Checkpointing (检查点) 策略。
    """
    H2, Hint = hamiltonian.H2_spinless, hamiltonian.H4_spinless

    # === 配置参数 ===
    # 检查点间隔：数值越大，越省内存，但重算时间越长。
    # 对于 n=64，建议设为 20-50。
    ckpt_step = 20  
    
    print(f'Starting Flow with Checkpointing. System size n={n}, Checkpoint interval={ckpt_step}')
    
    # 1. 前向流：只存储稀疏的 Checkpoints
    # ------------------------------------------------------------------
    # checkpoints 列表用于存储关键帧：(step_index, H2_state, Hint_state)
    checkpoints = []
    
    # 初始化
    curr_H2 = jnp.array(H2)
    curr_Hint = jnp.array(Hint)
    
    # 记录原始dtype以避免在numpy转换时提升精度
    orig_dtype = curr_H2.dtype
    
    # 存入第 0 步
    checkpoints.append((0, np.array(curr_H2, dtype=orig_dtype), np.array(curr_Hint, dtype=orig_dtype)))
    
    k = 1
    J0 = 1.0
    
    # 只需要两个变量在内存中滚动，不需要大数组
    # 使用 jax.jit 编译核心步进函数以加速
    # 注意：这里为了简化展示逻辑，仍使用 ode 接口，实际循环中开销很小
    
    # Keep ODE tolerances consistent with standard flow_static_int
    _rtol = float(os.environ.get('PYFLOW_ODE_RTOL', '1e-6'))
    _atol = float(os.environ.get('PYFLOW_ODE_ATOL', '1e-6'))
    print(f"        Forward integration (checkpointing): max_steps={len(dl_list)}, cutoff={cutoff:.2e}, rtol={_rtol:.0e}, atol={_atol:.0e}")
    last_progress = 0
    approx = _approx_enabled()
    _approx_cache: dict = {}
    while k < len(dl_list) and ((forced is not None) or (J0 > cutoff)):
        # 计算当前步长
        steps = np.linspace(dl_list[k-1], dl_list[k], num=2, endpoint=True)
        
        # 前向积分一步
        if approx and (not norm):
            curr_H2, curr_Hint = _approx_step_h2_h4(
                curr_H2,
                curr_Hint,
                float(dl_list[k-1]),
                float(dl_list[k]),
                step_idx=k,
                method=method,
                cache=_approx_cache,
                norm=norm,
                Hflow=True,
            )
        else:
            soln = ode(int_ode, [curr_H2, curr_Hint], steps, rtol=_rtol, atol=_atol)
            curr_H2 = soln[0][-1]
            curr_Hint = soln[1][-1]
        
        # 检查是否需要存档
        if k % ckpt_step == 0:
            checkpoints.append((k, np.array(curr_H2, dtype=orig_dtype), np.array(curr_Hint, dtype=orig_dtype)))

        # 收敛检查
        J0 = jnp.max(jnp.abs(curr_H2 - jnp.diag(jnp.diag(curr_H2))))

        # Progress printing every 10% or every 50 steps
        progress = (k * 100) // len(dl_list)
        if progress >= last_progress + 10 or k % 50 == 0:
            print(f"        Step {k}/{len(dl_list)} ({progress}%) | l={dl_list[k]:.4f} | off-diag={J0:.2e} | ckpts={len(checkpoints)}", flush=True)
            last_progress = progress

        # Add memlog record
        if k % 10 == 0:
            memlog("flow:step", step=k, mode="checkpoint")
        k += 1
    
    # 记录实际结束的步数
    final_k = k
    # 确保存入最后一步（如果没存过）
    if (final_k-1) % ckpt_step != 0:
        checkpoints.append((final_k-1, np.array(curr_H2, dtype=orig_dtype), np.array(curr_Hint, dtype=orig_dtype)))
        
    print(f"        Forward pass converged at step {final_k-1}, total checkpoints: {len(checkpoints)}")

    # 截断 dl_list 到实际结束位置
    dl_list_final = dl_list[:final_k]

    # 提取最终对角化结果
    H0_diag = curr_H2
    Hint2 = curr_Hint
    
    # 2. 准备 LIOM 计算
    # ------------------------------------------------------------------
    # 提取有效相互作用
    HFint = jnp.zeros(n**2).reshape(n,n)
    for i in range(n):
        for j in range(n):
            HFint = HFint.at[i,j].set(Hint2[i,i,j,j] - Hint2[i,j,j,i])

    lbits = jnp.zeros(n-1)
    for q in range(1,n):
        lbits = lbits.at[q-1].set(jnp.median(jnp.log10(jnp.abs(jnp.diag(HFint,q)+jnp.diag(HFint,-q))/2.)))

    # 初始化 LIOM (在对角基底下是简单的 n_i)
    init_liom2 = jnp.zeros((n,n))
    init_liom2 = init_liom2.at[n//2,n//2].set(1.0)
    init_liom4 = jnp.zeros((n,n,n,n))

    # 3. 后向流：分段重算 (Recomputation)
    # ------------------------------------------------------------------
    num_segments = len(checkpoints) - 1
    print(f"        Backward integration: {num_segments} segments to process")
    
    jit_update = jit(update) # 确保 update 函数被 JIT 编译
    
    for i in range(num_segments, 0, -1):
        seg_progress = num_segments - i + 1
        print(f"        Segment {seg_progress}/{num_segments} (steps {checkpoints[i-1][0]}-{checkpoints[i][0]})", flush=True)
        # 当前段的起始和结束步数索引
        start_step_idx = checkpoints[i-1][0]
        end_step_idx = checkpoints[i][0]
        
        # 1. 重算阶段 (Recompute Forward)
        # 从 start_step 恢复状态
        temp_H2 = jnp.array(checkpoints[i-1][1])
        temp_Hint = jnp.array(checkpoints[i-1][2])
        
        # 临时存储这一小段的完整轨迹
        segment_len = end_step_idx - start_step_idx
        approx = _approx_enabled()
        if approx and (not norm):
            # JIT-compiled block runner (removes Python-per-step overhead)
            traj2, traj4 = _approx_run_block(temp_H2, temp_Hint, dl_list_final, start_step_idx, end_step_idx, method=method)
            # 【关键优化】立即转换为numpy并强制同步，释放JAX设备内存
            traj2.block_until_ready()
            traj4.block_until_ready()
            # 保持原始dtype，避免精度提升导致内存翻倍
            traj2_np = np.array(traj2, dtype=orig_dtype)
            traj4_np = np.array(traj4, dtype=orig_dtype)
            del traj2, traj4  # 释放JAX数组
        else:
            # 预分配内存 (注意：这里只存几十步，内存是可以接受的)
            seg_sol2 = [None] * segment_len
            seg_sol4 = [None] * segment_len
            
            # 在这一段内前向积分，填充轨迹
            # 注意：这里不需要极高的精度，只要和之前一致即可
            curr_step = start_step_idx
            while curr_step < end_step_idx:
                # 存下当前状态 (相对于 segment 的索引)
                local_idx = curr_step - start_step_idx
                seg_sol2[local_idx] = temp_H2
                seg_sol4[local_idx] = temp_Hint
                
                # 往前走一步
                t_span = np.linspace(dl_list_final[curr_step], dl_list_final[curr_step+1], num=2, endpoint=True)
                soln = ode(int_ode, [temp_H2, temp_Hint], t_span, rtol=_rtol, atol=_atol)
                temp_H2 = soln[0][-1]
                temp_Hint = soln[1][-1]
                curr_step += 1
            
        # 2. 回溯阶段 (Backward Integrate LIOM)
        # 现在我们有了 start_step 到 end_step 之间的密集轨迹 seg_sol
        # 倒着跑：从 end_step-1 跑到 start_step
        
        for local_idx in range(segment_len - 1, -1, -1):
            global_step = start_step_idx + local_idx
            
            # 获取当前时刻的 H (用于生成 eta)
            if approx and (not norm):
                h2_now = jnp.array(traj2_np[local_idx])
                hint_now = jnp.array(traj4_np[local_idx])
            else:
                h2_now = seg_sol2[local_idx]
                hint_now = seg_sol4[local_idx]
            
            # 逆向积分一步 LIOM
            # 注意 dl_list 的方向
            t_span_bck = np.linspace(dl_list_final[global_step+1], dl_list_final[global_step], num=2, endpoint=True)
            
            # 这里调用 update 或者 jit_update
            # update 函数通常接受 steps=[t_start, t_end]，内部计算 dl = t_end - t_start
            # 所以我们传入逆序的时间点即可
            
            init_liom2, init_liom4 = jit_update(init_liom2, init_liom4, h2_now, hint_now, t_span_bck)
            
        # 释放这一段的临时内存
        if approx and (not norm):
            del traj2_np, traj4_np
        else:
            del seg_sol2, seg_sol4
        
        # 【关键优化】处理完segment i后，checkpoints[i]不再需要，立即释放
        checkpoints[i] = None
        
        # Python 的 GC 可能不及时，手动释放一下保险
        # gc.collect() 
        if i % 5 == 0:
            print(f"Backward progress: Segment {num_segments-i+1}/{num_segments} done.")

    # 打包结果
    output = {
        "H0_diag": np.array(H0_diag), 
        "Hint": np.array(Hint2),
        "LIOM Interactions": lbits,
        "LIOM2": np.array(init_liom2),
        "LIOM4": np.array(init_liom4),
        # Compatibility with standard flow_static_int output schema.
        # (ckpt implementation currently only provides the final operators; we expose them under *_FWD too.)
        "LIOM2_FWD": np.array(init_liom2),
        "LIOM4_FWD": np.array(init_liom4),
        "Invariant": 0,  # 如果需要计算不变量，需额外处理
        # Keep output schema consistent with other flow routines so main scripts can always export.
        "dl_list": np.array(dl_list_final),
        "truncation_err": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    }
    
    return output


def flow_static_int_recursive(n, hamiltonian, dl_list, qmax, cutoff, method='tensordot', norm=False, Hflow=False, store_flow=False):
    forced = _force_steps()
    if forced is not None:
        dl_list = dl_list[: min(len(dl_list), forced + 1)]
        print(f"        [FORCE_STEPS] Running exactly {len(dl_list)-1} steps (ignoring cutoff)")
    """
    [算法优化] 递归检查点 (Recursive/Binomial Checkpointing)
    
    核心价值:
    - 将内存复杂度从线性 O(T) 降低到对数级 O(log T)。
    - 允许模拟极长流动时间 (l -> infinity)。
    
    Update: 
    - 增加了类似 standard/ckpt 模式的实时进度打印 (Print) 和内存日志 (Memlog)。
    """
    # Uses module-level jnp/jit imports
    
    H2, Hint = hamiltonian.H2_spinless, hamiltonian.H4_spinless
    
    # 预编译核心更新函数
    jit_update = jit(update) 
    
    # 初始状态
    H2_init = jnp.array(H2)
    Hint_init = jnp.array(Hint)

    # NOTE: adaptive-grid probing below uses the same `update()` step as the main forward pass
    # (not odeint) to avoid mismatched evolution/metrics.

    # IMPORTANT (JAX tracer safety):
    # When USE_JIT_FLOW=1, `int_ode` may lazily call `_get_jit_ode(n)` which mutates a Python cache
    # (and calls ex_helper/no_helper). If this happens during `odeint` tracing (adaptive grid build),
    # JAX raises UnexpectedTracerError. Pre-warm the cache here.
    if _USE_JIT_FLOW:
        _get_jit_ode(int(n))
    
    # 记录原始dtype以避免在numpy转换时提升精度
    orig_dtype = H2_init.dtype
    
    # Split strategy: uniform (index bisection) vs physics-informed (tau bisection)
    split_strategy = os.environ.get("PYFLOW_RECURSIVE_SPLIT", "uniform").strip().lower()

    # Tau metric mode (many options):
    # - delta_h2 (default): ||ΔH2||_F
    # - delta_h2_h4: ||ΔH2||_F + w4*||ΔH4||_F
    # - rhs: ||dH2/dl||_F*dl + w4*||dH4/dl||_F*dl   (closest to ||[η,H]||)
    # - offdiag: |Δ offdiag(H2)|  (off-diagonal Fro/max)
    # - inv: |Δ inv| where inv ~= Tr(H2^2) + w4*||H4||_F^2  (drift proxy)
    # - err: step-doubling error estimate (expensive)
    # - combo: weighted sum of several components
    tau_mode = os.environ.get("PYFLOW_TAU_MODE", "delta_h2").strip().lower()
    tau_include_hint = os.environ.get("PYFLOW_TAU_INCLUDE_HINT", "0") in ("1", "true", "True", "on", "ON")
    try:
        tau_w4 = float(os.environ.get("PYFLOW_TAU_W4", "1.0"))
    except Exception:
        tau_w4 = 1.0
    try:
        tau_w_off = float(os.environ.get("PYFLOW_TAU_W_OFF", "1.0"))
    except Exception:
        tau_w_off = 1.0
    try:
        tau_w_rhs = float(os.environ.get("PYFLOW_TAU_W_RHS", "1.0"))
    except Exception:
        tau_w_rhs = 1.0
    try:
        tau_w_inv = float(os.environ.get("PYFLOW_TAU_W_INV", "1.0"))
    except Exception:
        tau_w_inv = 1.0
    try:
        tau_w_err = float(os.environ.get("PYFLOW_TAU_W_ERR", "1.0"))
    except Exception:
        tau_w_err = 1.0
    # Step-doubling configuration
    tau_err_enabled = os.environ.get("PYFLOW_TAU_ERR_ENABLE", "0") in ("1", "true", "True", "on", "ON")

    # NaN/Inf debugging (disabled by default)
    _dbg_nan = os.environ.get("PYFLOW_DEBUG_NAN", "0") in ("1", "true", "True", "on", "ON")
    try:
        _dbg_every = int(os.environ.get("PYFLOW_DEBUG_EVERY", "50"))
    except Exception:
        _dbg_every = 50
    _dbg_include_hint = os.environ.get("PYFLOW_DEBUG_INCLUDE_HINT", "1") in ("1", "true", "True", "on", "ON")
    _dbg_last_good = None  # type: ignore

    # Keep ODE tolerances consistent with standard flow_static_int
    _rtol = float(os.environ.get('PYFLOW_ODE_RTOL', '1e-6'))
    _atol = float(os.environ.get('PYFLOW_ODE_ATOL', '1e-6'))

    # ─────────────────────────────────────────────────────────────────────────
    # Adaptive grid (benchmark-friendly controller: reduces total steps T)
    # ─────────────────────────────────────────────────────────────────────────
    adaptive_grid = os.environ.get("PYFLOW_ADAPTIVE_GRID", "0") in ("1", "true", "True", "on", "ON")
    adaptive_method = os.environ.get("PYFLOW_ADAPTIVE_METHOD", "controller").strip().lower()  # controller | (debug) step_doubling
    # Target "arc-length" per step; smaller => more steps, larger => fewer steps.
    try:
        adapt_target = float(os.environ.get("PYFLOW_ADAPTIVE_TARGET", "1e-3"))
    except Exception:
        adapt_target = 1e-3
    try:
        adapt_min_dl = float(os.environ.get("PYFLOW_ADAPTIVE_MIN_DL", "1e-8"))
    except Exception:
        adapt_min_dl = 1e-8
    try:
        adapt_max_dl = float(os.environ.get("PYFLOW_ADAPTIVE_MAX_DL", "1e2"))
    except Exception:
        adapt_max_dl = 1e2
    try:
        adapt_max_steps = int(os.environ.get("PYFLOW_ADAPTIVE_MAX_STEPS", str(len(dl_list) * 5)))
    except Exception:
        adapt_max_steps = int(len(dl_list) * 5)
    # Controller aggressiveness
    try:
        adapt_grow = float(os.environ.get("PYFLOW_ADAPTIVE_GROW", "2.0"))
    except Exception:
        adapt_grow = 2.0
    try:
        adapt_shrink = float(os.environ.get("PYFLOW_ADAPTIVE_SHRINK", "0.5"))
    except Exception:
        adapt_shrink = 0.5
    try:
        adapt_log_every = int(os.environ.get("PYFLOW_ADAPTIVE_LOG_EVERY", "0"))
    except Exception:
        adapt_log_every = 0
    # If dtau is extremely small (or exactly 0 due to numerical underflow), do NOT shrink dl.
    # Instead, grow dl to quickly traverse slow-evolution regions.
    try:
        adapt_min_dtau = float(os.environ.get("PYFLOW_ADAPTIVE_MIN_DTAU", "1e-14"))
    except Exception:
        adapt_min_dtau = 1e-14
    # Use H4 in dtau for controller? Off by default because H4 may start near-zero and can destabilize
    # relative-change metrics (huge dh4 / tiny ||H4||).
    adapt_include_h4 = os.environ.get("PYFLOW_ADAPTIVE_INCLUDE_H4", "0") in ("1", "true", "True", "on", "ON")
    try:
        adapt_w4 = float(os.environ.get("PYFLOW_ADAPTIVE_W4", "1.0"))
    except Exception:
        adapt_w4 = 1.0

    def _step_ode(l0: float, l1: float, h2, h4):
        """One integration step matching Phase-1 evolution (ode(int_ode, ...))."""
        steps = np.linspace(float(l0), float(l1), num=2, endpoint=True)
        # Avoid calling int_ode directly when JIT is enabled; use cached JIT RHS.
        if _USE_JIT_FLOW:
            soln = ode(_get_jit_ode(int(n)), [h2, h4], steps, rtol=_rtol, atol=_atol)
        else:
            def _rhs(y, l):
                return int_ode(y, l, method=method, norm=norm, Hflow=True)
            soln = ode(_rhs, [h2, h4], steps, rtol=_rtol, atol=_atol)
        return soln[0][-1], soln[1][-1]

    def _dtau_controller(prev_h2, prev_h4, next_h2, next_h4) -> float:
        # Use a *relative* change metric so the controller isn't dominated by the absolute scale of H.
        # This makes PYFLOW_ADAPTIVE_TARGET more portable across L / disorder realizations.
        eps = 1e-12
        dh2 = _fro_norm(next_h2 - prev_h2)
        h2n = max(_fro_norm(prev_h2), _fro_norm(next_h2), eps)
        val = dh2 / h2n
        if adapt_include_h4:
            dh4 = _fro_norm(next_h4 - prev_h4)
            h4n = max(_fro_norm(prev_h4), _fro_norm(next_h4), eps)
            val += adapt_w4 * (dh4 / h4n)
        return float(val)

    def _build_adaptive_dl_list(h2_init, h4_init, l_start: float, l_end: float, dl0: float):
        """Online controller: one ODE solve per accepted step; adjust dl based on ||ΔH||."""
        _t0 = time.perf_counter()
        l = float(l_start)
        dl = float(max(adapt_min_dl, min(adapt_max_dl, dl0)))
        h2 = h2_init
        h4 = h4_init
        l_points = [l]
        accepted = 0
        retries = 0
        hit_cap = False
        dt_min = float("inf")
        dt_max = 0.0
        dl_min = float("inf")
        dl_max = 0.0
        last_dt = float("nan")
        while (l < l_end) and (accepted < adapt_max_steps):
            l1 = min(l_end, l + dl)
            h2n, h4n = _step_ode(l, l1, h2, h4)
            if (not _isfinite_all(h2n)) or (not _isfinite_all(h4n)):
                # instability: shrink and retry
                dl = max(adapt_min_dl, dl * adapt_shrink)
                retries += 1
                if dl <= adapt_min_dl:
                    break
                continue
            dt = _dtau_controller(h2, h4, h2n, h4n)
            last_dt = float(dt)
            if np.isfinite(dt):
                dt_min = min(dt_min, float(dt))
                dt_max = max(dt_max, float(dt))
            dl_min = min(dl_min, float(l1 - l))
            dl_max = max(dl_max, float(l1 - l))
            if (not np.isfinite(dt)):
                # If dtau is unusable, keep dl conservative
                dl = max(adapt_min_dl, min(adapt_max_dl, dl * adapt_shrink))
            elif dt <= adapt_min_dtau:
                # Evolution is too slow / underflowed: accelerate.
                dl = max(adapt_min_dl, min(adapt_max_dl, dl * adapt_grow))
            else:
                # PI-like controller: scale dl to keep dt close to target, clamp growth
                ratio = adapt_target / dt
                ratio = max(adapt_shrink, min(adapt_grow, ratio))
                dl = max(adapt_min_dl, min(adapt_max_dl, dl * ratio))
            # accept
            l = l1
            h2, h4 = h2n, h4n
            l_points.append(l)
            accepted += 1
            if adapt_log_every and ((accepted % adapt_log_every) == 0):
                dt_str = f"{last_dt:.3e}" if np.isfinite(last_dt) else "nan"
                print(
                    f"        [Adaptive Grid] build: accepted={accepted}/{adapt_max_steps} "
                    f"l={l:.4g}/{l_end:.4g} dl={float(l_points[-1]-l_points[-2]):.3e} dtau={dt_str} "
                    f"retries={retries} elapsed={time.perf_counter()-_t0:.1f}s",
                    flush=True,
                )
        if (l < l_end) and (accepted >= adapt_max_steps):
            hit_cap = True
        return np.array(l_points, dtype=float), {
            "accepted": accepted,
            "retries": retries,
            "final_dl": dl,
            "hit_cap": hit_cap,
            "l_final": float(l),
            "dt_min": float(dt_min if np.isfinite(dt_min) else float("nan")),
            "dt_max": float(dt_max),
            "dl_min": float(dl_min if np.isfinite(dl_min) else float("nan")),
            "dl_max": float(dl_max),
        }

    def _isfinite_all(x) -> bool:
        try:
            return bool(jnp.all(jnp.isfinite(x)))
        except Exception:
            return False

    def _dbg_snapshot(tag: str, k_step: int, l_val: float, h2, hint):
        """Collect and maybe print a debug snapshot. Raises on first non-finite detection."""
        nonlocal _dbg_last_good
        if not _dbg_nan:
            return
        try:
            h2_max = float(jnp.max(jnp.abs(h2)))
        except Exception:
            h2_max = float("nan")
        hint_max = float("nan")
        if _dbg_include_hint:
            try:
                hint_max = float(jnp.max(jnp.abs(hint)))
            except Exception:
                hint_max = float("nan")
        h2_ok = _isfinite_all(h2)
        hint_ok = _isfinite_all(hint) if _dbg_include_hint else True
        snap = {
            "tag": tag,
            "step": int(k_step),
            "l": float(l_val),
            "max_abs_h2": h2_max,
            "max_abs_h4": hint_max,
            "finite_h2": bool(h2_ok),
            "finite_h4": bool(hint_ok),
        }

        # Periodic print
        if (k_step % _dbg_every) == 0:
            print(f"        [DBG] step={snap['step']} l={snap['l']:.4g} max|H2|={snap['max_abs_h2']:.3e} "
                  f"max|H4|={snap['max_abs_h4']:.3e} finite(H2,H4)=({snap['finite_h2']},{snap['finite_h4']})",
                  flush=True)

        # Detect first non-finite
        if (not h2_ok) or (not hint_ok) or (not np.isfinite(h2_max)) or (_dbg_include_hint and (not np.isfinite(hint_max))):
            print("        [DBG] ===== FIRST NON-FINITE DETECTED =====", flush=True)
            if _dbg_last_good is not None:
                print(f"        [DBG] last_good: {_dbg_last_good}", flush=True)
            print(f"        [DBG] current  : {snap}", flush=True)
            raise FloatingPointError("Non-finite values encountered in recursive forward integration")

        _dbg_last_good = snap

    def _fro_norm(x):
        # Frobenius norm for JAX arrays, returns Python float
        return float(jnp.sqrt(jnp.sum(jnp.abs(x) ** 2)))

    def _offdiag_metric(h2) -> float:
        # Off-diagonal magnitude proxy
        try:
            off = h2 - jnp.diag(jnp.diag(h2))
            # Use max abs (cheap, matches existing logging semantics)
            return float(jnp.max(jnp.abs(off)))
        except Exception:
            return float("nan")

    def _inv_proxy(h2, hint) -> float:
        # Cheap invariant/drift proxy (not the same as the full "Invariant" but catches runaway).
        try:
            e2 = float(jnp.trace(jnp.dot(h2, h2)))
        except Exception:
            e2 = float("nan")
        if tau_include_hint:
            try:
                # ||H4||_F^2
                h4n2 = float(jnp.sum(jnp.abs(hint) ** 2))
            except Exception:
                h4n2 = float("nan")
            return e2 + tau_w4 * h4n2
        return e2

    def _rhs_norm(l0: float, h2, hint, dl: float) -> float:
        # Closest to tau(l)=∫||[η,H]|| dl. Here we approximate with ||dH/dl|| * dl.
        try:
            dH2, dH4 = int_ode([h2, hint], l0, method=method, norm=norm, Hflow=True)
            val = _fro_norm(dH2) * abs(dl)
            if tau_include_hint:
                val += tau_w4 * _fro_norm(dH4) * abs(dl)
            return float(val)
        except Exception:
            return float("nan")

    def _step_doubling_err(l0: float, l1: float, h2, hint) -> float:
        # Expensive local error estimate (one step vs two half steps).
        # Only for diagnosis/comparison; enable via PYFLOW_TAU_ERR_ENABLE=1.
        if not tau_err_enabled:
            return 0.0
        try:
            mid = 0.5 * (l0 + l1)
            # One full step
            sol_full = ode(int_ode, [h2, hint], np.linspace(l0, l1, num=2, endpoint=True), rtol=_rtol, atol=_atol)
            h2_full, h4_full = sol_full[0][-1], sol_full[1][-1]
            # Two half steps
            sol_half1 = ode(int_ode, [h2, hint], np.linspace(l0, mid, num=2, endpoint=True), rtol=_rtol, atol=_atol)
            h2_mid, h4_mid = sol_half1[0][-1], sol_half1[1][-1]
            sol_half2 = ode(int_ode, [h2_mid, h4_mid], np.linspace(mid, l1, num=2, endpoint=True), rtol=_rtol, atol=_atol)
            h2_two, h4_two = sol_half2[0][-1], sol_half2[1][-1]
            err = _fro_norm(h2_two - h2_full)
            if tau_include_hint:
                err += tau_w4 * _fro_norm(h4_two - h4_full)
            return float(err)
        except Exception:
            return float("nan")

    def _dtau(prev_h2, prev_hint, next_h2, next_hint, l0: float, l1: float) -> float:
        # Compute increment for selected tau mode.
        dl = float(l1 - l0)
        if tau_mode in ("delta_h2", "dh2"):
            return _fro_norm(next_h2 - prev_h2)
        if tau_mode in ("delta_h2_h4", "dh2dh4"):
            val = _fro_norm(next_h2 - prev_h2)
            return val + (tau_w4 * _fro_norm(next_hint - prev_hint))
        if tau_mode in ("rhs", "flow", "speed"):
            return tau_w_rhs * _rhs_norm(l0, prev_h2, prev_hint, dl)
        if tau_mode in ("offdiag", "off", "offdiag_max"):
            return tau_w_off * abs(_offdiag_metric(next_h2) - _offdiag_metric(prev_h2))
        if tau_mode in ("inv", "invariant", "drift"):
            return tau_w_inv * abs(_inv_proxy(next_h2, next_hint) - _inv_proxy(prev_h2, prev_hint))
        if tau_mode in ("err", "error", "step_doubling"):
            return tau_w_err * _step_doubling_err(l0, l1, prev_h2, prev_hint)
        if tau_mode in ("combo", "mix"):
            val = 0.0
            # Always include ΔH2 as baseline
            val += _fro_norm(next_h2 - prev_h2)
            if tau_include_hint:
                val += tau_w4 * _fro_norm(next_hint - prev_hint)
            val += tau_w_off * abs(_offdiag_metric(next_h2) - _offdiag_metric(prev_h2))
            val += tau_w_inv * abs(_inv_proxy(next_h2, next_hint) - _inv_proxy(prev_h2, prev_hint))
            val += tau_w_rhs * _rhs_norm(l0, prev_h2, prev_hint, dl)
            if tau_err_enabled:
                val += tau_w_err * _step_doubling_err(l0, l1, prev_h2, prev_hint)
            return float(val)
        # Default fallback
        return _fro_norm(next_h2 - prev_h2)

    # 递归基准大小 (Base Case Block Size)
    # Keep consistent across methods for fair memory comparisons.
    BASE_CASE_STEPS = int(os.environ.get("PYFLOW_BASE_CASE_STEPS", "20"))
    
    if split_strategy in ("tau", "arc", "arclength", "physics", "physics_informed"):
        strat = "Tau/Arc-Length Bisection (Physics-Informed)"
    else:
        strat = "Pure Index Bisection"
    print(f"        [Recursive Flow] Strategy: {strat} (Memory: O(log T))")
    print(f"        [Recursive Flow] Total steps: {len(dl_list)}, Base case threshold: {BASE_CASE_STEPS}")
    if split_strategy in ("tau", "arc", "arclength", "physics", "physics_informed"):
        # Render tau mode
        extra = tau_mode
        if tau_mode in ("delta_h2_h4", "combo") and tau_include_hint:
            extra += f"(w4={tau_w4:g})"
        if tau_mode in ("rhs", "combo") and tau_w_rhs != 1.0:
            extra += f"(w_rhs={tau_w_rhs:g})"
        if tau_mode in ("offdiag", "combo") and tau_w_off != 1.0:
            extra += f"(w_off={tau_w_off:g})"
        if tau_mode in ("inv", "combo") and tau_w_inv != 1.0:
            extra += f"(w_inv={tau_w_inv:g})"
        if tau_mode in ("err", "combo") and tau_err_enabled:
            extra += f"(w_err={tau_w_err:g})"
        print(f"        [Recursive Flow] Split metric: tau mode={extra} include_H4={bool(tau_include_hint)} err_enable={bool(tau_err_enabled)}")

    # --- 辅助函数：前向积分 Hamiltonian ---
    # 增加 log_progress 参数，仅在 Phase 1 开启日志，递归内部重算时不打印
    # NOTE: Phase 1 runs before `stats` is defined; keep tau diagnostics separate here.
    tau_diag = {"tau_dt_nonzero": 0, "tau_dt_nonfinite": 0}
    def integrate_h_forward(h2, hint, t_start_idx, t_end_idx, log_progress=False, tau_out=None):
        curr_h2, curr_hint = h2, hint
        
        # 进度条控制变量
        total_steps = len(dl_list)
        last_progress = 0
        
        k = t_start_idx
        # Adaptive grid: we run Phase 1 on an already-built dl_list (controller), so approx can remain.
        # But for correctness benchmarking we keep approx disabled when adaptive_grid is on.
        approx = _approx_enabled() and (not adaptive_grid)
        _approx_cache: dict = {}
        off_diag = float("inf")
        while k < t_end_idx:
            prev_h2, prev_hint = curr_h2, curr_hint
            # 使用高精度步长
            l0 = float(dl_list[k])
            l1 = float(dl_list[k + 1])
            steps = np.linspace(l0, l1, num=2, endpoint=True)
            # 积分一步
            if approx and (not norm):
                curr_h2, curr_hint = _approx_step_h2_h4(
                    curr_h2,
                    curr_hint,
                    l0,
                    l1,
                    step_idx=k,
                    method=method,
                    cache=_approx_cache,
                    norm=norm,
                    Hflow=True,
                )
            else:
                soln = ode(int_ode, [curr_h2, curr_hint], steps, rtol=_rtol, atol=_atol)
                curr_h2, curr_hint = soln[0][-1], soln[1][-1]
            k += 1
            # Debug: detect NaN/Inf ASAP
            try:
                l_now = float(dl_list[min(k, len(dl_list) - 1)])
            except Exception:
                l_now = float("nan")
            _dbg_snapshot("recursive_fwd", k, l_now, curr_h2, curr_hint)
            if tau_out is not None:
                try:
                    dt = _dtau(prev_h2, prev_hint, curr_h2, curr_hint, l0, l1)
                    if not np.isfinite(dt):
                        # Keep tau monotone and usable; degrade by adding 0.
                        tau_diag["tau_dt_nonfinite"] += 1
                        dt = 0.0
                    elif dt != 0.0:
                        tau_diag["tau_dt_nonzero"] += 1
                    tau_out.append(tau_out[-1] + float(dt))
                except Exception:
                    # Always keep array length consistent
                    tau_out.append(tau_out[-1])
            # Phase 1 only: allow early stop based on cutoff
            if log_progress and (forced is None):
                off_diag = float(jnp.max(jnp.abs(curr_h2 - jnp.diag(jnp.diag(curr_h2)))))
                if off_diag <= cutoff:
                    print(f"        [Phase 1] Reached cutoff={cutoff:.2e} at step {k} | l={dl_list[min(k, total_steps-1)]:.4f} | off-diag={off_diag:.2e}", flush=True)
                    break
            
            # --- 日志逻辑 (仅 Phase 1) ---
            if log_progress:
                # 1. Memlog (每 10 步)
                if k % 10 == 0:
                    memlog("flow:step", step=k, mode="recursive_fwd")
                
                # 2. Print Progress (每 10% 或 50 步)
                progress = (k * 100) // total_steps
                if (progress >= last_progress + 10) or (k % 50 == 0) or (k == total_steps):
                    # 计算非对角元大小用于监控收敛
                    if off_diag == float("inf"):
                        off_diag = float(jnp.max(jnp.abs(curr_h2 - jnp.diag(jnp.diag(curr_h2)))))
                    print(f"        [Phase 1] Step {k}/{total_steps} ({progress}%) | l={dl_list[min(k, total_steps-1)]:.4f} | off-diag={off_diag:.2e}", flush=True)
                    last_progress = progress

        return curr_h2, curr_hint, k

    # 1. 第一阶段：快速前向跑到底 (Phase 1: Forward Pass)
    print("        [Recursive Flow] Phase 1: Forward pass to determine H(inf)...")
    _t_start = time.perf_counter()
    
    # Optional: build adaptive dl_list (reduces total steps)
    _adapt_allow_force = os.environ.get("PYFLOW_ADAPTIVE_ALLOW_FORCE", "0") in ("1", "true", "True", "on", "ON")
    if adaptive_grid and adaptive_method == "controller" and ((forced is None) or _adapt_allow_force):
        l_start = float(dl_list[0])
        l_end = float(dl_list[-1])
        # Start from mean step; logflow can have extremely tiny initial dl which would explode step counts.
        dl0 = float((l_end - l_start) / max(1, (len(dl_list) - 1)))
        print(
            f"        [Adaptive Grid] Enabled (controller): target_dtau={adapt_target:.1e} "
            f"min_dl={adapt_min_dl:.1e} max_dl={adapt_max_dl:.1e} max_steps={adapt_max_steps} "
            f"include_H4={bool(adapt_include_h4)} w4={adapt_w4:g}",
            flush=True,
        )
        dl_list_new, adp_stats = _build_adaptive_dl_list(H2_init, Hint_init, l_start, l_end, dl0)
        if adp_stats.get("hit_cap", False):
            print(
                f"        [Adaptive Grid] WARNING: hit max_steps={adapt_max_steps} before reaching lmax "
                f"(reached l={adp_stats.get('l_final', float('nan')):.4g}); "
                f"dtau[min,max]=({adp_stats.get('dt_min', float('nan')):.3e},{adp_stats.get('dt_max', float('nan')):.3e}) "
                f"dl[min,max]=({adp_stats.get('dl_min', float('nan')):.3e},{adp_stats.get('dl_max', float('nan')):.3e}); "
                f"falling back to original grid (steps={len(dl_list)-1}). Consider increasing "
                f"PYFLOW_ADAPTIVE_TARGET or PYFLOW_ADAPTIVE_MAX_STEPS.",
                flush=True,
            )
        else:
            print(f"        [Adaptive Grid] Steps: {len(dl_list_new)-1} (was {len(dl_list)-1}) retries={adp_stats['retries']}", flush=True)
            dl_list = dl_list_new

    # 开启日志 log_progress=True
    tau_cum = None
    if split_strategy in ("tau", "arc", "arclength", "physics", "physics_informed"):
        tau_cum = [0.0]
    H2_final, Hint_final, k_stop = integrate_h_forward(
        H2_init, Hint_init, 0, len(dl_list)-1, log_progress=True, tau_out=tau_cum
    )
    # If we stopped early (cutoff reached), truncate dl_list (and tau) so recursion doesn't waste work later.
    if (forced is None) and (k_stop < (len(dl_list) - 1)):
        dl_list = dl_list[: k_stop + 1]
        if tau_cum is not None:
            tau_cum = tau_cum[: k_stop + 1]
    
    print(f"        [Recursive Flow] Phase 1 done in {time.perf_counter()-_t_start:.2f}s")
    if tau_cum is not None and len(tau_cum) >= 2:
        print(f"        [Recursive Flow] Tau span: {tau_cum[-1]:.3e} over {len(tau_cum)-1} steps")
    
    # 提取 l-bits
    H0_diag = H2_final
    Hint2 = Hint_final
    HFint = jnp.zeros(n**2).reshape(n,n)
    for i in range(n):
        for j in range(n):
            HFint = HFint.at[i,j].set(Hint2[i,i,j,j] - Hint2[i,j,j,i])
    lbits = jnp.zeros(n-1)
    for q in range(1,n):
        lbits = lbits.at[q-1].set(jnp.median(jnp.log10(jnp.abs(jnp.diag(HFint,q)+jnp.diag(HFint,-q))/2.)))
        
    print("        [Recursive Flow] Phase 2: Recursive Backward integration for LIOMs...")
    
    # 初始化 LIOM
    liom2 = jnp.zeros((n,n))
    liom2 = liom2.at[n//2,n//2].set(1.0)
    liom4 = jnp.zeros((n,n,n,n))
    
    stats = {
        'recomputes': 0,
        'max_depth': 0,
        # How many base-case blocks were executed (each materializes a dense trajectory locally)
        'base_cases': 0,
        # Total number of per-step snapshots materialized across all base-cases (sum(block_len+1))
        'traj_snapshots_total': 0,
        # Max block length encountered (in steps)
        'max_block_len': 0,
        # Tau diagnostics
        'tau_dt_nonzero': 0,
        'tau_dt_nonfinite': 0,
        'tau_mid_fallback': 0,
        'tau_mid_diff': 0,
    }

    # --- 核心递归函数 (Recursive Solver) ---
    def recursive_solve(t_start_idx, t_end_idx, h2_start, hint_start, current_liom2, current_liom4, depth):
        stats['max_depth'] = max(stats['max_depth'], depth)
        
        # === Base Case: 线性/密集区间 ===
        if (t_end_idx - t_start_idx) <= BASE_CASE_STEPS:
            stats['base_cases'] += 1
            stats['max_block_len'] = max(int(stats['max_block_len']), int(t_end_idx - t_start_idx))
            # 日志：处理 Base Case
            # 注意：这是逆向过程，t_start_idx 代表我们当前“回溯”到的时间点
            if t_start_idx % 20 == 0: # 减少打印频率
                print(f"        [Phase 2] Backtracking LIOM: steps {t_end_idx} -> {t_start_idx}", flush=True)
            memlog("flow:step", step=t_start_idx, mode="recursive_bwd")

            # 1. Forward: 生成当前块的密集轨迹
            approx = _approx_enabled()
            curr_h2, curr_hint = h2_start, hint_start
            block_len = t_end_idx - t_start_idx
            
            # ─────────────────────────────────────────────────────────────────────
            # Adaptive + Mixed Precision (borrowed from Hybrid):
            # For adaptive mode we want to reduce memory by avoiding Python list-of-tuples
            # trajectories and instead storing a contiguous numpy buffer in FP16/FP32.
            #
            # Default: enabled when adaptive_grid is on. Disable via PYFLOW_ADAPTIVE_MP=0.
            # Override dtype via:
            #   PYFLOW_ADAPTIVE_BUFFER_DTYPE=float16|float32
            # Fallback to the hybrid knob if unset:
            #   PYFLOW_HYBRID_BUFFER_DTYPE=float16|float32
            # ─────────────────────────────────────────────────────────────────────
            adaptive_mp = adaptive_grid and (os.environ.get("PYFLOW_ADAPTIVE_MP", "1") in ("1", "true", "True", "on", "ON"))

            if adaptive_mp:
                dtype_env = os.environ.get("PYFLOW_ADAPTIVE_BUFFER_DTYPE", "").strip().lower()
                if not dtype_env:
                    dtype_env = os.environ.get("PYFLOW_HYBRID_BUFFER_DTYPE", "").strip().lower()

                if dtype_env in ("float16", "fp16"):
                    buffer_dtype = np.float16
                elif dtype_env in ("float32", "fp32"):
                    buffer_dtype = np.float32
                else:
                    # Heuristic: if magnitude is close to FP16 range, use FP32.
                    try:
                        h2_abs = float(jnp.max(jnp.abs(h2_start)))
                        hint_abs = float(jnp.max(jnp.abs(hint_start)))
                        max_abs = max(h2_abs, hint_abs)
                    except Exception:
                        max_abs = float("inf")
                    buffer_dtype = np.float32 if (not np.isfinite(max_abs) or max_abs > 1.0e4) else np.float16

                buffer_h2 = np.zeros((block_len + 1, n, n), dtype=buffer_dtype)
                buffer_hint = np.zeros((block_len + 1, n, n, n, n), dtype=buffer_dtype)

                # Store initial point
                curr_h2.block_until_ready()
                buffer_h2[0] = np.array(curr_h2, dtype=buffer_dtype)
                buffer_hint[0] = np.array(curr_hint, dtype=buffer_dtype)

                if approx and (not norm):
                    traj2, traj4 = _approx_run_block(curr_h2, curr_hint, dl_list, t_start_idx, t_end_idx, method=method)
                    traj2.block_until_ready()
                    traj4.block_until_ready()
                    buffer_h2[:, :, :] = np.array(traj2, dtype=buffer_dtype)
                    buffer_hint[:, :, :, :, :] = np.array(traj4, dtype=buffer_dtype)
                    del traj2, traj4
                else:
                    for k in range(t_start_idx, t_end_idx):
                        steps = np.linspace(dl_list[k], dl_list[k + 1], num=2, endpoint=True)
                        soln = ode(int_ode, [curr_h2, curr_hint], steps, rtol=_rtol, atol=_atol)
                        curr_h2, curr_hint = soln[0][-1], soln[1][-1]
                        local_idx = k - t_start_idx + 1
                        curr_h2.block_until_ready()
                        buffer_h2[local_idx] = np.array(curr_h2, dtype=buffer_dtype)
                        buffer_hint[local_idx] = np.array(curr_hint, dtype=buffer_dtype)

                stats['traj_snapshots_total'] += int(buffer_h2.shape[0])
            else:
                if approx and (not norm):
                    # 使用approx模式生成轨迹
                    traj2, traj4 = _approx_run_block(curr_h2, curr_hint, dl_list, t_start_idx, t_end_idx, method=method)
                    # 【关键优化】立即转换为numpy并强制同步，释放JAX设备内存
                    traj2.block_until_ready()
                    traj4.block_until_ready()
                    # 保持原始dtype，避免精度提升导致内存翻倍
                    traj2_np = np.array(traj2, dtype=orig_dtype)
                    traj4_np = np.array(traj4, dtype=orig_dtype)
                    del traj2, traj4  # 释放JAX数组
                    stats['traj_snapshots_total'] += int(traj2_np.shape[0])
                else:
                    trajectory = []
                    trajectory.append((curr_h2, curr_hint))
                    for k in range(t_start_idx, t_end_idx):
                        steps = np.linspace(dl_list[k], dl_list[k+1], num=2, endpoint=True)
                        soln = ode(int_ode, [curr_h2, curr_hint], steps, rtol=_rtol, atol=_atol)
                        curr_h2, curr_hint = soln[0][-1], soln[1][-1]
                        trajectory.append((curr_h2, curr_hint))
                    stats['traj_snapshots_total'] += int(len(trajectory))
            
            # 2. Backward: 利用密集轨迹逆向演化 LIOM
            l2, l4 = current_liom2, current_liom4
            
            for i in range(block_len - 1, -1, -1):
                global_step = t_start_idx + i
                if adaptive_mp:
                    # From mixed-precision contiguous buffers
                    h2_now = jnp.array(buffer_h2[i])
                    hint_now = jnp.array(buffer_hint[i])
                else:
                    if approx and (not norm):
                        # 从numpy数组恢复为JAX数组
                        h2_now = jnp.array(traj2_np[i])
                        hint_now = jnp.array(traj4_np[i])
                    else:
                        h2_now, hint_now = trajectory[i]
                t_span_bck = np.linspace(dl_list[global_step+1], dl_list[global_step], num=2, endpoint=True)
                l2, l4 = jit_update(l2, l4, h2_now, hint_now, t_span_bck)
            
            # 释放临时轨迹内存
            if adaptive_mp:
                del buffer_h2, buffer_hint
            else:
                if approx and (not norm):
                    del traj2_np, traj4_np
                else:
                    del trajectory
            
            return l2, l4

        # === Recursive Step: 二分 ===
        else:
            # Pick split point.
            # Default: uniform index bisection.
            # Tau mode: choose mid so that tau is split evenly between [start,end].
            uniform_mid = (t_start_idx + t_end_idx) // 2
            mid_idx = uniform_mid
            if tau_cum is not None:
                try:
                    tau_s = float(tau_cum[t_start_idx])
                    tau_e = float(tau_cum[t_end_idx])
                    if tau_e > tau_s:
                        tau_target = 0.5 * (tau_s + tau_e)
                        # searchsorted assumes non-decreasing tau
                        mid_idx = int(np.searchsorted(np.array(tau_cum), tau_target, side="left"))
                        if mid_idx != uniform_mid:
                            stats['tau_mid_diff'] += 1
                    else:
                        stats['tau_mid_fallback'] += 1
                except Exception:
                    stats['tau_mid_fallback'] += 1
                    mid_idx = uniform_mid

            # Guard against degenerate splits
            if mid_idx <= t_start_idx:
                mid_idx = t_start_idx + 1
            if mid_idx >= t_end_idx:
                mid_idx = t_end_idx - 1
            
            # 1. Forward Recompute (不打印日志)
            stats['recomputes'] += (mid_idx - t_start_idx)
            h2_mid, hint_mid, _ = integrate_h_forward(h2_start, hint_start, t_start_idx, mid_idx, log_progress=False, tau_out=None)
            
            # 2. Recurse Right (先处理后半段)
            l2_mid, l4_mid = recursive_solve(mid_idx, t_end_idx, h2_mid, hint_mid, current_liom2, current_liom4, depth+1)

            # === 【核心优化】 ===
            # 右半段处理完后，h2_mid 和 hint_mid 在左半段（Recurse Left）中完全不需要！
            # 显式删除它们，防止它们在处理左半段时一直占用内存。
            del h2_mid, hint_mid
            # ===================
            
            # 3. Recurse Left (再处理前半段)
            l2_final, l4_final = recursive_solve(t_start_idx, mid_idx, h2_start, hint_start, l2_mid, l4_mid, depth+1)
            
            return l2_final, l4_final

    # 启动递归
    _t_start_bck = time.perf_counter()
    liom2_final, liom4_final = recursive_solve(0, len(dl_list)-1, H2_init, Hint_init, liom2, liom4, 0)
    print(f"        [Recursive Flow] Phase 2 done in {time.perf_counter()-_t_start_bck:.2f}s")
    print(f"        [Recursive Flow] Stats: Max recursion depth: {stats['max_depth']}, Recomputed steps: {stats['recomputes']}")
    # Optional machine-readable stats line for benchmarking / comparisons
    if os.environ.get("PYFLOW_RECURSIVE_STATS", "0") in ("1", "true", "True", "on", "ON"):
        import json as _json  # local import
        payload = {
            "split_strategy": split_strategy,
            "tau_mode": tau_mode,
            "tau_include_hint": bool(tau_include_hint),
            "tau_w4": float(tau_w4),
            "tau_w_off": float(tau_w_off),
            "tau_w_rhs": float(tau_w_rhs),
            "tau_w_inv": float(tau_w_inv),
            "tau_err_enabled": bool(tau_err_enabled),
            "tau_w_err": float(tau_w_err),
            "steps_total": int(len(dl_list) - 1),
            "max_depth": int(stats["max_depth"]),
            "recomputed_steps": int(stats["recomputes"]),
            "base_cases": int(stats["base_cases"]),
            "traj_snapshots_total": int(stats["traj_snapshots_total"]),
            "max_block_len": int(stats["max_block_len"]),
            # Phase-1 tau diagnostics
            "tau_dt_nonzero": int(tau_diag.get("tau_dt_nonzero", 0)),
            "tau_dt_nonfinite": int(tau_diag.get("tau_dt_nonfinite", 0)),
            "tau_mid_fallback": int(stats.get("tau_mid_fallback", 0)),
            "tau_mid_diff": int(stats.get("tau_mid_diff", 0)),
            # log-memory "checkpoints on stack" proxy
            "stack_checkpoints_max": int(stats["max_depth"]) + 1,
        }
        print(f"[RECURSIVE_STATS] {_json.dumps(payload, sort_keys=True)}", flush=True)
    
    # 简单的 Invariant 计算
    delta = jnp.max(Hint)
    e1 = jnp.trace(jnp.dot(H2,H2))
    Hflat = HFint.reshape(n**2)
    inv = 2*jnp.sum(Hflat**2)
    e2 = jnp.trace(jnp.dot(H0_diag,H0_diag))
    inv2 = jnp.abs(e1 - e2 + ((2*delta)**2)*(n-1) - inv)/jnp.abs(e2+((2*delta)**2)*(n-1))

    # 打包结果
    output = {
        "H0_diag": np.array(H0_diag),
        "Hint": np.array(Hint2),
        "LIOM Interactions": lbits,
        "LIOM2": np.array(liom2_final),
        "LIOM4": np.array(liom4_final),
        "LIOM2_FWD": np.array(liom2_final), 
        "LIOM4_FWD": np.array(liom4_final),
        "Invariant": inv2,
        "dl_list": np.array(dl_list),
        "truncation_err": np.array([0.,0.,0.,0.])
    }
    
    return output

def flow_static_int_hybrid(n, hamiltonian, dl_list, qmax, cutoff, method='tensordot', norm=False, Hflow=False, store_flow=False):
    forced = _force_steps()
    if forced is not None:
        dl_list = dl_list[: min(len(dl_list), forced + 1)]
        print(f"        [FORCE_STEPS] Running exactly {len(dl_list)-1} steps (ignoring cutoff)")
    """
    [模式 D - 连续内存版] 混合精度递归 (Hybrid: Recursive + Quantized + Contiguous Buffer)
    
    关键改进:
    - Pre-allocation: 放弃 list.append，改为预分配连续的 (BlockSize, N, N) Numpy 数组。
    - Zero Fragmentation: 彻底消除小对象造成的内存碎片。
    - Forced Sync: 数据搬运时强制同步，防止 JAX 指令队列堆积。
    """
    # Uses module-level jnp/jit/np imports
    import gc
    
    H2, Hint = hamiltonian.H2_spinless, hamiltonian.H4_spinless
    jit_update = jit(update) 
    
    # Initial state dtype:
    # - Do NOT force float64: respect JAX_ENABLE_X64 / PYFLOW_FLOAT32 config from main.
    import jax
    dtype = jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32
    H2_init = jnp.array(H2, dtype=dtype)
    Hint_init = jnp.array(Hint, dtype=dtype)

    # IMPORTANT (JAX tracer safety): prewarm the JIT ODE cache so odeint tracing does not
    # mutate Python globals (which causes UnexpectedTracerError).
    if _USE_JIT_FLOW:
        _get_jit_ode(int(n))
    
    # Base Case Block Size (align with recursive for fair comparisons)
    BASE_CASE_STEPS = int(os.environ.get("PYFLOW_BASE_CASE_STEPS", "20"))

    # Keep ODE tolerances consistent with standard flow_static_int
    _rtol = float(os.environ.get('PYFLOW_ODE_RTOL', '1e-6'))
    _atol = float(os.environ.get('PYFLOW_ODE_ATOL', '1e-6'))

    # ─────────────────────────────────────────────────────────────────────────
    # Adaptive grid (same controller as recursive; reduces total steps T)
    # ─────────────────────────────────────────────────────────────────────────
    adaptive_grid = os.environ.get("PYFLOW_ADAPTIVE_GRID", "0") in ("1", "true", "True", "on", "ON")
    adaptive_method = os.environ.get("PYFLOW_ADAPTIVE_METHOD", "controller").strip().lower()
    _adapt_allow_force = os.environ.get("PYFLOW_ADAPTIVE_ALLOW_FORCE", "0") in ("1", "true", "True", "on", "ON")
    try:
        adapt_target = float(os.environ.get("PYFLOW_ADAPTIVE_TARGET", "1e-3"))
    except Exception:
        adapt_target = 1e-3
    try:
        adapt_min_dl = float(os.environ.get("PYFLOW_ADAPTIVE_MIN_DL", "1e-8"))
    except Exception:
        adapt_min_dl = 1e-8
    try:
        adapt_max_dl = float(os.environ.get("PYFLOW_ADAPTIVE_MAX_DL", "1e2"))
    except Exception:
        adapt_max_dl = 1e2
    try:
        adapt_max_steps = int(os.environ.get("PYFLOW_ADAPTIVE_MAX_STEPS", "20000"))
    except Exception:
        adapt_max_steps = 20000
    try:
        adapt_grow = float(os.environ.get("PYFLOW_ADAPTIVE_GROW", "2.0"))
    except Exception:
        adapt_grow = 2.0
    try:
        adapt_shrink = float(os.environ.get("PYFLOW_ADAPTIVE_SHRINK", "0.5"))
    except Exception:
        adapt_shrink = 0.5
    try:
        adapt_min_dtau = float(os.environ.get("PYFLOW_ADAPTIVE_MIN_DTAU", "1e-14"))
    except Exception:
        adapt_min_dtau = 1e-14
    try:
        adapt_log_every = int(os.environ.get("PYFLOW_ADAPTIVE_LOG_EVERY", "0"))
    except Exception:
        adapt_log_every = 0
    adapt_include_h4 = os.environ.get("PYFLOW_ADAPTIVE_INCLUDE_H4", "0") in ("1", "true", "True", "on", "ON")
    try:
        adapt_w4 = float(os.environ.get("PYFLOW_ADAPTIVE_W4", "1.0"))
    except Exception:
        adapt_w4 = 1.0

    def _isfinite_all(x) -> bool:
        try:
            return bool(jnp.all(jnp.isfinite(x)))
        except Exception:
            return False

    def _fro_norm(x):
        return float(jnp.sqrt(jnp.sum(jnp.abs(x) ** 2)))

    def _dtau_controller(prev_h2, prev_h4, next_h2, next_h4) -> float:
        eps = 1e-12
        dh2 = _fro_norm(next_h2 - prev_h2)
        h2n = max(_fro_norm(prev_h2), _fro_norm(next_h2), eps)
        val = dh2 / h2n
        if adapt_include_h4:
            dh4 = _fro_norm(next_h4 - prev_h4)
            h4n = max(_fro_norm(prev_h4), _fro_norm(next_h4), eps)
            val += adapt_w4 * (dh4 / h4n)
        return float(val)

    def _step_ode(l0: float, l1: float, h2, h4):
        steps = np.linspace(float(l0), float(l1), num=2, endpoint=True)
        if _USE_JIT_FLOW:
            soln = ode(_get_jit_ode(int(n)), [h2, h4], steps, rtol=_rtol, atol=_atol)
        else:
            def _rhs(y, l):
                return int_ode(y, l, method=method, norm=norm, Hflow=True)
            soln = ode(_rhs, [h2, h4], steps, rtol=_rtol, atol=_atol)
        return soln[0][-1], soln[1][-1]

    def _build_adaptive_dl_list(h2_init, h4_init, l_start: float, l_end: float, dl0: float):
        _t0 = time.perf_counter()
        l = float(l_start)
        dl = float(max(adapt_min_dl, min(adapt_max_dl, dl0)))
        h2 = h2_init
        h4 = h4_init
        l_points = [l]
        accepted = 0
        retries = 0
        hit_cap = False
        last_dt = float("nan")
        while (l < l_end) and (accepted < adapt_max_steps):
            l1 = min(l_end, l + dl)
            h2n, h4n = _step_ode(l, l1, h2, h4)
            if (not _isfinite_all(h2n)) or (not _isfinite_all(h4n)):
                dl = max(adapt_min_dl, dl * adapt_shrink)
                retries += 1
                if dl <= adapt_min_dl:
                    break
                continue
            dt = _dtau_controller(h2, h4, h2n, h4n)
            last_dt = float(dt)
            if (not np.isfinite(dt)):
                dl = max(adapt_min_dl, min(adapt_max_dl, dl * adapt_shrink))
            elif dt <= adapt_min_dtau:
                dl = max(adapt_min_dl, min(adapt_max_dl, dl * adapt_grow))
            else:
                ratio = adapt_target / dt
                ratio = max(adapt_shrink, min(adapt_grow, ratio))
                dl = max(adapt_min_dl, min(adapt_max_dl, dl * ratio))
            l = l1
            h2, h4 = h2n, h4n
            l_points.append(l)
            accepted += 1
            if adapt_log_every and ((accepted % adapt_log_every) == 0):
                dt_str = f"{last_dt:.3e}" if np.isfinite(last_dt) else "nan"
                print(
                    f"        [Adaptive Grid] build: accepted={accepted}/{adapt_max_steps} "
                    f"l={l:.4g}/{l_end:.4g} dl={float(l_points[-1]-l_points[-2]):.3e} dtau={dt_str} "
                    f"retries={retries} elapsed={time.perf_counter()-_t0:.1f}s",
                    flush=True,
                )
        if (l < l_end) and (accepted >= adapt_max_steps):
            hit_cap = True
        return np.array(l_points, dtype=float), {"accepted": accepted, "retries": retries, "hit_cap": hit_cap, "l_final": float(l)}
    
    print(f"        [Hybrid Flow] Strategy: Recursive + (FP16/FP32) Contiguous Buffer + Dynamic Exponent Scaling")
    print(f"        [Hybrid Flow] Steps: {len(dl_list)}, Base Block: {BASE_CASE_STEPS}")

    # Optional: build adaptive dl_list before Phase 1 forward pass
    if adaptive_grid and adaptive_method == "controller" and ((forced is None) or _adapt_allow_force):
        l_start = float(dl_list[0])
        l_end = float(dl_list[-1])
        dl0 = float((l_end - l_start) / max(1, (len(dl_list) - 1)))
        print(
            f"        [Adaptive Grid] Enabled (controller): target_dtau={adapt_target:.1e} "
            f"min_dl={adapt_min_dl:.1e} max_dl={adapt_max_dl:.1e} max_steps={adapt_max_steps} "
            f"include_H4={bool(adapt_include_h4)} w4={adapt_w4:g}",
            flush=True,
        )
        dl_list_new, adp_stats = _build_adaptive_dl_list(H2_init, Hint_init, l_start, l_end, dl0)
        if adp_stats.get("hit_cap", False):
            print(
                f"        [Adaptive Grid] WARNING: hit max_steps={adapt_max_steps} before reaching lmax "
                f"(reached l={adp_stats.get('l_final', float('nan')):.4g}); "
                f"falling back to original grid (steps={len(dl_list)-1}).",
                flush=True,
            )
        else:
            print(f"        [Adaptive Grid] Steps: {len(dl_list_new)-1} (was {len(dl_list)-1}) retries={adp_stats['retries']}", flush=True)
            dl_list = dl_list_new

    # --- 前向积分 ---
    def integrate_h_forward(h2, hint, t_start_idx, t_end_idx, log_progress=False):
        curr_h2, curr_hint = h2, hint
        total_steps = len(dl_list)
        last_progress = 0
        k = t_start_idx
        approx = _approx_enabled()
        _approx_cache: dict = {}
        off_diag = float("inf")
        while k < t_end_idx:
            steps = np.linspace(dl_list[k], dl_list[k+1], num=2, endpoint=True)
            if approx and (not norm):
                curr_h2, curr_hint = _approx_step_h2_h4(
                    curr_h2,
                    curr_hint,
                    float(dl_list[k]),
                    float(dl_list[k + 1]),
                    step_idx=k,
                    method=method,
                    cache=_approx_cache,
                    norm=norm,
                    Hflow=True,
                )
            else:
                soln = ode(int_ode, [curr_h2, curr_hint], steps, rtol=_rtol, atol=_atol)
                curr_h2, curr_hint = soln[0][-1], soln[1][-1]
            k += 1
            if log_progress:
                if k % 10 == 0: memlog("flow:step", step=k, mode="hybrid_fwd")
                progress = (k * 100) // total_steps
                off_diag = float(jnp.max(jnp.abs(curr_h2 - jnp.diag(jnp.diag(curr_h2)))))
                if (forced is None) and (off_diag <= cutoff):
                    print(f"        [Hybrid Phase 1] Reached cutoff={cutoff:.2e} at step {k} | off-diag={off_diag:.2e}", flush=True)
                    break
                if (progress >= last_progress + 10) or (k % 50 == 0) or (k == total_steps):
                    print(f"        [Hybrid Phase 1] Step {k}/{total_steps} ({progress}%) | off-diag={off_diag:.2e}", flush=True)
                    last_progress = progress
        return curr_h2, curr_hint

    # Phase 1: Forward
    print("        [Hybrid Flow] Phase 1: Forward pass (float64)...")
    H2_final, Hint_final = integrate_h_forward(H2_init, Hint_init, 0, len(dl_list)-1, log_progress=True)
    
    # l-bits extraction
    H0_diag = H2_final
    Hint2 = Hint_final
    HFint = jnp.zeros(n**2).reshape(n,n)
    for i in range(n):
        for j in range(n):
            HFint = HFint.at[i,j].set(Hint2[i,i,j,j] - Hint2[i,j,j,i])
    lbits = jnp.zeros(n-1)
    
    print("        [Hybrid Flow] Phase 2: Backward integration (Contiguous Buffer)...")
    liom2 = jnp.zeros((n,n)); liom2 = liom2.at[n//2,n//2].set(1.0)
    liom4 = jnp.zeros((n,n,n,n))
    
    stats = {'recomputes': 0, 'max_depth': 0, 'quantized_blocks': 0}

    # --- 递归求解器 ---
    def recursive_solve(t_start_idx, t_end_idx, h2_start, hint_start, current_liom2, current_liom4, depth):
        stats['max_depth'] = max(stats['max_depth'], depth)
        
        # === Base Case: 密集区间 ===
        if (t_end_idx - t_start_idx) <= BASE_CASE_STEPS:
            stats['quantized_blocks'] += 1
            if t_start_idx % 50 == 0: memlog("flow:step", step=t_start_idx, mode="hybrid_bwd")

            block_len = t_end_idx - t_start_idx
            
            # 【关键优化】预分配连续内存块 (Contiguous Memory Allocation)
            # 形状：(Steps, N, N) 和 (Steps, N, N, N, N)
            #
            # 原实现使用 float16 节省空间，但在部分参数下会发生溢出/饱和（float16 最大约 6.5e4），
            # 导致 LIOM/ITC 等物理量严重失真（例如 C(t) 爆到 > 1）。
            # 这里改为“自适应精度”：
            # - 默认允许 FP16（省内存）
            # - 若检测到当前区间数据幅值过大，则自动退化到 FP32（更稳）
            # 也可用环境变量强制选择：
            #   PYFLOW_HYBRID_BUFFER_DTYPE=float16|float32
            #
            # 进一步：动态指数缩放（Dynamic Exponent Scaling）
            # - 存储前提取 mu=max(|T|)，存 (T/mu) 到低精度 buffer，再存 mu 到高精度标量数组
            # - 读取时乘回 mu，避免 FP16 溢出/下溢
            # - 默认开启；可用 PYFLOW_HYBRID_EXP_SCALE=0 关闭
            #
            # 这里的 +1 是为了存储起点
            dtype_env = os.environ.get("PYFLOW_HYBRID_BUFFER_DTYPE", "").strip().lower()
            if dtype_env in ("float16", "fp16"):
                buffer_dtype = np.float16
            elif dtype_env in ("float32", "fp32"):
                buffer_dtype = np.float32
            else:
                # Heuristic: if magnitude is close to FP16 range, use FP32.
                # Use both H2 and Hint because Hint can become large during flow.
                try:
                    h2_abs = float(jnp.max(jnp.abs(h2_start)))
                    hint_abs = float(jnp.max(jnp.abs(hint_start)))
                    max_abs = max(h2_abs, hint_abs)
                except Exception:
                    max_abs = float("inf")
                # Conservative threshold below fp16 max to avoid saturation when casting trajectories.
                buffer_dtype = np.float32 if (not np.isfinite(max_abs) or max_abs > 1.0e4) else np.float16

            exp_scale = os.environ.get("PYFLOW_HYBRID_EXP_SCALE", "1") in ("1", "true", "True", "on", "ON")
            # Phase-2 pruning (sparsity & significance filtering) for CPU RAM:
            # Store only "meaningful" elements in COO-like form, per snapshot.
            # - Hard-threshold: keep |x| >= eps
            # - Active-space filter (optional): only keep entries touching "active sites"
            # Default off to preserve baseline behavior.
            prune = os.environ.get("PYFLOW_HYBRID_PRUNE", "0") in ("1", "true", "True", "on", "ON")
            try:
                prune_eps = float(os.environ.get("PYFLOW_PRUNE_EPS", "1e-7"))
            except Exception:
                prune_eps = 1e-7
            try:
                active_radius = int(os.environ.get("PYFLOW_PRUNE_ACTIVE_RADIUS", "-1"))
            except Exception:
                active_radius = -1

            # Precompute active sites (2D only) if requested.
            active_site = None
            if active_radius >= 0:
                try:
                    Lside = int(round(float(n) ** 0.5))
                    if Lside * Lside == int(n) and Lside > 0:
                        cx, cy = Lside // 2, Lside // 2
                        act = np.zeros((n,), dtype=np.bool_)
                        for s in range(n):
                            x, y = divmod(s, Lside)
                            if (abs(x - cx) + abs(y - cy)) <= active_radius:
                                act[s] = True
                        active_site = act
                except Exception:
                    active_site = None

            if not prune:
                buffer_h2 = np.zeros((block_len + 1, n, n), dtype=buffer_dtype)
                buffer_hint = np.zeros((block_len + 1, n, n, n, n), dtype=buffer_dtype)
            else:
                # Sparse COO storage: concatenate indices/values across snapshots with ptr offsets.
                h2_ptr = [0]
                h2_idx_chunks: list[np.ndarray] = []
                h2_val_chunks: list[np.ndarray] = []
                hint_ptr = [0]
                hint_idx_chunks: list[np.ndarray] = []
                hint_val_chunks: list[np.ndarray] = []
            # Store per-snapshot scaling factors in high precision (scalar only; negligible memory)
            scale_h2 = np.ones((block_len + 1,), dtype=np.float32)
            scale_hint = np.ones((block_len + 1,), dtype=np.float32)

            def _safe_mu_jax(x) -> float:
                """Return finite mu=max(|x|); fall back to 1.0."""
                try:
                    mu = float(jnp.max(jnp.abs(x)))
                except Exception:
                    mu = float("nan")
                if (not np.isfinite(mu)) or mu <= 0.0:
                    return 1.0
                return mu

            curr_h2, curr_hint = h2_start, hint_start
            
            # 1. Forward: 填充 buffer
            # 必须先把 JAX 数组 block 住，确保计算完成
            curr_h2.block_until_ready()
            
            def _append_sparse_snapshot(local_idx: int, h2_jax, h4_jax) -> None:
                mu2 = _safe_mu_jax(h2_jax) if exp_scale else 1.0
                mu4 = _safe_mu_jax(h4_jax) if exp_scale else 1.0
                scale_h2[local_idx] = np.float32(mu2)
                scale_hint[local_idx] = np.float32(mu4)

                # Store normalized (or raw) snapshots into low-precision numpy arrays
                h2_np = np.array((h2_jax / mu2) if exp_scale else h2_jax, dtype=buffer_dtype)
                h4_np = np.array((h4_jax / mu4) if exp_scale else h4_jax, dtype=buffer_dtype)

                # Threshold in the stored domain:
                # If exp_scale: stored values are divided by mu, so absolute threshold becomes eps/mu.
                th2 = (prune_eps / mu2) if exp_scale else prune_eps
                th4 = (prune_eps / mu4) if exp_scale else prune_eps
                if th2 <= 0.0:
                    th2 = 0.0
                if th4 <= 0.0:
                    th4 = 0.0

                # H2 sparse
                h2_flat = h2_np.reshape(-1)
                h2_mask = np.abs(h2_flat) >= th2
                h2_idx = np.flatnonzero(h2_mask).astype(np.int32, copy=False)
                if active_site is not None and h2_idx.size:
                    ii = (h2_idx // n)
                    jj = (h2_idx - ii * n)
                    keep = active_site[ii] | active_site[jj]
                    h2_idx = h2_idx[keep]
                h2_val = h2_flat[h2_idx].astype(buffer_dtype, copy=False) if h2_idx.size else np.zeros((0,), dtype=buffer_dtype)
                h2_idx_chunks.append(h2_idx)
                h2_val_chunks.append(h2_val)
                h2_ptr.append(h2_ptr[-1] + int(h2_idx.size))

                # H4 sparse
                h4_flat = h4_np.reshape(-1)
                h4_mask = np.abs(h4_flat) >= th4
                h4_idx = np.flatnonzero(h4_mask).astype(np.int32, copy=False)
                if active_site is not None and h4_idx.size:
                    n2 = n * n
                    n3 = n2 * n
                    i0 = (h4_idx // n3)
                    rem = (h4_idx - i0 * n3)
                    j0 = (rem // n2)
                    rem2 = (rem - j0 * n2)
                    k0 = (rem2 // n)
                    l0 = (rem2 - k0 * n)
                    keep = active_site[i0] | active_site[j0] | active_site[k0] | active_site[l0]
                    h4_idx = h4_idx[keep]
                h4_val = h4_flat[h4_idx].astype(buffer_dtype, copy=False) if h4_idx.size else np.zeros((0,), dtype=buffer_dtype)
                hint_idx_chunks.append(h4_idx)
                hint_val_chunks.append(h4_val)
                hint_ptr.append(hint_ptr[-1] + int(h4_idx.size))

                # Free temporary dense numpy arrays ASAP
                del h2_np, h4_np, h2_flat, h4_flat, h2_mask, h4_mask

            # 存入起点 (JAX -> Numpy)
            if not prune:
                if exp_scale:
                    mu2 = _safe_mu_jax(curr_h2)
                    mu4 = _safe_mu_jax(curr_hint)
                    scale_h2[0] = np.float32(mu2)
                    scale_hint[0] = np.float32(mu4)
                    buffer_h2[0] = np.array(curr_h2 / mu2, dtype=buffer_dtype)
                    buffer_hint[0] = np.array(curr_hint / mu4, dtype=buffer_dtype)
                else:
                    buffer_h2[0] = np.array(curr_h2, dtype=buffer_dtype)
                    buffer_hint[0] = np.array(curr_hint, dtype=buffer_dtype)
            else:
                _append_sparse_snapshot(0, curr_h2, curr_hint)

            approx = _approx_enabled()
            if approx and (not norm):
                traj2, traj4 = _approx_run_block(curr_h2, curr_hint, dl_list, t_start_idx, t_end_idx, method=method)
                if not prune:
                    if exp_scale:
                        # Scale per snapshot (small block_len), avoids overflow in low precision storage
                        for jj in range(block_len + 1):
                            mu2 = _safe_mu_jax(traj2[jj])
                            mu4 = _safe_mu_jax(traj4[jj])
                            scale_h2[jj] = np.float32(mu2)
                            scale_hint[jj] = np.float32(mu4)
                            buffer_h2[jj] = np.array(traj2[jj] / mu2, dtype=buffer_dtype)
                            buffer_hint[jj] = np.array(traj4[jj] / mu4, dtype=buffer_dtype)
                    else:
                        # Single host transfer per block
                        buffer_h2[:, :, :] = np.array(traj2, dtype=buffer_dtype)
                        buffer_hint[:, :, :, :, :] = np.array(traj4, dtype=buffer_dtype)
                else:
                    for jj in range(block_len + 1):
                        _append_sparse_snapshot(jj, traj2[jj], traj4[jj])
                # 立即释放JAX轨迹数据，只保留numpy buffer
                del traj2, traj4
            else:
                for k in range(t_start_idx, t_end_idx):
                    steps = np.linspace(dl_list[k], dl_list[k+1], num=2, endpoint=True)
                    soln = ode(int_ode, [curr_h2, curr_hint], steps, rtol=_rtol, atol=_atol)
                    curr_h2, curr_hint = soln[0][-1], soln[1][-1]
                    
                    # 存入 buffer
                    local_idx = k - t_start_idx + 1
                    curr_h2.block_until_ready() # 强制同步
                    if not prune:
                        if exp_scale:
                            mu2 = _safe_mu_jax(curr_h2)
                            mu4 = _safe_mu_jax(curr_hint)
                            scale_h2[local_idx] = np.float32(mu2)
                            scale_hint[local_idx] = np.float32(mu4)
                            buffer_h2[local_idx] = np.array(curr_h2 / mu2, dtype=buffer_dtype)
                            buffer_hint[local_idx] = np.array(curr_hint / mu4, dtype=buffer_dtype)
                        else:
                            buffer_h2[local_idx] = np.array(curr_h2, dtype=buffer_dtype)
                            buffer_hint[local_idx] = np.array(curr_hint, dtype=buffer_dtype)
                    else:
                        _append_sparse_snapshot(local_idx, curr_h2, curr_hint)
            
            # Finalize sparse buffers (concatenate) before backward sweep
            if prune:
                if h2_idx_chunks:
                    h2_idx = np.concatenate(h2_idx_chunks).astype(np.int32, copy=False)
                    h2_val = np.concatenate(h2_val_chunks).astype(buffer_dtype, copy=False)
                else:
                    h2_idx = np.zeros((0,), dtype=np.int32)
                    h2_val = np.zeros((0,), dtype=buffer_dtype)
                if hint_idx_chunks:
                    hint_idx = np.concatenate(hint_idx_chunks).astype(np.int32, copy=False)
                    hint_val = np.concatenate(hint_val_chunks).astype(buffer_dtype, copy=False)
                else:
                    hint_idx = np.zeros((0,), dtype=np.int32)
                    hint_val = np.zeros((0,), dtype=buffer_dtype)
                # Release chunk lists to reduce Python overhead
                del h2_idx_chunks, h2_val_chunks, hint_idx_chunks, hint_val_chunks

            # 2. Backward: 从 buffer 读取
            l2, l4 = current_liom2, current_liom4
            
            for i in range(block_len - 1, -1, -1):
                global_step = t_start_idx + i
                local_idx = i # 对应 buffer 中的位置
                
                # Restore from storage into JAX arrays (match configured dtype)
                if not prune:
                    if exp_scale:
                        h2_now = jnp.array(buffer_h2[local_idx], dtype=dtype) * jnp.array(scale_h2[local_idx], dtype=dtype)
                        hint_now = jnp.array(buffer_hint[local_idx], dtype=dtype) * jnp.array(scale_hint[local_idx], dtype=dtype)
                    else:
                        h2_now = jnp.array(buffer_h2[local_idx], dtype=dtype)
                        hint_now = jnp.array(buffer_hint[local_idx], dtype=dtype)
                else:
                    # Materialize dense tensors on-the-fly from sparse COO in CPU RAM.
                    # Note: this keeps only sparse data resident across the block.
                    start2, end2 = int(h2_ptr[local_idx]), int(h2_ptr[local_idx + 1])
                    idx2 = h2_idx[start2:end2]
                    val2 = h2_val[start2:end2]
                    # Dense H2
                    dense_dtype = np.float64 if dtype == jnp.float64 else np.float32
                    h2_dense = np.zeros((n * n,), dtype=dense_dtype)
                    if idx2.size:
                        h2_dense[idx2.astype(np.int64, copy=False)] = val2.astype(dense_dtype, copy=False)
                    h2_dense = h2_dense.reshape((n, n))
                    if exp_scale:
                        h2_dense *= float(scale_h2[local_idx])
                    h2_now = jnp.array(h2_dense, dtype=dtype)

                    start4, end4 = int(hint_ptr[local_idx]), int(hint_ptr[local_idx + 1])
                    idx4 = hint_idx[start4:end4]
                    val4 = hint_val[start4:end4]
                    # Dense H4/Hint
                    h4_dense = np.zeros((n**4,), dtype=dense_dtype)
                    if idx4.size:
                        h4_dense[idx4.astype(np.int64, copy=False)] = val4.astype(dense_dtype, copy=False)
                    h4_dense = h4_dense.reshape((n, n, n, n))
                    if exp_scale:
                        h4_dense *= float(scale_hint[local_idx])
                    hint_now = jnp.array(h4_dense, dtype=dtype)
                
                t_span_bck = np.linspace(dl_list[global_step+1], dl_list[global_step], num=2, endpoint=True)
                l2, l4 = jit_update(l2, l4, h2_now, hint_now, t_span_bck)
            
            # 3. 显式释放大块内存
            if not prune:
                del buffer_h2, buffer_hint, scale_h2, scale_hint
            else:
                del h2_idx, h2_val, hint_idx, hint_val, h2_ptr, hint_ptr, scale_h2, scale_hint
            gc.collect()
            
            return l2, l4

        # === Recursive Step: 二分 ===
        else:
            mid_idx = (t_start_idx + t_end_idx) // 2
            stats['recomputes'] += (mid_idx - t_start_idx)
            
            h2_mid, hint_mid = integrate_h_forward(h2_start, hint_start, t_start_idx, mid_idx, log_progress=False)
            
            l2_mid, l4_mid = recursive_solve(mid_idx, t_end_idx, h2_mid, hint_mid, current_liom2, current_liom4, depth+1)
            
            del h2_mid, hint_mid
            gc.collect() 
            
            l2_final, l4_final = recursive_solve(t_start_idx, mid_idx, h2_start, hint_start, l2_mid, l4_mid, depth+1)
            
            return l2_final, l4_final

    # 启动
    liom2_final, liom4_final = recursive_solve(0, len(dl_list)-1, H2_init, Hint_init, liom2, liom4, 0)
    
    # 输出部分不变
    output = {
        "H0_diag": np.array(H0_diag), "Hint": np.array(Hint2),
        "LIOM2": np.array(liom2_final), "LIOM4": np.array(liom4_final),
        "LIOM2_FWD": np.array(liom2_final), "LIOM4_FWD": np.array(liom4_final),
        "dl_list": np.array(dl_list),
        "truncation_err": np.array([0.,0.,0.,0.])
    }
    return output


def flow_static_int_hybrid_pruned(
    n,
    hamiltonian,
    dl_list,
    qmax,
    cutoff,
    method='tensordot',
    norm=False,
    Hflow=False,
    store_flow=False,
):
    """
    Wrapper for Hybrid with Phase-2 sparsity/pruning enabled.

    Enable:
      - PYFLOW_HYBRID_PRUNE=1
    Optional knobs:
      - PYFLOW_PRUNE_EPS (default 1e-7)
      - PYFLOW_PRUNE_ACTIVE_RADIUS (default -1: disabled)
      - PYFLOW_HYBRID_EXP_SCALE=1 (default on)
    """
    os.environ["PYFLOW_HYBRID_PRUNE"] = "1"
    return flow_static_int_hybrid(n, hamiltonian, dl_list, qmax, cutoff, method=method, norm=norm, Hflow=Hflow, store_flow=store_flow)

