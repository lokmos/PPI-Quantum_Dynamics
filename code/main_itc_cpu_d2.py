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

If you do use any of this code, please cite https://arxiv.org/abs/2110.02906.

---------------------------------------------

This file contains the main code used to set up and run the flow equation method,
and save the output as an HDF5 file containing various different datasets.

"""

import os, sys
from psutil import cpu_count
# Set up threading options for parallel solver
os.environ['OMP_NUM_THREADS']= str(int(cpu_count(logical=False))) # Set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']= str(int(cpu_count(logical=False))) # Set number of MKL threads to run in parallel
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"                         # Necessary on some versions of OS X
os.environ['KMP_WARNINGS'] = 'off'                                # Silence non-critical warning

# JAX options - must be set BEFORE importing the JAX library
#os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[5])             # Set which device to use ('' is CPU)
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
#os.environ['JAX_ENABLE_X64'] = 'true'                             # Enable 64-bit floats in JAX
# from jax.config import config
# config.update('jax_disable_jit', True)                          # Disable JIT compilation in JAX for debugging
# config.update("jax_enable_x64", True)                           # Alternate way to enable float64 in JAX

import jax.numpy as jnp 
import numpy as np
#from scipy.special import jv as jv
from datetime import datetime
import h5py,gc
import core.diag as diag
import models.models as models
import core.utility as utility
from core.memlog import memlog, close as memlog_close
# ED (QuSpin) is optional. For flow-only runs we skip ED entirely.
try:
    from ED.ed import ED  # type: ignore
except Exception:
    ED = None

#import matplotlib.pyplot as plt
# Part to change plotting system
#import matplotlib as mpl
#mpl.rcParams['figure.figsize'] = (12,6)
#plt.rc('font',family='serif')
#plt.rcParams.update({'font.size': 24})
#plt.rc('text', usetex=True)
#mpl.rcParams['mathtext.fontset'] = 'cm'
#mpl.rcParams['mathtext.rm'] = 'serif'
#------------------------------------------------------------------------------  
# Parameters
L = int(sys.argv[1])            # Linear system size
dim = 2                         # Spatial dimension
n = L**dim                      # Total number of sites
species = 'spinless fermion'    # Type of particle
dsymm = 'spin'                  # Type of disorder (spinful fermions only)
Ulist = [0.1]
# List of interaction strengths
J = 0.5                         # Nearest-neighbour hopping amplitude
cutoff = J*10**(-4)             # Cutoff for the off-diagonal elements to be considered zero
# By default, use the CLI disorder strength (matches main_itc_cpu.py behavior).
# If you want to sweep d=1..6, set PYFLOW_SWEEP_D=1.
if os.environ.get("PYFLOW_SWEEP_D", "0") in ("1", "true", "True"):
    dis = [1.0 * i for i in range(1, 7)]
else:
    dis = [float(sys.argv[4])]
order = 4                       # Order of terms to keep in running Hamiltonian (PROTOTYPE FEATURE)
# List of disorder strengths
lmax =  1000                     # Flow time max
qmax =  1000                     # Max number of flow time steps
#if dim == 2:
#    lmax *= 2
#    qmax *= 1.5
#    qmax = int(qmax)
                                # For forward-only transforms, this makes essentially no difference as
                                # additional steps are inserted by the solver as required
#reps = 128                     # Number of disorder realisations [DEPRECATED: CONTROLLED BY ENV VARIABLE]
norm = False                    # Normal-ordering, can be true or false
no_state = 'SDW'                # State to use for normal-ordering, can be CDW or SDW
                                # For vacuum normal-ordering, just set norm=False
ladder = False                 # TEST FEATURE: compute LIOMs using creation/annihilation operators
ITC = False                     # Infinite temp correlation function (TEST PARAMETER)
Hflow = True                    # Whether to store the flowing Hamiltonian (true) or generator (false)
                                # Storing H(l) allows SciPy ODE integration to add extra flow time steps
                                # Storing eta(l) reduces number of tensor contractions, at cost of accuracy
                                # NB: if the flow step dl is too large, this can lead to Vint diverging!
# precision = np.float64        # Precision with which to store running Hamiltonian/generator
                                # Default throughout is double precision (np.float64)
                                # Using np.float32/np.float16 will half the memory cost, at loss of precision
                                # Only affects the backwards transform, not the forward transform
method = str(sys.argv[3])       # Method for computing tensor contractions
                                # Options are 'einsum', 'tensordot','jit' or 'vec'
                                # In general 'tensordot' is fastest for small systems, 'jit' for large systems
                                # (Note that 'jit' requires compilation on the first run, increasing run time.)
print('Norm = %s' %norm)
intr = True                     # Turn on/off interactions
dyn = False                     # Run the dynamics
imbalance = True                # Sets whether to compute global imbalance or single-site dynamics
LIOM = 'bck'                    # Compute LIOMs with forward ('fwd') or backward ('bck') flow
                                # Forward uses less memory by a factor of qmax, and transforms a local operator
                                # in the initial basis into the diagonal basis; backward does the reverse
dyn_MF = True                   # Mean-field decoupling for dynamics (used only if dyn=True)
logflow = True                  # Use logarithmically spaced steps in flow time
store_flow = False              # Store the full flow of the Hamiltonian and LIOMs
dis_type = str(sys.argv[2])     # Options: 'random', 'QPgolden', 'QPsilver', 'QPbronze', 'QPrandom', 'linear', 'curved', 'prime'
                                # Also contains 'test' and 'QPtest', potentials that do not change from run to run
xlist = [1.]
# For 'dis_type = curved', controls the gradient of the curvature
if intr == False:               # Zero the interactions if set to False (for ED comparison and filename)
    delta = 0
if dis_type != 'curved':
    xlist = [0.0]
if (species == 'spinless fermion' and n > 12) or (species == 'spinful fermion' and n > 6) or qmax > 2000:
    print('SETTING store_flow = False DUE TO TOO MANY VARIABLES AND/OR FLOW TIME STEPS')
    store_flow = False

# Define list of timesteps for non-equilibrium dynamics
# Only used if 'dyn = True'
# tlist = [0.1*i for i in range(151)]
tlist = np.logspace(-2,5,151,base=10,endpoint=True)

# Make directory to store data
nvar = utility.namevar(dis_type,dim,dsymm,no_state,dyn,norm,n,LIOM,species,order)

if Hflow == False:
    print('*** Warning: Setting Hflow=False requires small flow time steps in order for backwards transform to be accurate. ***')
if intr == False and norm == True:
    print('Normal ordering is only for interacting systems.')
    norm = False
if norm == True and n%2 != 0:
    print('Normal ordering is only for even system sizes')
    norm = False
if ITC == True:
    store_flow = False

checkpoint_mode = os.environ.get("USE_CKPT", "0") == "1"
# if species == 'spinful fermion' and norm == True:
#     print('Normal ordering not implemented for spinful fermions.')
#     norm = False
#==============================================================================
# Run program
#==============================================================================

if __name__ == '__main__': 

    startTime = datetime.now()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Print header with configuration
    # ═══════════════════════════════════════════════════════════════════════════
    print("═" * 70)
    print("  Flow Equations for Many-Body Quantum Systems")
    print("═" * 70)
    print(f"  Start time:     {startTime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  System:         L={L}, dim={dim}, n={n} sites")
    print(f"  Species:        {species}")
    print(f"  Disorder:       {dis_type}, d={dis}")
    print(f"  Interactions:   U={Ulist}, J={J}")
    print(f"  Method:         {method}")
    print(f"  Flow params:    lmax={lmax}, qmax={qmax}, cutoff={cutoff}")
    print(f"  Checkpoint:     {'ON' if checkpoint_mode else 'OFF'}")
    print("═" * 70)
    
    memlog("main:start")

    overwrite = os.environ.get("PYFLOW_OVERWRITE", "0") in ("1", "true", "True")
    
    total_runs = len([int(sys.argv[5])]) * len(xlist) * len(dis) * len(Ulist)
    current_run = 0

    for p in [int(sys.argv[5])]:
        for x in xlist:
            for d in dis:
                for delta in Ulist:
                    current_run += 1
                    
                    print(f"\n{'─' * 70}")
                    print(f"  Run {current_run}/{total_runs}: d={d:.2f}, U={delta:.2f}, x={x:.2f}, p={p}")
                    print(f"{'─' * 70}")

                    # Create dictionary of parameters to pass to functions
                    params = {"n":n,"delta":delta,"J":J,"cutoff":cutoff,"dis":dis,"dsymm":dsymm,"NO_state":no_state,"lmax":lmax,"qmax":qmax,"norm":norm,"Hflow":Hflow,"method":method, "intr":intr,"dyn":dyn,"imbalance":imbalance,"species":species,
                                    "LIOM":LIOM, "dyn_MF":dyn_MF,"logflow":logflow,"dis_type":dis_type,"x":x,"tlist":tlist,"store_flow":store_flow,"ITC":ITC,
                                    "ladder":ladder,"order":order,"dim":dim,"checkpoint_mode":checkpoint_mode}

                    out_h5 = '%s/tflow-d%.2f-O%s-x%.2f-Jz%.2f-p%s.h5' % (nvar, d, order, x, delta, p)
                    if overwrite or (not os.path.exists(out_h5)):
                        #-----------------------------------------------------------------
                        # Initialise Hamiltonian
                        print("  [1/4] Initializing Hamiltonian...", end=" ", flush=True)
                        ham_init_start = datetime.now()
                        ham = models.hamiltonian(species,dis_type,intr=intr)
                        if species == 'spinless fermion':
                            ham.build(n,dim,d,J,x,delta=delta)
                        elif species == 'spinful fermion':
                            ham.build(n,dim,d,J,x,delta_onsite=delta,delta_up=0.,delta_down=0.,dsymm=dsymm)
                        print(f"done ({(datetime.now()-ham_init_start).total_seconds():.2f}s)")

                        # Initialise the number operator on the central lattice site
                        num = jnp.zeros((n,n))
                        num = num.at[n//2,n//2].set(1.0)
                        
                        # Initialise higher-order parts of number operator (empty)
                        num_int=jnp.zeros((n,n,n,n),dtype=jnp.float64)
                        
                        #-----------------------------------------------------------------

                        # Diag non-interacting system w/NumPy
                        print("  [2/4] Computing reference eigenvalues...", end=" ", flush=True)
                        diag_start = datetime.now()
                        eigvals = jnp.sort(jnp.linalg.eigvalsh(ham.H2_spinless))
                        print(f"done ({(datetime.now()-diag_start).total_seconds():.2f}s)")

                        #-----------------------------------------------------------------

                        # Diagonalise with flow equations
                        # If enabled, write memlog into repo-level test/ by default.
                        if os.environ.get("PYFLOW_MEMLOG", "0") in ("1", "true", "True"):
                            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                            test_dir = os.path.join(repo_root, "test")
                            os.makedirs(test_dir, exist_ok=True)
                            
                            is_ckpt = bool(params.get("checkpoint_mode", False))
                            mode_str = "ckpt" if is_ckpt else "original"
                            
                            log_filename = (
                                f"memlog-dim{dim}-L{L}-d{float(d):.2f}-O{order}-x{float(x):.2f}-Jz{float(delta):.2f}-p{p}-{mode_str}.jsonl"
                            )
                            
                            memlog_close()
                            os.environ["PYFLOW_MEMLOG_FILE"] = os.path.join(test_dir, log_filename)

                        print("  [3/4] Running flow equations...", flush=True)
                        memlog("main:before_flow", step=p, d=float(d), delta=float(delta), L=int(L), n=int(n), dim=int(dim))
                        flow_startTime = datetime.now()
                        flow = diag.CUT(params,ham,num,num_int)
                        flow_endTime = datetime.now()-flow_startTime
                        memlog("main:after_flow", step=p)
                        print(f"        Flow completed in {flow_endTime.total_seconds():.2f}s")

                        if species == 'spinless fermion':
                            ncut = 16
                        elif species == 'spinful fermion':
                            ncut = 6

                        # Diagonalise with ED (optional; requires QuSpin)
                        if ED is not None and n <= ncut:
                            print("  [3.5/4] Running exact diagonalization (QuSpin)...", end=" ", flush=True)
                            ed_startTime = datetime.now()
                            if dyn == True:
                                ed = ED(n, ham, tlist, dyn, imbalance, dim)
                                ed_endTime = datetime.now() - ed_startTime
                                ed_dyn = ed[1]
                            else:
                                ed = ED(n, ham, tlist, dyn, imbalance, dim)
                                ed_endTime = datetime.now() - ed_startTime
                                ed_itc = ed[1]
                            print(f"done ({ed_endTime.total_seconds():.2f}s)")
                        else:
                            ed = np.zeros(n)
                            if ED is None:
                                print("        (QuSpin not available, skipping ED)")
                            else:
                                print(f"        (System too large for ED: n={n} > {ncut})")

                        if (intr == False or n <= ncut) and ED is not None:
                            if species == 'spinless fermion':
                                flevels = utility.flow_levels(n,flow,intr,order)
                            elif species == 'spinful fermion':
                                flevels = utility.flow_levels_spin(n,flow,intr,order)
                            flevels = flevels-np.median(flevels)
                            ed = ed[0] - np.median(ed[0])
                        
                        else:
                            flevels=np.zeros(n)
                            ed=np.zeros(n)

                        if (intr == False or n <= ncut) and ED is not None:
                            lsr = utility.level_stat(flevels)
                            lsr2 = utility.level_stat(ed)

                            errlist = np.zeros(len(ed))
                            for i in range(len(ed)):
                                if np.round(ed[i],10)!=0.:
                                    errlist[i] = np.abs((ed[i]-flevels[i])/ed[i])

                            print('***** ERROR *****: ', np.median(errlist))  

                        #if dyn == True:
                        #    plt.plot(tlist,ed_dyn,label=r'ED')
                        #    if imbalance == True:
                        #        plt.plot(tlist,flow["Imbalance"],'o')
                        #        plt.ylabel(r'$\mathcal{I}(t)$')
                        #    else:
                        #        plt.plot(tlist,flow["Density Dynamics"],'o',label='Flow')
                        #        plt.ylabel(r'$\langle n_i(t) \rangle$')
                        #    plt.xlabel(r'$t$')
                        #    plt.legend()
                        #    plt.show()
                        #    plt.close()

                        #==============================================================
                        # Export data
                        print("  [4/4] Saving results to HDF5...", end=" ", flush=True)
                        save_start = datetime.now()
                        memlog("main:before_h5_write", step=p)
                        with h5py.File(out_h5, 'w') as hf:
                            hf.create_dataset('params',data=str(params))
                            hf.create_dataset('fe_runtime',data=str(flow_endTime))
                            hf.create_dataset('trunc_err',data=flow["truncation_err"])
                            hf.create_dataset('dl_list',data=flow["dl_list"])
                            hf.create_dataset('H2_diag',data=flow["H0_diag"])

                            if species == 'spinless fermion':
                                hf.create_dataset('H2_initial',data=ham.H2_spinless)
                            elif species == 'spinful fermion':
                                hf.create_dataset('H2_up',data=ham.H2_spinup)
                                hf.create_dataset('H2_dn',data=ham.H2_spindown)

                            if ED is not None and n <= ncut:
                                hf.create_dataset('ed_runtime',data=str(ed_endTime))
                                hf.create_dataset('flevels', data = flevels,compression='gzip', compression_opts=9)
                                hf.create_dataset('ed', data = ed, compression='gzip', compression_opts=9)
                                hf.create_dataset('lsr', data = [lsr,lsr2])
                                hf.create_dataset('err',data = errlist)
                                if store_flow == True:
                                    hf.create_dataset('flow2',data=flow["flow2"])
                                    hf.create_dataset('flow4',data=flow["flow4"])
                                    hf.create_dataset('dl_list',data=flow["dl_list"])
                            if intr == True:
                                    liom_int = flow.get("lbits", flow.get("LIOM Interactions", None))
                                    if liom_int is not None:
                                        hf.create_dataset('LIOM Interactions', data=liom_int)
                                    if ITC == False and ladder == False:
                                        hf.create_dataset('liom2', data = flow["LIOM2"], compression='gzip', compression_opts=9)
                                        hf.create_dataset('liom4', data = flow["LIOM4"], compression='gzip', compression_opts=9)
                                    hf.create_dataset('Hint', data = flow["Hint"], compression='gzip', compression_opts=9)
                                    if ladder == False:
                                        liom2_fwd = flow.get("LIOM2_FWD", flow.get("LIOM2", None))
                                        liom4_fwd = flow.get("LIOM4_FWD", flow.get("LIOM4", None))
                                        if liom2_fwd is not None:
                                            hf.create_dataset('liom2_fwd', data=liom2_fwd, compression='gzip', compression_opts=9)
                                        if liom4_fwd is not None:
                                            hf.create_dataset('liom4_fwd', data=liom4_fwd, compression='gzip', compression_opts=9)
                                    elif ladder == True:
                                        hf.create_dataset('liom1_fwd', data = flow["LIOM1_FWD"], compression='gzip', compression_opts=9)
                                        hf.create_dataset('liom3_fwd', data = flow["LIOM3_FWD"], compression='gzip', compression_opts=9)
                                        hf.create_dataset('liom2_fwd', data = flow["l2"], compression='gzip', compression_opts=9)
                                        #hf.create_dataset('liom4_fwd', data = flow["l4"], compression='gzip', compression_opts=9)
                                        # hf.create_dataset('nt_list',data = flow["nt_list"])
                                        hf.create_dataset('itc',data=flow["itc"])
                                        hf.create_dataset('itc_nonint',data=flow["itc_nonint"])
                                        if ED is not None and n <= ncut:
                                            hf.create_dataset('ed_itc',data=ed_itc)
                                    # hf.create_dataset('inv',data=flow["Invariant"])
                            if dyn == True:
                                hf.create_dataset('tlist', data = tlist)
                                if imbalance == True:
                                    hf.create_dataset('imbalance',data=flow["Imbalance"])
                                else:
                                    hf.create_dataset('flow_dyn', data = flow["Density Dynamics"])
                                if n <= ncut:
                                    hf.create_dataset('ed_dyn', data = ed_dyn)
                        memlog("main:after_h5_write", step=p)
                        print(f"done ({(datetime.now()-save_start).total_seconds():.2f}s)")
                        print(f"        Output: {out_h5}")
                    else:
                        print(f"        ⏭  Skipping (output exists): {out_h5}")
                                        
                gc.collect()
                memlog("main:after_gc", step=p)
                memlog_close()
                
                run_time = datetime.now() - startTime
                print(f"\n  ✓ Run completed in {run_time.total_seconds():.2f}s")

    # Final summary
    total_time = datetime.now() - startTime
    print("\n" + "═" * 70)
    print("  COMPLETED")
    print("═" * 70)
    print(f"  Total runs:     {total_runs}")
    print(f"  Total time:     {total_time}")
    print(f"  End time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 70)
