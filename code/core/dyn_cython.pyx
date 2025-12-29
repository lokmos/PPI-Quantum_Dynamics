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

This file contains Cython code for computing an autocorrelation function from time-evolved operators.

"""
cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
DTYPE = np.float64
       
#------------------------------------------------------------------------------

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.nonecheck(False)
@cython.cdivision(True)
def cytrace(int n,double t,double[:,:] H, double[:,:,:,:] Hint, double complex[:,:] n2b, double complex[:,:,:,:] n4b,double complex[:,:,:,:,:,:] n6b, double[:,:] statelist, double complex[:] corr_list):

    cdef int i,j,k,q,ns
    cdef double complex corr = 0.
    cdef double [:] state
    cdef double complex[:,:] n2a
    cdef double complex[:,:,:,:] n4a
    cdef double complex[:,:,:,:,:,:] n6a

    n2a,n4a,n6a = cy_tstep(n,t,H,Hint,n2b,n4b,n6b)

    for ns in range(len(statelist)):
        state = statelist[ns]
        corr = 0.
        # Product of quadratic terms
        for i in prange(n,nogil=True):
            for j in range(n):
                # Diagonal two-body contribution
                # <:c^{\dagger}_i c_j: :c^{\dagger}_k c_q:>

                corr += n2a[i,i]*n2b[j,j]*state[i]*state[j]

                #==========================================================
                # Subtract disconnected components
                #==========================================================

                # Diagonal two-body contribution
                if i == j:
                    corr += -0.5*n2a[i,i]*state[i]
                    corr += -0.5*n2b[i,i]*state[i]

                if i != j and state[i]*state[j] != 0:

                    # Four-body contribution
                    # -(1/2)*(<:c^{\dagger}_i c_j c^{\dagger}_k c_q:>)
                    # = (1/2)*(<:c^{\dagger}_i c^{\dagger}_k c_j c_q:>)
                    # => Two contributions: (i==j,k==q) and (i==q,k==j)

                    corr += 0.5*(n4a[i,i,j,j]-n4a[i,j,j,i])
                    corr += 0.5*(n4b[i,i,j,j]-n4b[i,j,j,i])
                
                if i != j:
                    # Off-diagonal two-body contribution
                    corr += n2a[i,j]*n2b[j,i]*state[i]*(1-state[j])

                # Six-body contributions
                for k in range(n):
                    if i != j and i != k and j != k and state[i]*state[j]*state[k] != 0:
                        corr += -0.5*n6a[i,i,j,j,k,k]
                        corr += +0.5*n6a[i,i,j,k,k,j]
                        corr += +0.5*n6a[i,j,j,i,k,k]
                        corr += -0.5*n6a[i,j,j,k,k,i]
                        corr += +0.5*n6a[i,j,k,i,j,k]
                        corr += +0.5*n6a[i,j,k,k,j,i]

                        corr += -0.5*n6b[i,i,j,j,k,k]
                        corr += +0.5*n6b[i,i,j,k,k,j]
                        corr += +0.5*n6b[i,j,j,i,k,k]
                        corr += -0.5*n6b[i,j,j,k,k,i]
                        corr += +0.5*n6b[i,j,k,i,j,k]
                        corr += +0.5*n6b[i,j,k,k,j,i]

                    #-------------------------------------------------
                
                    # Product of quadratic and quartic terms
                    # Contributions should come in pairs, 2x4 and 4x2

                    #==========================================================
                    # n2a x n4b
                    #==========================================================

                    if i != j and i != k and j != k and state[i]*state[j]*state[k] != 0:
                        # Diagonal contribution
                        corr +=  +n2a[i,i]*n4b[j,j,k,k] #
                        corr +=  -n2a[i,i]*n4b[j,k,k,j] #
                        corr +=  -n2a[i,j]*n4b[j,i,k,k] #
                        corr +=  +n2a[i,j]*n4b[j,k,k,i] #
                        corr +=  -n2a[i,j]*n4b[k,k,j,i] #
                        corr +=  +n2a[i,j]*n4b[k,i,j,k] #

                    # Normal-ordering corrections
                    if i != k and state[i]*state[k] != 0:
                        corr += +n2a[i,j]*n4b[j,i,k,k] #
                        corr += -n2a[i,j]*n4b[j,k,k,i] #

                        corr += -n2a[i,j]*n4b[k,i,j,k] #
                        corr += +n2a[i,j]*n4b[k,k,j,i] #
                    

                    #==========================================================
                    # n4a x n2b
                    #==========================================================

                    if i != j and i != k and j != k and state[i]*state[j]*state[k]:
                        # Diagonal contribution
                        corr +=  +n2b[i,i]*n4a[j,j,k,k] #
                        corr +=  -n2b[i,i]*n4a[j,k,k,j] #
                        corr +=  -n2b[i,j]*n4a[j,i,k,k] #
                        corr +=  +n2b[i,j]*n4a[j,k,k,i] #
                        corr +=  -n2b[i,j]*n4a[k,k,j,i] #
                        corr +=  +n2b[i,j]*n4a[k,i,j,k] #

                    # Normal-ordering corrections
                    if i != k and state[i]*state[k] != 0:
                        corr += +n2b[i,j]*n4a[j,i,k,k] #
                        corr += -n2b[i,j]*n4a[j,k,k,i] #

                        corr += -n2b[i,j]*n4a[k,i,j,k] #
                        corr += +n2b[i,j]*n4a[k,k,j,i] #

                    # Terms with four or more number operators
                    for q in range(n):
                        if i != j and i != k and i != q and j != k and j != q and k != q:

                            #==========================================================
                            # n2a x n6b
                            #==========================================================

                            if state[i]*state[j]*state[k]*state[q] != 0:

                                corr += +n2a[i,i]*n6b[j,j,k,k,q,q] #
                                corr += -n2a[i,i]*n6b[j,j,k,q,q,k] #

                                corr += -n2a[i,i]*n6b[k,q,q,k,j,j] #
                                corr += +n2a[i,i]*n6b[k,q,j,k,q,j] #

                                corr += -n2a[i,i]*n6b[k,q,j,j,q,k] #
                                corr += +n2a[i,i]*n6b[k,q,q,j,j,k] #

                                corr += -n2a[i,j]*n6b[j,i,k,k,q,q] #
                                corr += +n2a[i,j]*n6b[j,i,k,q,q,k] #

                                corr += +n2a[i,j]*n6b[k,i,j,k,q,q] #
                                corr += -n2a[i,j]*n6b[k,i,q,k,j,q] #

                                corr += +n2a[i,j]*n6b[k,i,q,q,j,k] #
                                corr += -n2a[i,j]*n6b[k,i,j,q,q,k] #

                                corr += -n2a[i,j]*n6b[k,k,j,i,q,q] #
                                corr += +n2a[i,j]*n6b[k,k,q,i,j,q] #

                                corr += +n2a[i,j]*n6b[j,k,k,i,q,q] #
                                corr += -n2a[i,j]*n6b[j,k,q,i,k,q] #

                                corr += +n2a[i,j]*n6b[k,q,j,i,q,k] #
                                corr += -n2a[i,j]*n6b[k,q,q,i,j,k] #

                                corr += -n2a[i,j]*n6b[k,k,q,q,j,i] #
                                corr += +n2a[i,j]*n6b[k,k,j,q,q,i] #

                                corr += +n2a[i,j]*n6b[q,k,k,q,j,i] #
                                corr += -n2a[i,j]*n6b[q,k,j,q,k,i] #

                                corr += +n2a[i,j]*n6b[j,k,q,q,k,i] #
                                corr += -n2a[i,j]*n6b[j,k,k,q,q,i] #

                            if state[i]*state[k]*state[q] != 0:

                                corr += +n2a[i,j]*n6b[j,i,k,k,q,q] #
                                corr += -n2a[i,j]*n6b[j,i,k,q,q,k] #
                                corr += -n2a[i,j]*n6b[j,k,k,i,q,q] #
                                corr += +n2a[i,j]*n6b[j,k,q,i,k,q] #
                                corr += +n2a[i,j]*n6b[j,k,k,q,q,i] #
                                corr += -n2a[i,j]*n6b[j,k,q,k,q,i] #

                            if state[i]*state[k]*state[q] != 0:

                                corr += -n2a[i,j]*n6b[k,i,j,k,q,q] #
                                corr += +n2a[i,j]*n6b[k,i,j,q,q,k] #
                                corr += +n2a[i,j]*n6b[k,k,j,i,q,q] #
                                corr += -n2a[i,j]*n6b[k,q,j,i,q,k] #
                                corr += -n2a[i,j]*n6b[k,k,j,q,q,i] #
                                corr += +n2a[i,j]*n6b[k,q,j,k,q,i] #

                            if state[i]*state[k]*state[q] != 0:

                                corr += +n2a[i,j]*n6b[k,i,q,k,j,q] #
                                corr += -n2a[i,j]*n6b[k,i,q,q,j,k] #
                                corr += -n2a[i,j]*n6b[k,k,q,i,j,q] #
                                corr += +n2a[i,j]*n6b[k,q,q,i,j,k] #
                                corr += +n2a[i,j]*n6b[k,k,q,q,j,i] #
                                corr += -n2a[i,j]*n6b[k,q,q,k,j,i] #

                            #==========================================================
                            # n6a x n2b
                            #==========================================================

                            if state[i]*state[j]*state[k]*state[q] != 0:

                                corr += +n6a[i,i,j,j,k,k]*n2b[q,q] #
                                corr += -n6a[i,i,j,j,k,q]*n2b[q,k] #

                                corr += -n6a[i,i,k,q,q,k]*n2b[j,j] #
                                corr += +n6a[i,i,k,q,j,k]*n2b[q,j] #

                                corr += -n6a[i,i,k,q,j,j]*n2b[q,k] #
                                corr += +n6a[i,i,k,q,q,j]*n2b[j,k] #

                                corr += -n6a[i,j,j,i,k,k]*n2b[q,q] #
                                corr += +n6a[i,j,j,i,k,q]*n2b[q,k] #

                                corr += +n6a[i,j,k,i,j,k]*n2b[q,q] #
                                corr += -n6a[i,j,k,i,q,k]*n2b[j,q] #

                                corr += +n6a[i,j,k,i,q,q]*n2b[j,k] #
                                corr += -n6a[i,j,k,i,j,q]*n2b[q,k] #

                                corr += -n6a[i,j,k,k,j,i]*n2b[q,q] #
                                corr += +n6a[i,j,k,k,q,i]*n2b[j,q] #

                                corr += +n6a[i,j,j,k,k,i]*n2b[q,q] #
                                corr += -n6a[i,j,j,k,q,i]*n2b[k,q] #

                                corr += +n6a[i,j,k,q,j,i]*n2b[q,k] #
                                corr += -n6a[i,j,k,q,q,i]*n2b[j,k] #

                                corr += -n6a[i,j,k,k,q,q]*n2b[j,i] #
                                corr += +n6a[i,j,k,k,j,q]*n2b[q,i] #

                                corr += +n6a[i,j,q,k,k,q]*n2b[j,i] #
                                corr += -n6a[i,j,q,k,j,q]*n2b[k,i] #

                                corr += +n6a[i,j,j,k,q,q]*n2b[k,i] #
                                corr += -n6a[i,j,j,k,k,q]*n2b[q,i] #

                            if state[j]*state[k]*state[q] != 0:

                                corr += +n6a[k,k,q,q,j,i]*n2b[i,j] #
                                corr += -n6a[k,k,j,q,q,i]*n2b[i,j] #
                                corr += -n6a[k,q,q,k,j,i]*n2b[i,j] #
                                corr += +n6a[k,q,j,k,q,i]*n2b[i,j] #
                                corr += +n6a[j,k,k,q,q,i]*n2b[i,j] #
                                corr += -n6a[j,k,q,q,k,i]*n2b[i,j] #

                                corr += -n6a[k,k,q,i,j,q]*n2b[i,j] #
                                corr += +n6a[k,k,j,i,q,q]*n2b[i,j] #
                                corr += +n6a[k,q,q,i,j,k]*n2b[i,j] #
                                corr += -n6a[k,q,j,i,q,k]*n2b[i,j] #
                                corr += -n6a[j,k,k,i,q,q]*n2b[i,j] #
                                corr += +n6a[j,k,q,i,k,q]*n2b[i,j] #

                                corr += +n6a[k,i,q,k,j,q]*n2b[i,j] #
                                corr += -n6a[k,i,j,k,q,q]*n2b[i,j] #
                                corr += -n6a[k,i,q,q,j,k]*n2b[i,j] #
                                corr += +n6a[k,i,j,q,q,k]*n2b[i,j] #
                                corr += -n6a[j,i,k,q,q,k]*n2b[i,j] #
                                corr += +n6a[j,i,k,k,q,q]*n2b[i,j] #

                            #==========================================================
                            # n4a x n4b
                            #==========================================================

                            # Main contributions
                            if state[i]*state[j]*state[k]*state[q] != 0:
                                corr += +n4a[i,i,j,j]*n4b[k,k,q,q] #
                                corr += -n4a[i,i,j,j]*n4b[k,q,q,k] #

                                corr += -n4a[i,i,k,q]*n4b[q,k,j,j] #
                                corr += +n4a[i,i,k,q]*n4b[j,k,q,j] #

                                corr += -n4a[i,i,k,q]*n4b[j,j,q,k] #
                                corr += +n4a[i,i,k,q]*n4b[q,j,j,k] #

                                corr += -n4a[i,j,j,i]*n4b[k,k,q,q] #
                                corr += +n4a[i,j,j,i]*n4b[k,q,q,k] #

                                corr += +n4a[i,j,k,i]*n4b[j,k,q,q] #
                                corr += -n4a[i,j,k,i]*n4b[q,k,j,q] #

                                corr += +n4a[i,j,k,i]*n4b[q,q,j,k] #
                                corr += -n4a[i,j,k,i]*n4b[j,q,q,k] #

                                corr += -n4a[i,j,k,k]*n4b[j,i,q,q] #
                                corr += +n4a[i,j,k,k]*n4b[q,i,j,q] #

                                corr += +n4a[i,j,j,k]*n4b[k,i,q,q] #
                                corr += -n4a[i,j,j,k]*n4b[q,i,k,q] #

                                corr += +n4a[i,j,k,q]*n4b[j,i,q,k] #
                                corr += -n4a[i,j,k,q]*n4b[q,i,j,k] #

                                corr += -n4a[i,j,k,k]*n4b[q,q,j,i] #
                                corr += +n4a[i,j,k,k]*n4b[j,q,q,i] #

                                corr += +n4a[i,j,q,k]*n4b[k,q,j,i] #
                                corr += -n4a[i,j,q,k]*n4b[j,q,k,i] #

                                corr += +n4a[i,j,j,k]*n4b[q,q,k,i] #
                                corr += -n4a[i,j,j,k]*n4b[k,q,q,i] #

                        # Normal ordering contributions
                        if i != k and i != q and q != k:
                            if state[i]*state[k]*state[q] != 0:
                                corr += +n4a[i,j,k,i]*n4b[q,k,j,q] #
                                corr += -n4a[i,j,k,i]*n4b[q,q,j,k] #
                                corr += +n4a[i,j,k,k]*n4b[q,i,j,q] #
                                corr += -n4a[i,j,k,q]*n4b[q,i,j,k] #
                                corr += -n4a[i,j,k,q]*n4b[q,k,j,i] #
                                corr += +n4a[i,j,k,k]*n4b[q,q,j,i] #

                        if i != k and i != j and j != k:
                            if state[i]*state[k]*state[j] != 0:
                                corr += +n4a[i,i,k,q]*n4b[j,k,q,j] #
                                corr += -n4a[i,i,k,q]*n4b[j,j,q,k] #
                                corr += +n4a[i,j,j,q]*n4b[k,i,q,k] #
                                corr += -n4a[i,j,k,q]*n4b[j,i,q,k] #
                                corr += -n4a[i,j,j,q]*n4b[k,k,q,i] #
                                corr += +n4a[i,j,k,q]*n4b[j,k,q,i] #

                        if i != k and i != q and q != k:
                            if state[i]*state[k]*state[q] != 0:
                                corr += -n4a[i,j,k,i]*n4b[j,k,q,q] #
                                corr += +n4a[i,j,k,i]*n4b[j,q,q,k] #
                                corr += +n4a[i,j,k,k]*n4b[j,i,q,q] #
                                corr += -n4a[i,j,k,q]*n4b[j,i,q,k] #
                                corr += -n4a[i,j,k,k]*n4b[j,q,q,i] #
                                corr += +n4a[i,j,k,q]*n4b[j,k,q,i] #

                        if i != k and i != j and j != k:
                            if state[i]*state[j]*state[k] != 0:
                                corr += +n4a[i,i,k,q]*n4b[q,k,j,j] #
                                corr += -n4a[i,i,k,q]*n4b[q,j,j,k] #
                                corr += -n4a[i,j,j,q]*n4b[q,i,k,k] #
                                corr += +n4a[i,j,k,q]*n4b[q,i,j,k] #
                                corr += -n4a[i,j,k,q]*n4b[q,k,j,i] #
                                corr += +n4a[i,j,j,q]*n4b[q,k,k,i] #

                        if i != k:
                            if state[i]*state[k] != 0:
                                corr += n4a[i,j,k,q]*(-n4b[q,i,j,k]+n4b[j,i,q,k]+n4b[q,k,j,i]-n4b[j,k,q,i]) #

                        #for p in range(n):
                        #    if i != j and i != k and i != q and i != p and j != k and j != q and j != p and k != q and k != p and q != p:
                        #        if state[i]*state[j]*state[k]*state[q]*state[p] != 0:

                                    #==========================================================
                                    # n4a x n6b
                                    #==========================================================

                                    # Diagonal contributions
                        #            corr += n4a[i,i,j,j]*n6b[k,k,q,q,p,p]

                                    #==========================================================
                                    # n6a x n4b
                                    #==========================================================

                                    # Diagonal contributions
                        #            corr += n6a[i,i,j,j,k,k]*n4b[q,q,p,p]

        corr += 0.25
        corr_list[ns] = corr

    return corr_list

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.nonecheck(False)
@cython.cdivision(True)
def cytrace2(int n,double t,double[:,:] H, double[:,:,:,:] Hint, double complex[:,:] n2b, double complex[:,:,:,:] n4b,double complex[:,:,:,:,:,:] n6b, double[:,:] statelist):

    cdef int i,j,k,q,ns
    cdef double complex corr = 0.
    # cdef double [:] state
    cdef double complex[:,:] n2a
    cdef double complex[:,:,:,:] n4a
    cdef double complex[:,:,:,:,:,:] n6a

    cdef np.ndarray[np.complex128_t, ndim=2] corr2 = np.zeros((n,n), dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=3] corr3 = np.zeros((n,n,n), dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=4] corr4 = np.zeros((n,n,n,n), dtype=np.complex128)

    n2a,n4a,n6a = cy_tstep(n,t,H,Hint,n2b,n4b,n6b)

    for i in prange(n,nogil=True,schedule='static'):
        for j in range(n):
            # Diagonal two-body contribution
            # <:c^{\dagger}_i c_j: :c^{\dagger}_k c_q:>

            corr2[i,j] += n2a[i,i]*n2b[j,j]

            #==========================================================
            # Subtract disconnected components
            #==========================================================

            # Diagonal two-body contribution
            if i == j:
                corr2[i,i] += -0.5*n2a[i,i]
                corr2[i,i] += -0.5*n2b[i,i]

            if i != j:

                # Four-body contribution
                # -(1/2)*(<:c^{\dagger}_i c_j c^{\dagger}_k c_q:>)
                # = (1/2)*(<:c^{\dagger}_i c^{\dagger}_k c_j c_q:>)
                # => Two contributions: (i==j,k==q) and (i==q,k==j)

                corr2[i,j] += 0.5*(n4a[i,i,j,j]-n4a[i,j,j,i])
                corr2[i,j] += 0.5*(n4b[i,i,j,j]-n4b[i,j,j,i])
            
            if i != j:
                # Off-diagonal two-body contribution
                corr2[i,i] +=  n2a[i,j]*n2b[j,i]
                corr2[i,j] += -n2a[i,j]*n2b[j,i]

            # Six-body contributions
            for k in range(n):
                if i != j and i != k and j != k:
                    corr3[i,j,k] += -0.5*n6a[i,i,j,j,k,k]
                    corr3[i,j,k] += +0.5*n6a[i,i,j,k,k,j]
                    corr3[i,j,k] += +0.5*n6a[i,j,j,i,k,k]
                    corr3[i,j,k] += -0.5*n6a[i,j,j,k,k,i]
                    corr3[i,j,k] += +0.5*n6a[i,j,k,i,j,k]
                    corr3[i,j,k] += +0.5*n6a[i,j,k,k,j,i]

                    corr3[i,j,k] += -0.5*n6b[i,i,j,j,k,k]
                    corr3[i,j,k] += +0.5*n6b[i,i,j,k,k,j]
                    corr3[i,j,k] += +0.5*n6b[i,j,j,i,k,k]
                    corr3[i,j,k] += -0.5*n6b[i,j,j,k,k,i]
                    corr3[i,j,k] += +0.5*n6b[i,j,k,i,j,k]
                    corr3[i,j,k] += +0.5*n6b[i,j,k,k,j,i]

                #-------------------------------------------------
            
                # Product of quadratic and quartic terms
                # Contributions should come in pairs, 2x4 and 4x2

                #==========================================================
                # n2a x n4b
                #==========================================================

                if i != j and i != k and j != k:
                    # Diagonal contribution
                    corr3[i,j,k] +=  +n2a[i,i]*(n4b[j,j,k,k]-n4b[j,k,k,j]-n4b[j,i,k,k]+n4b[j,k,k,i]+n4b[k,i,j,k]-n4b[k,k,j,i]) #

                # Normal-ordering corrections
                if i != k:
                    corr2[i,k] += +n2a[i,j]*(n4b[j,i,k,k]-n4b[j,k,k,i]+n4b[k,k,j,i]-n4b[k,i,j,k]) #

                #==========================================================
                # n4a x n2b
                #==========================================================

                if i != j and i != k and j != k:
                    # Diagonal contribution
                    corr3[i,j,k] +=  +n2b[i,i]*(n4a[j,j,k,k]-n4a[j,k,k,j]+n4a[j,k,k,i]-n4a[j,i,k,k]+n4a[k,i,j,k]-n4a[k,k,j,i]) #

                # Normal-ordering corrections
                if i != k:
                    corr2[i,k] += +n2b[i,j]*(n4a[j,i,k,k]-n4a[j,k,k,i]+n4a[k,k,j,i]-n4a[k,i,j,k]) #

                # Terms with four or more number operators
                for q in range(n):
                    if i != j and i != k and i != q and j != k and j != q and k != q:

                        #==========================================================
                        # n2a x n6b
                        #==========================================================


                            corr4[i,j,k,q] += +n2a[i,i]*(n6b[j,j,k,k,q,q]-n6b[j,j,k,q,q,k]+n6b[k,q,j,k,q,j]-n6b[k,q,q,k,j,j]+n6b[k,q,q,j,j,k]-n6b[k,q,j,j,q,k]) #
                            corr4[i,j,k,q] += +n2a[i,j]*(n6b[j,i,k,q,q,k]-n6b[j,i,k,k,q,q]+n6b[k,i,j,k,q,q]-n6b[k,i,q,k,j,q]+n6b[k,i,q,q,j,k]-n6b[k,i,j,q,q,k]+n6b[k,k,q,i,j,q]-n6b[k,k,j,i,q,q]) #
                            corr4[i,j,k,q] += +n2a[i,j]*(n6b[j,k,k,i,q,q]-n6b[j,k,q,i,k,q]+n6b[k,q,j,i,q,k]-n6b[k,q,q,i,j,k]+n6b[k,k,j,q,q,i]-n6b[k,k,q,q,j,i]+n6b[q,k,k,q,j,i]-n6b[q,k,j,q,k,i]+n6b[j,k,q,q,k,i]-n6b[j,k,k,q,q,i]) #


                    if i != k and i != q and k != q:

                        corr3[i,k,q] += +n2a[i,j]*(n6b[j,i,k,k,q,q]-n6b[j,i,k,q,q,k]-n6b[j,k,k,i,q,q]+n6b[j,k,q,i,k,q]+n6b[j,k,k,q,q,i]-n6b[j,k,q,k,q,i]) #
                        corr3[i,k,q] += +n2a[i,j]*(n6b[k,i,j,q,q,k]-n6b[k,i,j,k,q,q]+n6b[k,k,j,i,q,q]-n6b[k,q,j,i,q,k]-n6b[k,k,j,q,q,i]+n6b[k,q,j,k,q,i]) #
                        corr3[i,k,q] += +n2a[i,j]*(n6b[k,i,q,k,j,q]-n6b[k,i,q,q,j,k]-n6b[k,k,q,i,j,q]+n6b[k,q,q,i,j,k]+n6b[k,k,q,q,j,i]-n6b[k,q,q,k,j,i]) #

                        #==========================================================
                        # n6a x n2b
                        #==========================================================
                    if i != j and i != k and i != q and j != k and j != q and k != q:
                            corr4[i,j,k,q] += +(n6a[i,i,j,j,k,k]-n6a[i,i,k,j,j,k]-n6a[i,j,j,i,k,k]+n6a[i,j,k,i,j,k]-n6a[i,j,k,k,j,i]+n6a[i,j,j,k,k,i])*n2b[q,q] #
                            corr4[i,j,k,q] += -n6a[i,i,j,j,k,q]*n2b[q,k] #

                            # corr4[i,j,k,q] += -n6a[i,i,k,q,q,k]*n2b[j,j] # ( j <-> q in the above recombination)
                            corr4[i,j,k,q] += +n6a[i,i,k,q,j,k]*n2b[q,j] #

                            corr4[i,j,k,q] += -n6a[i,i,k,q,j,j]*n2b[q,k] #
                            corr4[i,j,k,q] += +n6a[i,i,k,q,q,j]*n2b[j,k] #

                            # corr4[i,j,k,q] += -n6a[i,j,j,i,k,k]*n2b[q,q] #
                            corr4[i,j,k,q] += +n6a[i,j,j,i,k,q]*n2b[q,k] #

                            # corr4[i,j,k,q] += +n6a[i,j,k,i,j,k]*n2b[q,q] #
                            corr4[i,j,k,q] += -n6a[i,j,k,i,q,k]*n2b[j,q] #

                            corr4[i,j,k,q] += +n6a[i,j,k,i,q,q]*n2b[j,k] #
                            corr4[i,j,k,q] += -n6a[i,j,k,i,j,q]*n2b[q,k] #

                            # corr4[i,j,k,q] += -n6a[i,j,k,k,j,i]*n2b[q,q] #
                            corr4[i,j,k,q] += +n6a[i,j,k,k,q,i]*n2b[j,q] #

                            # corr4[i,j,k,q] += +n6a[i,j,j,k,k,i]*n2b[q,q] #
                            corr4[i,j,k,q] += -n6a[i,j,j,k,q,i]*n2b[k,q] #

                            corr4[i,j,k,q] += +n6a[i,j,k,q,j,i]*n2b[q,k] #
                            corr4[i,j,k,q] += -n6a[i,j,k,q,q,i]*n2b[j,k] #

                            corr4[i,j,k,q] += -n6a[i,j,k,k,q,q]*n2b[j,i] #
                            corr4[i,j,k,q] += +n6a[i,j,k,k,j,q]*n2b[q,i] #

                            corr4[i,j,k,q] += +n6a[i,j,q,k,k,q]*n2b[j,i] #
                            corr4[i,j,k,q] += -n6a[i,j,q,k,j,q]*n2b[k,i] #

                            corr4[i,j,k,q] += +n6a[i,j,j,k,q,q]*n2b[k,i] #
                            corr4[i,j,k,q] += -n6a[i,j,j,k,k,q]*n2b[q,i] #

                    if j != k and j != q and k != q:

                        corr3[j,k,q] += +(n6a[k,k,q,q,j,i]-n6a[k,k,j,q,q,i]-n6a[k,q,q,k,j,i]+n6a[k,q,j,k,q,i]+n6a[j,k,k,q,q,i]-n6a[j,k,q,q,k,i])*n2b[i,j] #
                        corr3[j,k,q] += +(n6a[k,k,j,i,q,q]-n6a[k,k,q,i,j,q]+n6a[k,q,q,i,j,k]-n6a[k,q,j,i,q,k]-n6a[j,k,k,i,q,q]+n6a[j,k,q,i,k,q])*n2b[i,j] #
                        corr3[j,k,q] += +(n6a[k,i,q,k,j,q]-n6a[k,i,j,k,q,q]-n6a[k,i,q,q,j,k]+n6a[k,i,j,q,q,k]-n6a[j,i,k,q,q,k]+n6a[j,i,k,k,q,q])*n2b[i,j] #

                        #==========================================================
                        # n4a x n4b
                        #==========================================================
                    if i != j and i != k and i != q and j != k and j != q and k != q:
                        # Main contributions
                            corr4[i,j,k,q] += +n4a[i,i,j,j]*(n4b[k,k,q,q]-n4b[k,q,q,k]) #
                            corr4[i,j,k,q] += +n4a[i,i,k,q]*(n4b[j,k,q,j]-n4b[q,k,j,j]+n4b[q,j,j,k]-n4b[j,j,q,k]) #
                            corr4[i,j,k,q] += +n4a[i,j,j,i]*(n4b[k,q,q,k]-n4b[k,k,q,q]) #
                            corr4[i,j,k,q] += +n4a[i,j,k,i]*(n4b[j,k,q,q]-n4b[q,k,j,q]+n4b[q,q,j,k]-n4b[j,q,q,k]) #
                            corr4[i,j,k,q] += +n4a[i,j,k,k]*(n4b[q,i,j,q]-n4b[j,i,q,q]+n4b[j,q,q,i]-n4b[q,q,j,i]+n4b[k,i,q,q]-n4b[q,i,k,q]) #
                            corr4[i,j,k,q] += +n4a[i,j,k,q]*(n4b[j,i,q,k]-n4b[q,i,j,k]) #
                            corr4[i,j,k,q] += +n4a[i,j,q,k]*(n4b[k,q,j,i]-n4b[j,q,k,i]) #
                            corr4[i,j,k,q] += +n4a[i,j,j,k]*(n4b[q,q,k,i]-n4b[k,q,q,i]) #

                    # Normal ordering contributions
                    if i != k and i != q and q != k:
                            corr3[i,k,q] += +n4a[i,j,k,i]*(n4b[q,k,j,q]-n4b[q,q,j,k]-n4b[j,k,q,q]+n4b[j,q,q,k]) #
                            corr3[i,k,q] += +n4a[i,j,k,k]*(n4b[q,i,j,q]+n4b[q,q,j,i]+n4b[j,i,q,q]-n4b[j,q,q,i]) #
                            corr3[i,k,q] += -n4a[i,j,k,q]*(n4b[q,i,j,k]+n4b[q,k,j,i]-n4b[j,i,q,k]+n4b[j,k,q,i]) #

                    if i != k and i != j and j != k:
                            corr3[i,k,j] += +n4a[i,i,k,q]*(n4b[j,k,q,j]-n4b[j,j,q,k]+n4b[q,k,j,j]-n4b[q,j,j,k]) #
                            corr3[i,k,j] += +n4a[i,j,j,q]*(n4b[k,i,q,k]-n4b[k,k,q,i]-n4b[q,i,k,k]+n4b[k,k,q,i]) #
                            corr3[i,k,j] += -n4a[i,j,k,q]*(n4b[j,i,q,k]+n4b[j,k,q,i]+n4b[q,i,j,k]-n4b[q,k,j,i]) #

                    if i != k:
                            corr2[i,k] += n4a[i,j,k,q]*(-n4b[q,i,j,k]+n4b[j,i,q,k]+n4b[q,k,j,i]-n4b[j,k,q,i]) #

    for ns in range(len(statelist)):
        for i in prange(n,nogil=True,schedule='static'):
            for j in range(n):
                corr += corr2[i,j]*statelist[ns,i]*statelist[ns,j]
                for k in range(n):
                    corr += corr3[i,j,k]*statelist[ns,i]*statelist[ns,j]*statelist[ns,k]
                    for q in range(n):
                        corr += corr4[i,j,k,q]*statelist[ns,i]*statelist[ns,j]*statelist[ns,k]*statelist[ns,q]

        corr += 0.25

    return corr/len(statelist)

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.nonecheck(False)
@cython.cdivision(True)
def cytrace3(int n,double t,double[:,:] H, double[:,:,:,:] Hint, double complex[:,:] n2b, double complex[:,:,:,:] n4b, double[:,:] statelist):

    cdef int i,j,k,q,ns
    cdef double complex corr = 0.
    # cdef double [:] state
    cdef double complex[:,:] n2a
    cdef double complex[:,:,:,:] n4a

    cdef np.ndarray[np.complex128_t, ndim=2] corr2 = np.zeros((n,n), dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=3] corr3 = np.zeros((n,n,n), dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=4] corr4 = np.zeros((n,n,n,n), dtype=np.complex128)

    n2a,n4a = cy_tstep2(n,t,H,Hint,n2b,n4b)

    for i in prange(n,nogil=True,schedule='static'):
        for j in range(n):
            # Diagonal two-body contribution
            # <:c^{\dagger}_i c_j: :c^{\dagger}_k c_q:>

            corr2[i,j] += n2a[i,i]*n2b[j,j]

            #==========================================================
            # Subtract disconnected components
            #==========================================================

            # Diagonal two-body contribution
            if i == j:
                corr2[i,i] += -0.5*n2a[i,i]
                corr2[i,i] += -0.5*n2b[i,i]

            if i != j:

                # Four-body contribution
                # -(1/2)*(<:c^{\dagger}_i c_j c^{\dagger}_k c_q:>)
                # = (1/2)*(<:c^{\dagger}_i c^{\dagger}_k c_j c_q:>)
                # => Two contributions: (i==j,k==q) and (i==q,k==j)

                corr2[i,j] += 0.5*(n4a[i,i,j,j]-n4a[i,j,j,i])
                corr2[i,j] += 0.5*(n4b[i,i,j,j]-n4b[i,j,j,i])
            
            if i != j:
                # Off-diagonal two-body contribution
                corr2[i,i] +=  n2a[i,j]*n2b[j,i]
                corr2[i,j] += -n2a[i,j]*n2b[j,i]

                #-------------------------------------------------
            
                # Product of quadratic and quartic terms
                # Contributions should come in pairs, 2x4 and 4x2

            for k in range(n):

                #==========================================================
                # n2a x n4b
                #==========================================================

                if i != j and i != k and j != k:
                    # Diagonal contribution
                    corr3[i,j,k] +=  +n2a[i,i]*(n4b[j,j,k,k]-n4b[j,k,k,j]-n4b[j,i,k,k]+n4b[j,k,k,i]+n4b[k,i,j,k]-n4b[k,k,j,i]) #

                # Normal-ordering corrections
                if i != k:
                    corr2[i,k] += +n2a[i,j]*(n4b[j,i,k,k]-n4b[j,k,k,i]+n4b[k,k,j,i]-n4b[k,i,j,k]) #

                #==========================================================
                # n4a x n2b
                #==========================================================

                if i != j and i != k and j != k:
                    # Diagonal contribution
                    corr3[i,j,k] +=  +n2b[i,i]*(n4a[j,j,k,k]-n4a[j,k,k,j]+n4a[j,k,k,i]-n4a[j,i,k,k]+n4a[k,i,j,k]-n4a[k,k,j,i]) #

                # Normal-ordering corrections
                if i != k:
                    corr2[i,k] += +n2b[i,j]*(n4a[j,i,k,k]-n4a[j,k,k,i]+n4a[k,k,j,i]-n4a[k,i,j,k]) #

                # Terms with four or more number operators
                for q in range(n):
                        #==========================================================
                        # n4a x n4b
                        #==========================================================
                    if i != j and i != k and i != q and j != k and j != q and k != q:
                        # Main contributions
                            corr4[i,j,k,q] += +n4a[i,i,j,j]*(n4b[k,k,q,q]-n4b[k,q,q,k]) #
                            corr4[i,j,k,q] += +n4a[i,i,k,q]*(n4b[j,k,q,j]-n4b[q,k,j,j]+n4b[q,j,j,k]-n4b[j,j,q,k]) #
                            corr4[i,j,k,q] += +n4a[i,j,j,i]*(n4b[k,q,q,k]-n4b[k,k,q,q]) #
                            corr4[i,j,k,q] += +n4a[i,j,k,i]*(n4b[j,k,q,q]-n4b[q,k,j,q]+n4b[q,q,j,k]-n4b[j,q,q,k]) #
                            corr4[i,j,k,q] += +n4a[i,j,k,k]*(n4b[q,i,j,q]-n4b[j,i,q,q]+n4b[j,q,q,i]-n4b[q,q,j,i]+n4b[k,i,q,q]-n4b[q,i,k,q]) #
                            corr4[i,j,k,q] += +n4a[i,j,k,q]*(n4b[j,i,q,k]-n4b[q,i,j,k]) #
                            corr4[i,j,k,q] += +n4a[i,j,q,k]*(n4b[k,q,j,i]-n4b[j,q,k,i]) #
                            corr4[i,j,k,q] += +n4a[i,j,j,k]*(n4b[q,q,k,i]-n4b[k,q,q,i]) #

                    # Normal ordering contributions
                    if i != k and i != q and q != k:
                            corr3[i,k,q] += +n4a[i,j,k,i]*(n4b[q,k,j,q]-n4b[q,q,j,k]-n4b[j,k,q,q]+n4b[j,q,q,k]) #
                            corr3[i,k,q] += +n4a[i,j,k,k]*(n4b[q,i,j,q]+n4b[q,q,j,i]+n4b[j,i,q,q]-n4b[j,q,q,i]) #
                            corr3[i,k,q] += -n4a[i,j,k,q]*(n4b[q,i,j,k]+n4b[q,k,j,i]-n4b[j,i,q,k]+n4b[j,k,q,i]) #

                    if i != k and i != j and j != k:
                            corr3[i,k,j] += +n4a[i,i,k,q]*(n4b[j,k,q,j]-n4b[j,j,q,k]+n4b[q,k,j,j]-n4b[q,j,j,k]) #
                            corr3[i,k,j] += +n4a[i,j,j,q]*(n4b[k,i,q,k]-n4b[k,k,q,i]-n4b[q,i,k,k]+n4b[k,k,q,i]) #
                            corr3[i,k,j] += -n4a[i,j,k,q]*(n4b[j,i,q,k]+n4b[j,k,q,i]+n4b[q,i,j,k]-n4b[q,k,j,i]) #

                    if i != k:
                            corr2[i,k] += n4a[i,j,k,q]*(-n4b[q,i,j,k]+n4b[j,i,q,k]+n4b[q,k,j,i]-n4b[j,k,q,i]) #

    for ns in range(len(statelist)):
        for i in prange(n,nogil=True,schedule='static'):
            for j in range(n):
                corr += corr2[i,j]*statelist[ns,i]*statelist[ns,j]
                for k in range(n):
                    corr += corr3[i,j,k]*statelist[ns,i]*statelist[ns,j]*statelist[ns,k]
                    for q in range(n):
                        corr += corr4[i,j,k,q]*statelist[ns,i]*statelist[ns,j]*statelist[ns,k]*statelist[ns,q]

        corr += 0.25

    return corr/len(statelist)

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.nonecheck(False)
@cython.cdivision(True)
def cytrace_nonint(int n,double t,double[:,:] H, double complex[:,:] n2b, double[:,:] statelist, double complex[:] corr_list):

    cdef int i,j,ns
    cdef double complex corr = 0.
    cdef double [:] state
    cdef double complex[:,:] n2a

    n2a = cy_tstep_nonint(n,t,H,n2b)

    for ns in range(len(statelist)):
        state = statelist[ns]
        corr = 0.
        # Product of quadratic terms
        for i in prange(n,nogil=True):
            for j in range(n):
                # Diagonal two-body contribution
                # <:c^{\dagger}_i c_j: :c^{\dagger}_k c_q:>

                corr += n2a[i,i]*n2b[j,j]*state[i]*state[j]

                #==========================================================
                # Subtract disconnected components
                #==========================================================

                # Diagonal two-body contribution
                if i == j:
                    corr += -0.5*n2a[i,i]*state[i]
                    corr += -0.5*n2b[i,i]*state[i]

                if i != j:
                    # Off-diagonal two-body contribution
                    corr += n2a[i,j]*n2b[j,i]*state[i]*(1-state[j])

        corr += 0.25
        corr_list[ns] = corr

    return corr_list

cdef extern from "complex.h" nogil:
    double complex cexp(double complex)

cdef int cround_old(double a, double b, double precision) nogil:
    cdef int c = 0
    if (a-b)**2 > precision**2:
        c=1
    return c

cdef int cround(double a, double b, double complex c, double d, double e) nogil:
    cdef int c0 = 0
    temp = ((c.real)**2+(c.imag)**2)
    if d != e: # and (d-e)**2 > 1e-4:
        #if temp*((a-b)/(d-e))**2 < 0.25:
        #    c0=1
        #else:
        #    with gil:
        #        print(a,b,c,d,e)
        c0 = 1
    else:
        with gil:
                print('DIVERGENT TERMS IN DYNAMICS',a,b,c,d,e)
    return c0
    
@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.nonecheck(False)
@cython.cdivision(True)
cdef cy_tstep(int n,double t,double[:,:] H, double[:,:,:,:] Hint, double complex[:,:] n2, double complex[:,:,:,:] n4, double complex[:,:,:,:,:,:] n6):

    cdef int i,j,k,q,m,l
    cdef double phase
    cdef np.ndarray[np.int32_t, ndim=4] test = np.zeros((n,n,n,n), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=6] test2 = np.zeros((n,n,n,n,n,n), dtype=np.int32)

    cdef np.ndarray[np.complex128_t, ndim=2] n2t = np.zeros((n,n), dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=4] n4t = np.zeros((n,n,n,n), dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=6] n6t = np.zeros((n,n,n,n,n,n), dtype=np.complex128)

    # Apply the phase shift to all orders
    for i in prange(n,nogil=True):
        for j in range(n):
            phase = (H[i,i]-H[j,j])
            n2t[i,j] = n2[i,j]*cexp(1j*phase*t)     
            for k in range(n):
                for q in range(n):
                    if test[i,j,k,q] == 0:
                        phase = (H[i,i]+H[k,k]-H[j,j]-H[q,q])
                        n4t[i,j,k,q] = n4[i,j,k,q]*cexp(1j*phase*t)
                        n4t[q,k,j,i] = n4[q,k,j,i]*cexp(-1j*phase*t)
                        test[q,k,j,i] = 1
                        test[i,j,k,q] = 1
                    for m in range(n):
                        for l in range(n):
                            if test2[i,j,k,q,m,l] == 0:
                                phase = (H[i,i]+H[k,k]+H[m,m]-H[j,j]-H[q,q]-H[l,l])
                                n6t[i,j,k,q,m,l] = n6[i,j,k,q,m,l]*cexp(1j*phase*t)
                                n6t[l,m,q,k,j,i] = n6[l,m,q,k,j,i]*cexp(-1j*phase*t)
                                test2[i,j,k,q,m,l] = 1
                                test2[l,m,q,k,j,i] = 1

    # Add any additional terms to all orders
    for i in prange(n,nogil=True):
        for j in range(n):       
            for k in range(n):
                # The 'cround' function here is used to filter out near-degeneracies that lead to divergent terms
                # Note that the 'else' statement below is deprecated and cannot be reached due to the cround call.

                if i != j and i != k and j != k: # and cround(Hint[i,i,j,j],Hint[i,i,k,k],(n2[j,k]*(cexp(1j*(H[j,j]-H[k,k])*t)-1)),H[j,j],H[k,k]) !=0:
                    if H[j,j] != H[k,k]:
                        n4t[i,i,j,k] = n4t[i,i,j,k] + 1*(Hint[i,i,j,j]-Hint[i,i,k,k])*n2[j,k]*(cexp(1j*(H[j,j]-H[k,k])*t)-1)/(H[j,j]-H[k,k])
                    else:
                        n4t[i,i,j,k] = n4t[i,i,j,k] + 1j*(Hint[i,i,j,j]-Hint[i,i,k,k])*n2[j,k]*t
                if i != k and j != k and i != j: # and cround(Hint[i,i,k,k],Hint[j,j,k,k],(n2[i,j]*(cexp(1j*(H[i,i]-H[j,j])*t)-1)),H[i,i],H[j,j]) != 0:
                    if H[i,i] != H[j,j]:
                        n4t[i,j,k,k] = n4t[i,j,k,k] + 1*(Hint[i,i,k,k]-Hint[j,j,k,k])*n2[i,j]*(cexp(1j*(H[i,i]-H[j,j])*t)-1)/(H[i,i]-H[j,j])
                    else:
                        n4t[i,j,k,k] = n4t[i,j,k,k] + 1j*(Hint[i,i,k,k]-Hint[j,j,k,k])*n2[i,j]*t

    # for i in prange(n,nogil=True):
    #     n4t[i,:,i,:] = 0.
    #     n4t[:,i,:,i] = 0.
    #     n6t[i,:,i,:,:,:] = 0.
    #     n6t[i,:,:,:,i,:] = 0.
    #     n6t[:,:,i,:,i,:] = 0.
    #     n6t[:,i,:,i,:,:] = 0.
    #     n6t[:,i,:,:,:,i] = 0.
    #     n6t[:,:,:,i,:,i] = 0.

    # Hermitian test: comment out for speed in 'real' simulations
    #for i in range(n):
    #    for j in range(n):
    #        for k in range(n):
    #            for q in range(n):
    #                if np.round(n4[i,j,k,q],6)!=np.round(np.conjugate((n4.T)[i,j,k,q]),6):
    #                    print('HTEST',np.round(n4[i,j,k,q],6),np.round(np.conjugate((n4.T)[i,j,k,q]),6))

    return n2t,n4t,n6t


@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.nonecheck(False)
@cython.cdivision(True)
cdef cy_tstep_nonint(int n,double t,double[:,:] H, double complex[:,:] n2):

    cdef int i,j
    cdef double phase
    cdef np.ndarray[np.complex128_t, ndim=2] n2t = np.zeros((n,n), dtype=np.complex128)

    # Apply the phase shift
    for i in prange(n,nogil=True):
        for j in range(n):
            phase = (H[i,i]-H[j,j])
            n2t[i,j] = n2[i,j]*cexp(1j*phase*t)     

    return n2t

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.nonecheck(False)
@cython.cdivision(True)
def cy_levels(int n, double[:,:] H0, double[:,:,:,:] Hint,int[:] lev0,int order,double[:,:,:,:,:,:] H6):
    cdef double flev = 0.
    cdef int j,k,q
    for j in prange(n,nogil=True):
        flev += H0[j,j]*(lev0[j])
        for q in range(n):
            if q !=j:
                flev += Hint[j,j,q,q]*(lev0[j])*(lev0[q]) 
                flev += Hint[j,q,q,j]*(lev0[j])*(1-(lev0[q]))
            if order == 6:
                for k in range(n):
                    if j != q and q != k and j != k:
                        flev += H6[j,j,q,q,k,k]*(lev0[j])*(lev0[q])*(lev0[k])
                        flev += H6[q,j,j,q,k,k]*(1-(lev0[j]))*(lev0[q])*(lev0[k])
                        flev += H6[j,k,q,q,k,j]*(lev0[j])*(lev0[q])*(1-(lev0[k]))
                        flev += H6[j,j,q,k,k,q]*(lev0[j])*(lev0[q])*(1-(lev0[k]))
                        flev += H6[j,q,q,k,k,j]*(lev0[j])*(1-(lev0[q]))*(1-(lev0[k]))
                        flev += -H6[j,q,k,j,q,k]*(lev0[j])*(1-(lev0[q]))*((lev0[k]))
    return flev


@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.nonecheck(False)
@cython.cdivision(True)
cdef cy_tstep2(int n,double t,double[:,:] H, double[:,:,:,:] Hint, double complex[:,:] n2, double complex[:,:,:,:] n4):

    cdef int i,j,k,q
    cdef double phase
    cdef np.ndarray[np.int32_t, ndim=4] test = np.zeros((n,n,n,n), dtype=np.int32)

    cdef np.ndarray[np.complex128_t, ndim=2] n2t = np.zeros((n,n), dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=4] n4t = np.zeros((n,n,n,n), dtype=np.complex128)

    # Apply the phase shift to all orders
    for i in prange(n,nogil=True):
        for j in range(n):
            phase = (H[i,i]-H[j,j])
            n2t[i,j] = n2[i,j]*cexp(1j*phase*t)     
            for k in range(n):
                for q in range(n):
                    if test[i,j,k,q] == 0:
                        phase = (H[i,i]+H[k,k]-H[j,j]-H[q,q])
                        n4t[i,j,k,q] = n4[i,j,k,q]*cexp(1j*phase*t)
                        n4t[q,k,j,i] = n4[q,k,j,i]*cexp(-1j*phase*t)
                        test[q,k,j,i] = 1
                        test[i,j,k,q] = 1

    # Add any additional terms to all orders
    for i in prange(n,nogil=True):
        for j in range(n):       
            for k in range(n):
                # The 'cround' function here is used to filter out near-degeneracies that lead to divergent terms
                # Note that the 'else' statement below is deprecated and cannot be reached due to the cround call.

                if i != j and i != k and j != k: # and cround(Hint[i,i,j,j],Hint[i,i,k,k],(n2[j,k]*(cexp(1j*(H[j,j]-H[k,k])*t)-1)),H[j,j],H[k,k]) !=0:
                    if H[j,j] != H[k,k]:
                        n4t[i,i,j,k] = n4t[i,i,j,k] + 1*(Hint[i,i,j,j]-Hint[i,i,k,k])*n2[j,k]*(cexp(1j*(H[j,j]-H[k,k])*t)-1)/(H[j,j]-H[k,k])
                    else:
                        n4t[i,i,j,k] = n4t[i,i,j,k] + 1j*(Hint[i,i,j,j]-Hint[i,i,k,k])*n2[j,k]*t
                if i != k and j != k and i != j: # and cround(Hint[i,i,k,k],Hint[j,j,k,k],(n2[i,j]*(cexp(1j*(H[i,i]-H[j,j])*t)-1)),H[i,i],H[j,j]) != 0:
                    if H[i,i] != H[j,j]:
                        n4t[i,j,k,k] = n4t[i,j,k,k] + 1*(Hint[i,i,k,k]-Hint[j,j,k,k])*n2[i,j]*(cexp(1j*(H[i,i]-H[j,j])*t)-1)/(H[i,i]-H[j,j])
                    else:
                        n4t[i,j,k,k] = n4t[i,j,k,k] + 1j*(Hint[i,i,k,k]-Hint[j,j,k,k])*n2[i,j]*t

    return n2t,n4t