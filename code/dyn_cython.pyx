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
def cytrace(int n,double t,double[:,:] H, double[:,:,:,:] Hint, double complex[:,:] matb, double complex[:,:] n2a, double complex[:,:,:,:] n4a,double complex[:,:,:,:,:,:] n6a,double complex[:,:] n2b, double complex[:,:,:,:] n4b,double complex[:,:,:,:,:,:] n6b, double[:,:] statelist, double complex[:] corr_list):

    cdef int i,j,k,q,p,ns
    cdef double complex corr = 0.
    cdef double [:] state

    n2a,n4a,n6a = cy_tstep(n,t,H,Hint,n2a,matb,n4a,n6a)

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

cdef extern from "complex.h" nogil:
    double complex cexp(double complex)

cdef cround(double a, double b, double precision):
    if (a-b)**2 < precision**2:
        return False
    else:
        return True

@cython.boundscheck(False)  
@cython.wraparound(False)  
@cython.nonecheck(False)
@cython.cdivision(True)
cdef cy_tstep(int n,double t,double[:,:] H, double[:,:,:,:] Hint, double complex[:,:] n2, double complex[:,:] matb, double complex[:,:,:,:] n4, double complex[:,:,:,:,:,:] n6):

    cdef int i,j,k,q,m,l
    cdef double phase
    cdef np.ndarray[np.int32_t, ndim=4] test = np.zeros((n,n,n,n), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=6] test2 = np.zeros((n,n,n,n,n,n), dtype=np.int32)

    # Apply the phase shift to all orders
    for i in prange(n,nogil=True):
        for j in range(n):
            phase = (H[i,i]-H[j,j])
            n2[i,j] = n2[i,j]*cexp(1j*phase*t)     
            for k in range(n):
                for q in range(n):
                    if test[i,j,k,q] == 0:
                        phase = (H[i,i]+H[k,k]-H[j,j]-H[q,q])
                        n4[i,j,k,q] = n4[i,j,k,q]*cexp(1j*phase*t)
                        n4[q,k,j,i] = n4[q,k,j,i]*cexp(-1j*phase*t)
                        test[q,k,j,i] = 1
                        test[i,j,k,q] = 1
                    for m in range(n):
                        for l in range(n):
                            if test2[i,j,k,q,m,l] == 0:
                                phase = (H[i,i]+H[k,k]+H[m,m]-H[j,j]-H[q,q]-H[l,l])
                                n6[i,j,k,q,m,l] = n6[i,j,k,q,m,l]*cexp(1j*phase*t)
                                n6[l,m,q,k,j,i] = n6[l,m,q,k,j,i]*cexp(-1j*phase*t)
                                test2[i,j,k,q,m,l] = 1
                                test2[l,m,q,k,j,i] = 1

    # Add any additional terms to all orders
    for i in range(n):
        for j in range(n):       
            for k in range(n):
                if i != j and i != k:
                    if H[j,j] != H[k,k] and cround(H[j,j],H[k,k],1e-3)==True:
                        n4[i,i,j,k] = n4[i,i,j,k] + 1*(Hint[i,i,j,j]-Hint[i,i,k,k])*matb[j,k]*(cexp(1j*(H[j,j]-H[k,k])*t)-1)/(H[j,j]-H[k,k])
                    else:
                        n4[i,i,j,k] = n4[i,i,j,k] + 1j*(Hint[i,i,j,j]-Hint[i,i,k,k])*matb[j,k]*t
                if i != k and j != k:
                    if H[i,i] != H[j,j] and cround(H[i,i],H[j,j],1e-3)==True:
                        n4[i,j,k,k] = n4[i,j,k,k] + 1*(Hint[i,i,k,k]-Hint[j,j,k,k])*matb[i,j]*(cexp(1j*(H[i,i]-H[j,j])*t)-1)/(H[i,i]-H[j,j])
                    else:
                        n4[i,j,k,k] = n4[i,j,k,k] + 1j*(Hint[i,i,k,k]-Hint[j,j,k,k])*matb[i,j]*t

    for i in prange(n,nogil=True):
        n4[i,:,i,:] = 0.
        n4[:,i,:,i] = 0.
        n6[i,:,i,:,:,:] = 0.
        n6[i,:,:,:,i,:] = 0.
        n6[:,:,i,:,i,:] = 0.
        n6[:,i,:,i,:,:] = 0.
        n6[:,i,:,:,:,i] = 0.
        n6[:,:,:,i,:,i] = 0.

    # Hermitian test: comment out for speed in 'real' simulations
    #for i in range(n):
    #    for j in range(n):
    #        for k in range(n):
    #            for q in range(n):
    #                if np.round(n4[i,j,k,q],6)!=np.round(np.conjugate((n4.T)[i,j,k,q]),6):
    #                    print('HTEST',np.round(n4[i,j,k,q],6),np.round(np.conjugate((n4.T)[i,j,k,q]),6))

    return n2,n4,n6