import h5py
import core.utility as utility
import numpy as np
import os, sys, copy
from quspin.basis import spinless_fermion_basis_1d

# Parameters
L = int(sys.argv[1])            # Linear system size
dim = int(sys.argv[2])          # Spatial dimension
n = L**dim                      # Total number of sites
species = 'spinless fermion'    # Type of particle
dsymm = 'charge'                # Type of disorder (spinful fermions only)
delta = 0.1
# List of interaction strengths
J = 1.0                         # Nearest-neighbour hopping amplitude
cutoff = J*10**(-3)             # Cutoff for the off-diagonal elements to be considered zero
dis = [float(sys.argv[3])]
# List of disorder strengths
order = 4
reps = 1024                     # Number of disorder realisations
norm = False                    # Normal-ordering, can be true or false
intr = True                     # Turn on/off interactions
dyn = False
imbalance = True                # Sets whether to compute global imbalance or single-site dynamics
LIOM = 'bck'                    # Compute LIOMs with forward ('fwd') or backward ('bck') flow
                                # Forward uses less memory by a factor of qmax, and transforms a local operator
                                # in the initial basis into the diagonal basis; backward does the reverse
dis_type = str(sys.argv[4])     # Options: 'random', 'QPgolden', 'QPsilver', 'QPbronze', 'QPrandom', 'linear', 'curved', 'prime'
                                # Also contains 'test' and 'QPtest', potentials that do not change from run to run
x = 0.

# Make directory to store data
nvar = utility.namevar(dis_type,dim,dsymm,'CDW',dyn,norm,n,LIOM,species,4)
nvar2 = utility.namevar3(dis_type,dim,dsymm,'CDW',dyn,norm,n,LIOM,species,4)
if not os.path.exists('%s' %(nvar2)):
    os.makedirs('%s' %(nvar2))

# @jit(parallel=True)
# def avg_n(c1,c3,n):
#     n1 = np.zeros(n)
#     n2 = np.zeros((n,n))
#     n3 = np.zeros((n,n,n))
#     for i in prange(n):
#         n1[i] += np.abs(c1[i])**2
#         for j in range(n):
#             if i != j:
#                 n2[i,j] += c1[i]*c3[i,j,j] + -1*c3[i,j,i]*c1[j]
#                 for k in range(n):
#                     if i != k and j != k:
#                         n3[i,j,k] =     c3[i,j,j]*c3[k,k,i]
#                         n3[i,j,k] += -1*c3[i,j,k]*c3[k,j,i]
#                         n2[i,j] += c3[i,j,k]*c3[k,j,i]
#                         n3[i,j,k] += -1*c3[i,j,i]*c3[k,k,j]
#                         n3[i,j,k] +=    c3[i,j,i]*c3[k,j,k]
#                         n3[i,j,k] +=    c3[i,j,j]*c3[k,i,k]
#                         n2[i,j] += -1*c3[i,j,j]*c3[k,i,k]
#     return n1,n2,n3

# @jit(parallel=True)
def avg_n(c1,c3,n):
    n1 = np.einsum('i,i->i',c1,c1)
    n2 = -np.einsum('i,jij->ij',c1,c3) + np.einsum('i,jji->ij',c1,c3) -np.einsum('iji,j->ij',c3,c1) + np.einsum('ijj,i->ij',c3,c1)
    n3 = np.einsum('ijj,kki->ijk',c3,c3) -np.einsum('ijk,kji->ijk',c3,c3) - np.einsum('iji,kkj->ijk',c3,c3) + np.einsum('iji,kjk->ijk',c3,c3) - np.einsum('ijj,kik->ijk',c3,c3) + np.einsum('ijk,kij->ijk',c3,c3)
    n2 += np.einsum('ijk,kji->ij',c3,c3) - np.einsum('ijk,kij->ij',c3,c3)

    return n1,n2,n3

# @jit(parallel=True)
# def long_time(statelist,n,n1,n2,n3):
#     no_states = len(statelist)
#     ltc = np.zeros(no_states)
#     for ns in prange(no_states):
#         for i in range(n):
#             ltc[ns] += -2*0.5*n1[i]*statelist[ns,i]
#             for j in range(n):  
#                 ltc[ns] += n1[i]*n1[j]*statelist[ns,i]*statelist[ns,j]
#                 ltc[ns] += -2*0.5*n2[i,j]*statelist[ns,i]*statelist[ns,j]
#                 for k in range(n):
#                     ltc[ns] += n1[i]*n2[j,k]*statelist[ns,i]*statelist[ns,j]*statelist[ns,k]
#                     ltc[ns] += n2[i,j]*n1[k]*statelist[ns,i]*statelist[ns,j]*statelist[ns,k]

#                     ltc[ns] += -2*0.5*n3[i,j,k]*statelist[ns,i]*statelist[ns,j]*statelist[ns,k]
#                     for q in range(n):
#                        ltc[ns] += n2[i,j]*n2[k,q]*statelist[ns,i]*statelist[ns,j]*statelist[ns,k]*statelist[ns,q]
#                         ltc[ns] += n1[i]*n3[j,k,q]*statelist[ns,i]*statelist[ns,j]*statelist[ns,k]*statelist[ns,q]
#                         ltc[ns] += n3[i,j,k]*n1[q]*statelist[ns,i]*statelist[ns,j]*statelist[ns,k]*statelist[ns,q]
#                         for m in range(n):
#                             ltc[ns] += n2[i,j]*n3[k,q,m]*statelist[ns,i]*statelist[ns,j]*statelist[ns,k]*statelist[ns,q]*statelist[ns,m]
#                             ltc[ns] += n3[i,j,k]*n2[q,m]*statelist[ns,i]*statelist[ns,j]*statelist[ns,k]*statelist[ns,q]*statelist[ns,m]
#                             for l in range(n):
#                                 ltc[ns] += n3[i,j,k]*n3[q,m,l]*statelist[ns,i]*statelist[ns,j]*statelist[ns,k]*statelist[ns,q]*statelist[ns,m]*statelist[ns,l]
#         n1_1 = n1 * statelist[ns]
#         n2_1 = n2 * np.einsum('i,j->ij',statelist[ns],statelist[ns])
#         n3_1 = n3 * np.einsum('i,j,k->ijk',statelist[ns],statelist[ns],statelist[ns])

#         temp = [n1_1,n2_1,n3_1]
#         for i in range(3):
#             for j in range(3):
#                 ltc[ns] += np.sum(temp[i])*np.sum(temp[j])
#             ltc[ns] += - np.sum(temp[i])
#         ltc[ns] += 0.25
#     return 4*ltc

def long_time(statelist,n,n1,n2,n3):
    no_states = len(statelist)
    ltc = np.zeros(no_states)
    for ns in range(no_states):
        n1_1 = n1 * statelist[ns]
        n2_1 = n2 * np.einsum('i,j->ij',statelist[ns],statelist[ns])
        n3_1 = n3 * np.einsum('i,j,k->ijk',statelist[ns],statelist[ns],statelist[ns])

        temp = [n1_1,n2_1,n3_1]
        for i in range(3):
            for j in range(3):
                ltc[ns] += np.sum(temp[i])*np.sum(temp[j])
            ltc[ns] += - np.sum(temp[i])
        ltc[ns] += 0.25
    return 4*ltc

#==============================================================================
# Run program
#==============================================================================

if __name__ == '__main__': 

    #startTime = datetime.now()
    #print('Start time: ', startTime)


    for d in dis:
        for p in range(1,reps+1):
            try:
                #print('proc/%s/tflow-d%.2f-O4-x%.2f-Jz%.2f-p%s.h5' %(nvar,d,x,delta,p))
                with h5py.File('%s/tflow-d%.2f-O%s-x%.2f-Jz%.2f-p%s.h5' %(nvar,d,order,x,delta,p),'r') as hf:
                    H2 = np.array(hf.get('H2_diag'))
                    maxV = np.max(np.abs(H2 -np.diag(np.diag(H2))))
                    # Error comparison with ED (for small systems only)
                    try:
                        err_med = np.median(np.array(hf.get('err')))
                        err_mean = np.mean(np.array(hf.get('err')))
                    except:
                        err_med,err_mean = 0.,0.
                    # Dynamics
                    itc =  4*np.array(hf.get('itc'))
                    itc0 = 4*np.array(hf.get('itc'))
                    if len(itc)>151:
                        itc = np.mean(itc,axis=0)
                        itc0 = np.mean(itc0,axis=0)
                    itc2 = 4*np.mean(np.array(hf.get('itc_nonint')),axis=0)
                    try:
                        edlist = 4*np.array(hf.get('ed_itc'))
                    except:
                        edlist = 0.
                    # Truncation error
                    try:
                        e1,e2,e3, e4 = np.array(hf.get('trunc_err'))
                        dl = np.array(hf.get('dl_list'))
                        trunc_err1 = e3/dl[-1]
                        trunc_err2 = e3
                    except:
                        e1,e2,e3,e4 = -1,-1,-1,-1
                        trunc_err1 = -10
                        trunc_err2 = -10
                    # LIOMs
                    c1 = np.array(hf.get('liom1_fwd'))
                    #print(c1)
                    c3 = np.array(hf.get('liom3_fwd'))
                    #print('normsum',np.sum([i**2 for i in c1]),np.sum([i**2 for i in c3]))
                    Hint = np.array(hf.get('Hint'))
                    dl = np.array(hf.get('dl_list'))

                #trunc_err1 = e4/dl[-1]
                #trunc_err2 = e3

                # Compute LIOM interaction terms from Hamiltonian
                
                HF = np.zeros((n,n))
                for i in range(n):
                    for j in range(n):
                        HF[i,j] = Hint[i,i,j,j]-Hint[i,j,j,i]
                if dim == 1:
                    felist = np.zeros(n-1)
                    for i in range(1,n):
                        felist[i-1] = np.median(np.abs(np.diag(HF,i)))

                elif dim == 2:
                    L = int(np.sqrt(n))
                    felist = np.zeros(2*L-2)
                    test = np.array([i for i in range(n)])
                    test = test.reshape((L,L))
                    for dist0 in range(2*L-2):
                        rlist = []
                        for i in range(n):
                            for j in range(i):
                                if i != j:
                                    loc1 = np.where(test == i)
                                    loc2 = np.where(test == j)
                                    dist = np.abs(loc1[0]-loc2[0]) + np.abs(loc1[1]-loc2[1])
                                    if dist == dist0:
                                        rlist += [np.abs(HF[i,j])]
                                    #felist[dist] += HF[i,j]
                        if len(rlist)>0:
                            felist[dist0] = np.median(rlist)

                # LIOM support
                if dim == 1:
                    frlist = np.zeros(n//2)
                    norm = np.sum([i**2 for i in c1]) + np.sum([i**2 for i in c3])
                    test = np.zeros(n)
                    for r in range(n//2):
                        for i in range(n):
                            if np.abs(i-n//2)<r:
                                test[i] = 1.0
                        test3 = np.einsum('i,j,k->ijk',test,test,test)
                        frlist2 = np.sum((test*c1)**2)+np.sum((test3*c3)**2)
                        frlist[r] += 1-frlist2/norm
                else:
                    L = int(np.sqrt(n))
                    frlist = np.zeros(2*L-1)
                    norm = np.sum([i**2 for i in c1]) + np.sum([i**2 for i in c3])
                    #if n%2 == 1:
                    centre = L//2
                   # else:
                   #     centre = L//2 + int(np.sqrt(L))//2
                    for r in range(2*L-1):
                        #print('***********',r)
                        test = np.zeros((L,L))
                        for i in range(L):
                            for j in range(L):
                                #if np.sqrt(np.abs(i-centre)**2+np.abs(j-centre)**2)<r:
                                if (np.abs(i-centre)+np.abs(j-centre)) < r:
                                    test[i,j] = 1.0
                        #print(test)
                        test = test.reshape(n)
                        #print(test)
                        test3 = np.einsum('i,j,k->ijk',test,test,test)
                        tc1 = (test*c1)
                        tc3 = (test3*c3)
                        frlist2 = np.sum([i**2 for i in tc1])+np.sum([i**2 for i in tc3])
                        #print(frlist2,norm,frlist2/norm)
                        frlist[r] += 1-frlist2/norm
                        if frlist[r] < 0:
                            print('norm error',frlist2,norm)
                    #print(frlist)

                # Complexity measures
                epsilon = 1e-6
                l3 = c3[c3 > epsilon]
                l1 = c1[c1 > epsilon]

                sparsity = (len(l1)+len(l3))/(n+n**3)

                l1 = np.sort(l1)[::-1]
                l3 = np.sort(l3)[::-1]
                l1 = [l1[i]*i for i in range(len(l1))]
                l3 = [l3[i]*i for i in range(len(l3))]

                weight1 = np.mean(l1)
                weight3 = np.mean(l3)

                # Rescale C(t) by minimising error w.r.t. non-interacting system
                points = 51
                errlist = np.zeros(points)
                pc = 0
                xlist = np.linspace(0,1,points,endpoint=True)
                for x1 in xlist:
                    test = np.array(copy.deepcopy(itc0))
                    #print(len(test),len(itc2))
                    test += x1*(1-test[0])
                    test *= 1/test[0]
                    err = np.mean([np.abs((test[i]-itc2[i])/itc2[i]) for i in range(75)])
                    errlist[pc] = err
                    pc += 1
                x0 = np.where(errlist == errlist.min())
                itc += xlist[x0]*(1-itc[0])
                itc *= 1/itc[0]

                # Dynamics - long time average (computed directly without time evolution)
                n1,n2,n3 = avg_n(c1,c3,n)
                if n <= 12:
                    basis = spinless_fermion_basis_1d(n,Nf=n//2)
                    no_states = min(basis.Ns,256)
                else:
                    no_states = 256
                print('No. states: ',no_states)
                statelist = np.zeros((no_states,n))
                for ns in range(no_states):
                    flag = False
                    while flag == False:
                        state = np.array(utility.nstate(n,'random_half'))
                        if not any((state == x).all() for x in np.array(statelist)):
                            statelist[ns] = state
                            flag = True
                ltc = long_time(statelist,n,n1,n2,n3)
                ltc = np.mean(ltc)
                ltc2 = ltc + xlist[x0]*(1-itc0[0])
                itc_0 = itc[0] + xlist[x0]*(1-itc[0])
                ltc2 *= 1/itc_0

                print('LONG TIME AVERAGE',ltc,ltc2,np.mean(itc[-50::]),xlist[x0][0],itc0[0])
        
                with h5py.File('%s/tflow-d%.2f-O%s-x%.2f-Jz%.2f-p%s.h5' %(nvar2,d,order,x,delta,p),'w') as hf:
                    hf.create_dataset('err_med',data=[err_med])
                    hf.create_dataset('err_mean',data=[err_mean])
                    hf.create_dataset('maxV',data=[maxV])
                    hf.create_dataset('trunc_err',data=[trunc_err1,trunc_err2])

                    hf.create_dataset('LIOM_int',data=felist)

                    hf.create_dataset('LIOM_sup',data=frlist)
                    hf.create_dataset('c1',data=c1)
                    hf.create_dataset('c3',data=c3)

                    hf.create_dataset('complexity',data=np.array([sparsity,weight1,weight3]))

                    hf.create_dataset('itc',data=itc)
                    hf.create_dataset('itc_nonint',data=itc2)
                    hf.create_dataset('ed_itc',data=edlist)
                    hf.create_dataset('rescale_params',data=np.array([xlist[x0][0],1/itc[0]]))
                    hf.create_dataset('ltc',data=ltc)
                    hf.create_dataset('ltc2',data=ltc2)
            except:
                None
