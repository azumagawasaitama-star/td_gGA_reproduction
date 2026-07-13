"""
Convenience routines for matrix operations (function, derivative).

Author: Tsung-Han Lee (2017), henhans74716@gmail.com
        Nicola Lanata (2017), Aarhus
"""
import numpy as np
import itertools as it
from scipy.linalg import eigh


lcomplex = False
if lcomplex:
  dtype_params = complex
else:
  dtype_params = float


# Fermi function at T=0
def calc_Fermi0(x,T=0.0):
    if T > 0.0:
        return 1.0 / (1.0 + np.exp(x / T))
    else:
        return np.where(x < 0.0, 1.0, 0.0)


# Generic function of an Hermitian matrix
def funcMat(H, function, T=0.0, pr=False):
    N = H.shape[0]
    assert(H.shape[0] == H.shape[1])
    if not np.allclose(H.conj().T,H):
      print (H)
    assert(np.allclose(H.conj().T,H))

    eigenvalues,U = eigh(H)
    Udagger = U.conj().T

    if T>0:
        functioneigenvalues = function(eigenvalues,T=T)
    else:
        functioneigenvalues = function(eigenvalues)
    if pr:
        print (functioneigenvalues)
    functionE = np.zeros(U.shape, dtype=dtype_params)
    functionE[np.arange(N), np.arange(N)] = np.real(functioneigenvalues)
    functionH =  np.dot(np.dot(U,functionE),Udagger)

    return functionH






def generate_orthonormal_basis(N):
    """
    Generate an orthonormal basis for the space of NxN symmetric matrices
    with respect to the Frobenius inner product.

    Parameters:
    N (int): The dimension of the space.

    Returns:
    list: The list of orthonormal basis matrices.
    """
    # Initialize basis
    basis = []
   


    for i in range(N):
        M = np.zeros((N, N))
        M[i, i] = 1
        basis.append(M)
 
    # Create basis matrices
    for i in range(N):
        for j in range(i+1, N):
            M = np.zeros((N, N))
            M[i, j] = M[j, i] = 1
            if i != j:
                M /= np.sqrt(2)
            basis.append(M)

    return basis






# Create canonical basis Hermitian NxN matrices (and its transpose)
def Hermitian_list(N):
    H_list=[]
    tH_list=[]

    Z=np.zeros((N,N),dtype=dtype_params)
    for i in range(N):
        H=Z*1.0
        H[i,i]=1.0
        H_list.append(H*1.0)
        tH_list.append(H*1.0)
    
    for i in range(N):
        for j in range(i+1,N):
            H=Z*1.0
            H[i,j]=1.0
            H[j,i]=1.0
            H_list.append(H/np.sqrt(2.0))
            tH_list.append(H/np.sqrt(2.0))
            H=Z*1.0
            H[i,j]=1.0j
            H[j,i]=-1.0j
            H_list.append(H/np.sqrt(2.0))
            tH_list.append(np.transpose(H)/np.sqrt(2.0))
    
    assert(len(H_list)==N**2)
    return H_list, tH_list



# Given REAL array x and list of matrices H_list, construct
# linear combination:  H = \sum_n x_n [H_list]_n
def realHcombination(x,H_list):
    M=len(x)
    N=H_list[0].shape[0]
    assert(H_list[0].shape[0]==H_list[0].shape[1])
    assert(len(x)==len(H_list))

    H=np.zeros((N,N), dtype=dtype_params)
    for i in range(M):
        H = H + x[i] * H_list[i]

    return H


# Given Hermitian matrix H extract components x_n with respect
# to a set H_list of Hermitian matrices
def inverse_realHcombination(H,H_list):
    N=H_list[0].shape[0]
    #H_list,tH_list=Hermitian_list(N)
    M=len(H_list)
    assert(H.shape[0]==H.shape[1])
    #assert(M==N**2)
    #
    x_list=[]
    for i in range(M):
        x_list.append(np.trace(np.dot(H_list[i],H)))
    x=np.array(x_list)
    #
    return x


# Given REAL array v=[x,y] and list of matrices H_list, construct
# linear combination:  R = \sum_n (x_n + i y_n) [H_list]_n
def complexHcombination(v,H_list):
    twiceM=len(v)
    M=twiceM//2
    N=H_list[0].shape[0]
    assert(twiceM % 2 == 0) # Checking that dimension of v is even
    assert(H_list[0].shape[0]==H_list[0].shape[1])
    assert(len(v)==2*len(H_list))
    
    x=v[0:M]
    y=v[M:twiceM]
    H=np.zeros((N,N), dtype=dtype_params)
    for i in range(M):
        H = H + x[i] * H_list[i]
    for i in range(M):
        H = H + 1j*y[i] * H_list[i]
    
    return H


# Given complex matrix H extract components x_n with respect
# to a set H_list of Hermitian matrices
def inverse_complexHcombination(H,H_list):
    N=H_list[0].shape[0]
    H_list,tH_list=Hermitian_list(N)
    M=len(H_list)
    assert(H.shape[0]==H.shape[1])
    assert(M==N**2)
    #
    xr_list=[]
    xi_list=[]
    for i in range(M):
        xr_list.append( np.real( np.trace(np.dot(H_list[i],H) ) ) )
        xi_list.append( np.imag( np.trace(np.dot(H_list[i],H) ) ) )
    xr=np.array(xr_list)
    xi=np.array(xi_list)

    return np.hstack((xr,xi))





# Given Hermitian matrix H extract following blocks,
# where S is sxs, and H is NxN
# -------------------------------------
# |         |                         | 
# |    S    |            V            | 
# |         |                         | 
# -------------------------------------
# |         |                         | 
# |         |                         | 
# |         |                         | 
# |  V^+    |            B            | 
# |         |                         | 
# |         |                         | 
# |         |                         | 
# ------------------------------------|
def get_blocks(H, s):
    N=H.shape[0]
    assert(H.shape[0]==H.shape[1])
    assert(s<=N)
    #
    S=H[0:s,0:s]
    B=H[s:N,s:N]
    V=H[0:s,s:N]
    Vdagger=H[s:N,0:s]
    #
    return S,B,V,Vdagger



def dF(A, H, function, d_function):
    """
    Evaluate matrix derivative using Loewner Theorem. Df(A)(H) = d/dt|_t=0 f(A+tH) = f^[1](A) o H
    """
    evals, evecs = eigh(A)
    Hbar = np.dot(np.conj(evecs).T, np.dot( H, evecs) ) #transform H to A's basis

    #create Loewner matrix in A's basis
    loewm = np.zeros(evecs.shape,dtype=dtype_params)
    for i in range(loewm.shape[0]):
        for j in range(loewm.shape[1]):
            if i==j:
                loewm[i,i] = np.real(d_function(evals[i]))# derivative(function, evals[i], dx=1e-12)
            if i!=j:
                if evals[i] != evals[j]:
                    loewm[i,j]= ( function(evals[i]) - function(evals[j]) )/(evals[i]-evals[j])
                else:
                    loewm[i,j] = np.real(d_function(evals[i]))# derivative(function, evals[i], dx=1e-12)

    # Perform the Schur product in A's basis then transform back to original basis.
    deriv = np.dot(evecs, np.dot( loewm*Hbar, np.conj(evecs).T ) )
    return deriv


def dF_numerical(A, H, function):
    """
    Evaluate matrix derivative numericaly element by element
    """
    t = 1e-10
    AptH = A + t * H
    
    evals, evecs = eigh(A)
    diag_A = np.dot( np.conj(evecs).T , np.dot( A, evecs ) )
    #print diag_A
    for i in range(diag_A.shape[0]):
        diag_A[i][i] = function( diag_A[i][i] )
    #print diag_A
    fA = np.dot( evecs, np.dot( diag_A , np.conj(evecs).T ) )
    
    evals, evecs = eigh(AptH)
    diag_AptH = np.dot( np.conj(evecs).T , np.dot( AptH, evecs ) )
    #print diag_AptH
    for i in range(diag_AptH.shape[0]):
        diag_AptH[i][i] = function( diag_AptH[i][i] )    
    #print diag_AptH
    fAptH = np.dot( evecs, np.dot( diag_AptH , np.conj(evecs).T ) )
    
    deriv = ( fAptH - fA )/t
    return deriv


#################################################


# Compute density matrix
def calc_C(H,T=0.0):
    N = H.shape[0]
    assert(H.shape[0] == H.shape[1])

    C = funcMat(H, calc_Fermi0,T)

    return C


# [D(1-D)]^(0.5)
def denRm1(x):
    x_safe = np.clip(np.real(x), 1e-14, 1.0 - 1e-14)
    return (x_safe*(1.0-x_safe))**(0.5)


#  [D(1-D)]^(-0.5)
def denR(x):
    x_safe = np.clip(np.real(x), 1e-14, 1.0 - 1e-14)
    return (x_safe*(1.0-x_safe))**(-0.5)


#d/dD [D(1-D)]^(0.5) = (0.5-D)*[D(1-D]^(-0.5)
def ddenRm1(x):
    x_safe = np.clip(np.real(x), 1e-14, 1.0 - 1e-14)
    return ((0.5-x_safe)/(x_safe*(1.0-x_safe))**0.5)


def duplicate_in_spin_space(A):
    """
    Take  matrix acting in single-particle spinless space and
    duplicate it to act in spin space, i.e construct :math:`\tilde{A}` such that:
    
    .. math::
               \tilde{A}_{2i, 2j} = \tilde{A}_{2i+1,2j+1} = A_{ij}
    
    Parameters
    ----------
    A : NxN matrix
    
    Returns
    -------
    At : 2Nx2N matrix

    See also
    --------
    spin_symmetrize
    """
    if "Gf" in str(type(A)):
        At = type(A)(mesh = A.mesh, shape = (A.target_shape[0]*2,A.target_shape[1]*2))
        for iw in range(len(At.mesh)):
            for i,j in it.product(list(range(A.target_shape[0])), list(range(A.target_shape[1]))):
                At.data[iw, 2*i,2*j] = A.data[iw,i,j]
                At.data[iw, 2*i+1,2*j+1] = A.data[iw,i,j]
    else:
        At = np.zeros((A.shape[0]*2,A.shape[1]*2), dtype=A.dtype)
        for i,j in it.product(list(range(A.shape[0])), list(range(A.shape[1]))):
            At[2*i,2*j]=A[i,j]
            At[2*i+1,2*j+1]=A[i,j]
    return At

def spin_symmetrize(A, tol=1e-12):
    """Symmetrize spin up and dn, i.e compute matrix :math:`A^\mathrm{sym}` such that

    .. math::

      A^\mathrm{sym}_{ij} = \\frac{1}{2} (A_{2i,2j} + A_{2i+1,2j+1})

    Parameters
    ------------
    A : matrix or GfImFreq
    
    See also
    --------
    duplicate_in_spin_space
    """     
    if "Gf" in str(type(A)):
        A_sym = type(A)(mesh = A.mesh, shape = (A.target_shape[0]/2, A.target_shape[1]/2))
        for i,j in it.product(list(range(A_sym.target_shape[0])), list(range(A_sym.target_shape[1]))):
            diff = A[2*i,2*j] - A[2*i+1,2*j+1]
            for iw in range(len(diff.mesh)):
                assert (abs(diff.data[iw,0,0])<tol), "symmetrizing matrix with spin differentiation for element %s, %s! : %s neq %s"%(i,j, A[2*i, 2*j], A[2*i+1, 2*j+1])
            A_sym[i,j] = 0.5*(A[2*i,2*j] + A[2*i+1,2*j+1])
    elif (type(A)==np.ndarray and A.ndim==3) or type(A)==list:
        A_sym = np.zeros((A.shape[0],A.shape[1]/2,A.shape[2]/2), dtype = A.dtype)
        for iw in range(len(A)):
            tmp = np.zeros((A[iw].shape[0]/2,A[iw].shape[1]/2), dtype = A.dtype)
            for i,j in it.product(list(range(tmp.shape[0])), list(range(tmp.shape[1]))):
                diff = A[iw,2*i,2*j] - A[iw,2*i+1,2*j+1]
                assert (abs(diff)<tol), "symmetrizing matrix with spin differentiation for element %s, %s! : %s neq %s"%(i,j, A[iw,2*i, 2*j], A[iw,2*i+1, 2*j+1])
                tmp[i,j] = 0.5*(A[iw,2*i,2*j] + A[iw,2*i+1,2*j+1])
            A_sym[iw,:,:] = tmp
    else:
        A_sym = np.zeros((A.shape[0]//2,A.shape[1]//2), dtype = A.dtype)
        for i,j in it.product(list(range(A_sym.shape[0])), list(range(A_sym.shape[1]))):
            assert (abs(A[2*i,2*j] - A[2*i+1,2*j+1])<tol), "symmetrizing matrix with spin differentiation for element %s, %s! : %s neq %s"%(i,j, A[2*i, 2*j], A[2*i+1, 2*j+1])
            A_sym[i,j] = 0.5*(A[2*i,2*j] + A[2*i+1,2*j+1])
    
    return A_sym




def calc_offdiag_dd(X, nqspo=3):
    def root(x):
       dd = cr.realHcombination(x,H_list_tmp)*np.sqrt(2)
    
       y = []
       for i in range(self.nqspo*(self.nqspo-1)//2):
         y.append(np.trace(dd.dot(Mc[i]))-K[i][0])
        
    
       return y
    
    
    K = []
    Mc = [] 
    # i*h = Antisymmetric matrix
    ihmat = [np.array(((0,-1,0),(1,0,0),(0,0,0))),
             np.array(((0,0,0),(0,0,-1),(0,1,0))),
             np.array(((0,0,-1),(0,0,0),(1,0,0)))]

    # Basis of Hermitian matrices
    H_list_tmp = cr.generate_orthonormal_basis(3)[3:]
    #H_list_tmp = self.H_list[3:]
    
    # Precompute intermediates M and K
    for i in range(nqspo*(nqspo-1)//2):
        # Compute K
        M = ihmat[i].T.dot(D)
        K.append(2.*fdaggerc[0].dot(M))
    
        # Compute Mc
        Mc.append(ihmat[i].dot(Lmbdac) - Lmbdac.dot(ihmat[i]))


    xinit=(0,0,0)
    # Solve for root of the equations
    solver = optimize.root(root,xinit,method='hybr')

    # Construct full <dd> matrix from coefficients
    offdd = cr.realHcombination(solver.x,H_list_tmp)*np.sqrt(2)

    return offdd 
