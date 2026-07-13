import matplotlib.pyplot as plt
from scipy.linalg import inv, eigh
import numpy as np
import itertools as its

class lattice(object):
  '''
  Lattice tight-binding class
  '''
  def __init__(self, dim, basis, nxyz, natom, atomvec):
    '''
    Note: the basis, nxyz and atomvec always have dimension 3
          with [0,0,0] for redundant dimension.
    dim : int, dimension of the system.
    basis : 3x3 array, real space basis in cartesian coordinate (unit cell vectors).
    nxyz: 3x1 array,  number of cells in x, y, and z direction.
    natom: int, number of atom per cell.
    atomvecs: natom*3 array, the coordinate of the atoms in cell in cartesian coordinate.
    '''
    self.dim = dim # dimension of the system
    self.basis = np.zeros((3,3)) # initialize unit cell basis in real-space coordinate
    self.nxyz = np.zeros((3),dtype=int) # number of mesh for each dimension
    # store unitcell basis as [[a1],[a2],[a3]]
    for d in range(3):
      self.basis[d,:] = basis[d]
      self.nxyz[d] = nxyz[d]
    self.natom = natom # number of atom
    # store atoms' vector in real-space coordinate as
    # [[atom1vec],[atom2vec],...[atomNvec]]
    self.atomvec = np.zeros((natom,3))
    for a in range(natom):
      self.atomvec[a] = atomvec[a]
    # construct k-reciprocal vectors a list of 3x3 matrices that contains k-vectors in
    # Brillouine zone.
    # reciprocal_space is the Brillouin Zone vectors
    # stored as [[k1],[k2],[k3]]
    # reciprocal_basis is the mesh of the Billouin Zone
    # stored as [[dk1],[dk2],[dk3]]
    #print self.basis
    if dim == 3:
      self.reciprocal_basis = 2.*np.pi*inv(self.basis)
      self.reciprocal_space = np.copy(self.reciprocal_basis)
      self.reciprocal_basis[0]/=self.nxyz[0]
      self.reciprocal_basis[1]/=self.nxyz[1]
      self.reciprocal_basis[2]/=self.nxyz[2]
    elif dim == 2:
      self.reciprocal_basis = np.zeros((3,3))
      self.reciprocal_basis[:2,:2] = 2.*np.pi*inv(self.basis[:2,:2].T)
      self.reciprocal_space = np.copy(self.reciprocal_basis)
      self.reciprocal_basis[0]/=self.nxyz[0]
      self.reciprocal_basis[1]/=self.nxyz[1]
    elif dim == 1:
      self.reciprocal_basis = np.zeros((3,3))
      self.reciprocal_basis[:1,:1] = 2*np.pi*inv(self.basis[:1,:1].T)
      self.reciprocal_space = np.copy(self.reciprocal_basis)
      self.reciprocal_basis[0]/=self.nxyz[0]

    #print self.reciprocal_basis
    #print self.nxyz
    # build reciprocal klist, real space rlist, and the lookup table
    self.kdict = {} # look up table contains position index and real space coordinate, [i,j,k,atom,rcoordinate]
    self.rdict = {} # look up table contains k-space index and k-space coordinate, [i,j,k,atom,kcoordinate]
    for i in range(self.nxyz[0]):
      for j in range(self.nxyz[1]):
        for k in range(self.nxyz[2]):
          for a in range(self.natom):
            idx = i*self.nxyz[1]*self.nxyz[2]*self.natom + j*self.nxyz[2]*self.natom + k*self.natom + a
            self.rdict[idx] = [ i, j, k, a, self.basis[0,:]*i+self.basis[1,:]*j+self.basis[2,:]*k+self.atomvec[a,:]]
            k_atom = np.dot(self.reciprocal_space,self.atomvec[a,:])
            self.kdict[idx] = [ i, j, k, a, self.reciprocal_basis[0,:]*i+self.reciprocal_basis[1,:]*j+self.reciprocal_basis[2,:]*k + k_atom]
            #print 'real', i, j, k, a, np.dot(self.basis.T,(i,j,k)) + self.atomvec[a,:]
            #print 'kspace', i, j, k, a, self.reciprocal_basis[0,:]*i+self.reciprocal_basis[1,:]*j+self.reciprocal_basis[2,:]*k
            #self.klist.append(np.dot(self.reciprocal_basis,(i,j,k)))

    # initial real coordinate hopping matrix
    self.hop_ij = None
    # initial k-space hopping matrix
    self.hop_k = None
  
  def print_lattice_info(self):
    print("dim=",self.dim)
    print("basis=")
    for d in range(3):
      print(self.basis[d])
    print("reciprocal space")
    for d in range(3):
      print(self.reciprocal_space[d])
    print("reciprocal basis")
    for d in range(3):
      print(self.reciprocal_basis[d])
    print("nxyz=",self.nxyz)
    print("natom=",self.natom)
    print("atomvec=")
    for a in range(self.natom):
      print(self.atomvec[a])

  def plot_lattice(self):
    #if self.dim == 1:
    #  for i in range(self.nxyz[0]):
    #    plt.plot(i*self.basis[0],0,'ro',ms=2)
    #  plt.show()
    if self.dim == 2:
      print(self.nxyz*self.basis)
      xmax = max(self.nxyz*self.basis[:,0])#(self.nxyz[0]*self.basis[0][0])
      xmin = min(self.nxyz*self.basis[:,0])#0.
      ymax = max(self.nxyz*self.basis[:,1])#(self.nxyz[1]*self.basis[1][1])
      ymin = min(self.nxyz*self.basis[:,1])#0.
      print('xmax=',xmax, ' xmin=', xmin, ' ymax=',ymax, ' ymin=',ymin)
      for i,j in its.product(list(range(self.nxyz[0])),list(range(self.nxyz[1]))):
        for a in range(self.natom):
          #print i, j, a
          coord = (i*self.basis[0])+(j*self.basis[1])+self.atomvec[a]
          #plt.plot(coord[0]%xmax,coord[1]%ymax,'ro',ms=5)
          plt.plot(coord[0],coord[1],'ro',ms=4)
      plt.axis('equal')
      #plt.xlim(-4,4)
      #plt.ylim(-4,4)
      plt.show()
    if self.dim == 3:
      raise Exception('3D plotting not implemented!')

  def build_mat_TB_PBC(self, hops, mu=0, antiperiodic=False):
    '''
    Build general TB model
    matrix ordered as 
    (0,0,0) (1,0,0) ... (nx,0,0) (nx,1,0) ... (nx,ny,0) (nx,ny,1) ... (nx,ny,nz)
    each () block contain number Natom of atoms

    input: hop, hopping dictionary. key: atom start, elements [atom end, dx, dy, dz, hopping amplitude]
    '''
    Nx, Ny, Nz, Natom = self.nxyz[0], self.nxyz[1], self.nxyz[2], self.natom
    N = Nx*Ny*Nz*Natom
    #print(N)
    mat_TB = np.zeros((N,N),dtype=complex)
    mat_TB += (mu)*np.eye(N,dtype=complex)

    for i in range(Nx):
      for j in range(Ny):
        for k in range(Nz):
          for a in range(Natom):
            #print i, j, k, a, i*Ny*Nz*Natom + j*Nz*Natom + k*Natom + a
            # lookup index
            ii = i*Ny*Nz*Natom + j*Nz*Natom + k*Natom + a
            for hop in hops[a]:
#            for h in range(len(hops[a])):
#              hop = hops[a][h]
              jj = (i+hop[1])%Nx*Ny*Nz*Natom + (j+hop[2])%Ny*Nz*Natom + (k+hop[3])%Nz*Natom + hop[0]
              if not antiperiodic:
                mat_TB[ii,jj] += hop[4]
              if antiperiodic:
                if (i+hop[1])>=Nx or (j+hop[2])>=Ny or (k+hop[3])>=Nz or (i+hop[1])<0 or (j+hop[2])<0 or (k+hop[3])<0:
#                  print i,j,k,'to',i+hop[1],j+hop[2],k+hop[3],'(',(i+hop[1])%Nx,(j+hop[2])%Ny,(k+hop[3])%Nz,')','with t=1'
                  mat_TB[ii,jj] += -hop[4]
                else:
                  mat_TB[ii,jj] += hop[4]

    self.hop_ij = mat_TB

    return self.hop_ij

  def loc_basis_trans_hop_ij(self, rot):
    '''
    Perform local basis transformation
    Input: 
      rot: np.array  
    '''
    Nx, Ny, Nz, Natom = self.nxyz[0], self.nxyz[1], self.nxyz[2], self.natom
    N = Nx*Ny*Nz
    rot_full = np.kron(np.eye(N), rot)
    self.hop_ij = np.dot(rot_full.conj().T, np.dot(self.hop_ij, rot_full))
    return self.hop_ij

  def FT_hop_ij(self):
    '''
    Fourier transform real-space hopping matrix to k-space
    '''
    Nx, Ny, Nz, Natom = self.nxyz[0], self.nxyz[1], self.nxyz[2], self.natom
    N = Nx*Ny*Nz*Natom
    #build Fourier transform matrix
    Urk = np.zeros((N//Natom,N//Natom),dtype=complex)
    hop_k = np.zeros((N,N),dtype=complex)

    for idr in np.arange(0,N,Natom):
      for idk in np.arange(0,N,Natom):
#        for ia in range(Natom):
          Urk[idr//Natom,idk//Natom] = np.exp(-1j*np.dot(self.rdict[idr][4],self.kdict[idk][4]))
    Urk = np.kron(Urk,np.eye(Natom)) 
    Urk/=np.sqrt(N//Natom)
    #print 'Urk='
    #print Urk
    #assert(np.dot(Urk.conj().T,Urk), np.eye(Urk.shape[0]))

    self.hop_k = np.dot(Urk.conj().T, np.dot(self.hop_ij, Urk))

    return self.hop_k

  def get_dos(self, oms, eta):
    '''
    get density of state
    oms: frequency
    eta: brodening
    '''
    evals, evecs = eigh(self.hop_ij)
    Ge = np.array([[1./((om+1j*eta)-e) for om in oms] for e in evals])
    G = np.sum(Ge, axis=0)/len(evals)
    dos = -G[:].imag/np.pi
    return dos

if __name__ == '__main__':
  test = '2Dsquare'
  #test = '2Dtriangular'
  #test = '2Dgraphene'
  #test = '3Dcubic'

  # square
  if test == '2Dsquare':
    # lattice(dimension,3D unit-cell vector, number of k-points in three directions, number of atoms, atom's Cartesian coordinates)
    lat = lattice(2,[[1,0,0],[0,1,0],[0,0,0]],[50,50,1],1,[[0,0,0]])
#    lat.plot_lattice()
    # hopping dictionary {atom index:, [[atom index, (cell xindex, cell yindex, cell zindex), hopping]]}
    t = -1. # hopping amplitude
    hop = {}
    hop[0] = [[0,1,0,0,t],
              [0,-1,0,0,t],
              [0,0,1,0,t],
              [0,0,-1,0,t]]
    hop_mat = lat.build_mat_TB_PBC(hop,antiperiodic=False)
    evals, evecs = eigh(hop_mat)
 
    #Fourier to k-space
    #hop_k = lat.FT_hop_ij()
  
    #test non-interacting DOS
    oms = np.linspace(-12.0,12.0,501)
    import matplotlib.pyplot as plt
    dos = lat.get_dos(oms, 0.15)
    plt.plot(oms, dos)
#    plt.show()
  
    #print(hop_mat.real)
    #print(eigh(hop_mat)[0])
    from scipy.integrate import quad
    from scipy.interpolate import interp1d

    fintr = interp1d(oms,dos,fill_value="extrapolate")
    print(quad(fintr, -12, 12))

  # triangular
  if test == '2Dtriangular':

    # lattice(dimension,3D unit-cell vector, number of k-points in three directions, number of atoms, atom's Cartesian coordinates)
    lat = lattice(2,[[1,0,0],[0.5,1,0],[0,0,0]],[50,50,1],1,[[0,0,0]])
    lat.print_lattice_info()
#    lat.plot_lattice()

    # hopping dictionary {atom index:, [[atom index, (cell xindex, cell yindex, cell zindex), hopping]]}
    t = -1.0
    hop = {}
    hop[0] = [[0,1,0,0,t],
              [0,0,1,0,t],
              [0,-1,1,0,t],
              [0,-1,0,0,t],
              [0,0,-1,0,t],
              [0,1,-1,0,t]]
    hop_mat = lat.build_mat_TB_PBC(hop,antiperiodic=False)
    evals, evecs = eigh(hop_mat)
 
    #Fourier to k-space
    #hop_k = lat.FT_hop_ij()
  
    #test non-interacting DOS
    oms = np.linspace(-12.0,12.0,501)
    import matplotlib.pyplot as plt
    dos = lat.get_dos(oms, 0.15)
    plt.plot(oms, dos)
#    plt.show()

  # graphene
  if test == '2Dgraphene':
    # lattice(dimension,3D unit-cell vector, number of k-points in three directions, number of atoms, atom's Cartesian coordinates)
    lat = lattice(2,[[3./2.,np.sqrt(3)/2,0],[0,np.sqrt(3),0],[0,0,0.]],[25,25,1],2,[[0,0,0],[0.5,np.sqrt(3)/2,0]])
    lat.print_lattice_info()
#    lat.plot_lattice()

    # hopping dictionary {atom index:, [[atom index, (cell xindex, cell yindex, cell zindex), hopping]]}
    t1 = 1.
    hop = {} #key: atom0, element: hopping matrix [atom1,dx,dy,dz,t]
    hop[0] = [[1,0,0,0,t1],
              [1,-1,0,0,t1],
              [1,0,-1,0,t1]]
    hop[1] = [[0,0,0,0,t1],
              [0,1,0,0,t1],
              [0,0,1,0,t1]]
    hop_mat = lat.build_mat_TB_PBC(hop,antiperiodic=False)
    evals, evecs = eigh(hop_mat)

    #Fourier to k-space
    #hop_k = lat.FT_hop_ij()
  
    #test non-interacting DOS
    oms = np.linspace(-12.0,12.0,501)
    import matplotlib.pyplot as plt
    dos = lat.get_dos(oms, 0.05)
    plt.plot(oms, dos)
#    plt.show()

  if test == '3Dcubic':
    # lattice(dimension,3D unit-cell vector, number of k-points in three directions, number of atoms, atom's Cartesian coordinates) 
    lat = lattice(3,[[1,0,0],[0,1,0],[0,0,1]],[16,16,16],1,[[0,0,0]])
    #lat.plot_lattice()

    # hopping dictionary {atom index:, [[atom index, (cell xindex, cell yindex, cell zindex), hopping]]}
    t = -1.
    hop = {}
    hop[0] = [[0,1,0,0,t],
              [0,-1,0,0,t],
              [0,0,1,0,t],
              [0,0,-1,0,t],
              [0,0,0,1,t],
              [0,0,0,-1,t]]
    hop_mat = lat.build_mat_TB_PBC(hop,antiperiodic=False)
    evals, evecs = eigh(hop_mat)

    #Fourier to k-space
    #hop_k = lat.FT_hop_ij()
  
    #test non-interacting DOS
    oms = np.linspace(-12,12,501)
    import matplotlib.pyplot as plt
    dos = lat.get_dos(oms, 0.2)
    plt.plot(oms, dos)
#    plt.show()
