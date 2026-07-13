"""
Impurity solver code for the ghost-GA approximation based on exact diagonalization.

Author: Marius Frank (2023), Marius.Frank@phys.au.dk
"""
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse
from scipy.sparse.linalg import eigsh
import primme
import sys
import convenience_routines as cr
import itertools as it
np.set_printoptions(suppress=True)


lcomplex = False
if lcomplex:
  dtype_params = complex
else:
  dtype_params = float


class edSolver:
    """
    A class for solving an impurity problem based on exact diagonalization.

    Attributes
    ----------
    n_phys_orb : int
        Number of physical orbitals.
    n_ghost_orb : int
        Number of ghost orbitals.
    n_bath_orb : int
        Number of bath orbitals, which is the sum of physical and ghost orbitals.
    n_tot_orb : int
        Total number of orbitals, including both physical and bath orbitals.
    n_half : int
        Half of the total number of orbitals.
    hsize_full : int
        Full Hilbert space size, which is 2 to the power of the total number of orbitals.
    hsize_half : int
        Half-filled Hilbert space size, given by binomial coefficient of the total number of orbitals.
    impurity_nr : int
        Impurity number.
    impurity_type : str
        Impurity type ('s' or 'p').
    l1s : bool
        True if impurity type is 's', False otherwise.
    l2p : bool
        True if impurity type is 'p', False otherwise.
    spin_pen : float, optional, default: 10.0
        Spin penalty to obtain singlet state.
    tol : float, optional, default: 1e-4
        Tolerance for the error in the state of Hemb.

    Methods
    -------
    build_creation_ops():
        Create a list of fermionic creation operators.
    build_op_prods(FH_list):
        Create products of creation and annihilation operators and store them on disk.
    """

    def __init__(self, n_phys_orb, n_ghost_orb, impurity_nr, impurity_type, lspin_sym= True, spin_pen=10., tol=1e-4):
        """
        Initialize the edSolver class with given parameters.

        Parameters
        ----------
        n_phys_orb : int
            Number of physical orbitals.
        n_ghost_orb : int
            Number of ghost orbitals.
        impurity_nr : int
            Impurity number.
        impurity_type : str
            Impurity type ('s' or 'p').
        lspin_sym : bool, optional
            If True, assumes the system is spin-symmetric. Default is True.
        spin_pen : float, optional, default: 10.0
            Spin penalty to obtain singlet state.
        tol : float, optional, default: 1e-4
            Tolerance for the error in the state of Hemb.
        """
        self.n_phys_orb  = n_phys_orb
        self.n_ghost_orb = n_ghost_orb
        self.n_bath_orb  = n_phys_orb + n_ghost_orb
        self.n_tot_orb   = self.n_phys_orb + self.n_bath_orb
        self.n_half      = self.n_tot_orb//2 

        self.hsize_full = 2**self.n_tot_orb
        self.hsize_half = int(scipy.special.binom(self.n_tot_orb, self.n_half)) 

        self.impurity_nr = impurity_nr

        self.impurity_type = impurity_type
        self.l1s = impurity_type == "s" # s-orbital?
        self.l2p = impurity_type == "p" # p-orbitals?
        if not (self.l1s or self.l2p): raise ValueError(f"Unrecognized or not implemented impurity: {impurity_type}")

        # Spin penalty to obtain singlet state
        self.spin_pen = spin_pen
        # Tolerance for the error in the state of Hemb
        self.tol      = tol

        self.lspin_sym = lspin_sym

        # Create temporary list of fermionic creation operators
        FH_list = self.build_creation_ops()

        # Create products of creation and annihilitation operators and store them on disc
        self.build_op_prods(FH_list)



    def build_creation_ops(self):
        """
        Build a list of fermionic creation operators for the impurity problem.

        The method generates fermionic creation operators for each orbital in the total
        orbital space, considering all possible configurations. It returns a list of
        sparse matrices representing the creation operators.

        Returns
        -------
        list
            A list of csr_matrix objects representing fermionic creation operators.
        """
        strb = '{:0' + str(self.n_tot_orb) + 'b}'
        
        # Generate binary strings for all configurations
        config_b_list = [strb.format(iconf) for iconf in range(self.hsize_full)]
        
        # Filter binary strings by their occupation number
        filtered_config_b_list = [
            config_b for iocc in range(self.n_tot_orb + 1)
            for config_b in config_b_list
            if sum(int(b) for b in config_b) == iocc
        ]
        
        def find_matching_row(config_b, o):
            config_a = ''.join('1' if i == o else b for i, b in enumerate(config_b))
            return next(l for l, conf in enumerate(filtered_config_b_list) if conf == config_a)
    
        def calculate_exponent(config_b, o):
            return sum(int(b) for b in config_b[:o])
        
        def build_fh(o):
            rows, cols, data = zip(*(
                (find_matching_row(config_b, o), b, (-1.) ** calculate_exponent(config_b, o))
                for b, config_b in enumerate(filtered_config_b_list)
                if config_b[o] == '0'
            ))
            return csr_matrix((data, (rows, cols)), shape=(self.hsize_full, self.hsize_full), dtype=float)
        
        return [build_fh(o) for o in range(self.n_tot_orb)]


    
    def build_op_prods(self,FH_list):
        """
        Build and save operator products for the impurity problem (save only the block corresponding to half-filling)

        This method generates one- and two-body operators, as well as total spin operators,
        and saves them to disk as .npz files. The input is a list of fermionic creation
        operators obtained from the build_creation_ops method.

        Parameters
        ----------
        FH_list : list
            A list of csr_matrix objects representing fermionic creation operators.
        """

        # Get size of Fock corresponding to half-filliing and compute offsets
        ioff = sum(int(scipy.special.binom(self.n_tot_orb, i)) for i in range(self.n_half))
        iend = ioff + self.hsize_half


        # Helper routine for saving operator products to file
        def write_prod_1e(i_start, j_start, n_orb1, n_orb2, type_orb1, type_orb2, lreverse=False):
            if not all(check in ("phys", "bath")  for check in [type_orb1, type_orb2]): 
                raise ValueError("write_prod only accepts physical or bath orbitals")
       
            # Define prefix of file name based on orbital types, impurity number and type
            prefix = f"{type_orb1}-{type_orb2}_imp-{str(self.impurity_nr)}_{self.impurity_type}"

            for i in range(n_orb1):
                for j in range(n_orb2):
                    idx = i + i_start
                    jdx = j + j_start

                    if not lreverse: prod = FH_list[idx].dot( FH_list[jdx].getH() )[ioff:iend,ioff:iend]
                    if lreverse: prod = FH_list[jdx].getH().dot( FH_list[idx] )[ioff:iend,ioff:iend]
                    fname = f"{prefix}_op+{str(i)}-{str(j)}"
                    scipy.sparse.save_npz(fname,prod)
        
 
        # Build and save one-body operators (physical and bath)
        write_prod_1e(0, 0, self.n_phys_orb, self.n_phys_orb, 'phys', 'phys')
        write_prod_1e(self.n_phys_orb, 0, self.n_bath_orb, self.n_phys_orb, 'bath', 'phys')
        write_prod_1e(0, self.n_phys_orb, self.n_phys_orb, self.n_bath_orb, 'phys', 'bath')
        write_prod_1e(self.n_phys_orb, self.n_phys_orb, self.n_bath_orb, self.n_bath_orb, 'bath', 'bath', lreverse=True)

        # Build and save physical two-body operators
        prefix = f"phys-phys_V2E_imp-{str(self.impurity_nr)}_{self.impurity_type}"
        indices = range(self.n_phys_orb)
        for i, j, k, l in it.product(indices, repeat=4):               
            fname = f"{prefix}_op+{str(i)}-{str(j)}-{str(k)}-{str(l)}"
            prod = FH_list[i].dot( FH_list[j].dot( FH_list[l].getH().dot(FH_list[k].getH() ) ) )[ioff:iend,ioff:iend]
            scipy.sparse.save_npz(fname,prod)
   
        # Build total spin operators S+
        Sp = csr_matrix((self.hsize_half,self.hsize_half), dtype=dtype_params)
        for i in range(self.n_half): #no//2):
            Sp += FH_list[2*i].dot(FH_list[2*i+1].getH())[ioff:iend,ioff:iend]
        # Build total spin operators S-
        Sm = csr_matrix((self.hsize_half,self.hsize_half), dtype=dtype_params)
        for i in range(self.n_half): #no//2):
            Sm += FH_list[2*i+1].dot(FH_list[2*i].getH())[ioff:iend,ioff:iend]
        Sz = csr_matrix((self.hsize_half,self.hsize_half), dtype=dtype_params)
        for i in range(self.n_half): #no//2):
            Sz += ( 0.5*FH_list[2*i].dot(FH_list[2*i].getH())[ioff:iend,ioff:iend] 
                   - 0.5*FH_list[2*i+1].dot(FH_list[2*i+1].getH())[ioff:iend,ioff:iend] )

        # Build S^2 operator and save
        S2 = Sm.dot(Sp)+Sz.dot(Sz)+Sz
        scipy.sparse.save_npz("S2-imp"+str(self.impurity_nr) + self.impurity_type, S2)



    def set_parameters(self, D, H1, Lc, U):
        """
        Set and duplicate parameters for the embedding Hamiltonian in spin space.

        This method takes input parameters for the impurity problem, duplicates them
        in spin space, and generates a two-body interaction tensor in spin space.

        Parameters
        ----------
        D : numpy.ndarray
            Matrix representing the hybridization between the impurity and bath orbitals.
        H1 : numpy.ndarray
            One-body Hamiltonian matrix for the physical block of the embedding Hamiltonian.
        Lc : numpy.ndarray
            One-body Hamiltonian matrix for the bath block of the embedding Hamiltonian.
        U : float
            Two-body interaction strength.

        Returns
        -------
        tuple
            A tuple containing the duplicated parameters (D, H1E, LAMBDA) and the
            two-body interaction tensor (V2E) in spin space.
        """
        # Set up/duplicate parameters for embedding Hamitonian in spin space
        H1E    = cr.duplicate_in_spin_space(H1)
        D      = cr.duplicate_in_spin_space(D)
        LAMBDA = cr.duplicate_in_spin_space(Lc)

        V2E = np.zeros((self.n_phys_orb,self.n_phys_orb, self.n_phys_orb, self.n_phys_orb))
        for i,j,k,l in it.product(list(range(self.n_phys_orb//2)), list(range(self.n_phys_orb//2)), 
                                  list(range(self.n_phys_orb//2)), list(range(self.n_phys_orb//2))):
              V2E[2*i+1,2*j+1,2*k,2*l]= U
              V2E[2*i,2*j,2*k+1,2*l+1]= U

        return D, H1E, LAMBDA, V2E 




    def read_prod_1e(self, facMat, i_start, j_start, n_orb1, n_orb2, type_orb1, type_orb2, lexpectation=False):
        """
        Read one-electron operator products from files and apply the given factor matrix.

        This method reads one-electron operator products from files and computes their 
        linear combinations using the provided factor matrix. It can also compute the
        expectation values of the products if `lexpectation` is set to True.

        Parameters
        ----------
        facMat : numpy.ndarray
            Factor matrix used for the linear combination of one-electron operator products.
        i_start : int
            Start index for the first set of orbitals.
        j_start : int
            Start index for the second set of orbitals.
        n_orb1 : int
            Number of orbitals in the first set.
        n_orb2 : int
            Number of orbitals in the second set.
        type_orb1 : str
            Type of the first set of orbitals. Must be "phys" or "bath".
        type_orb2 : str
            Type of the second set of orbitals. Must be "phys" or "bath".
        lexpectation : bool, optional
            If True, computes the expectation values of the products. Default is False.

        Returns
        -------
        scipy.sparse.csr_matrix or numpy.ndarray
            The linear combination of one-electron operator products or the expectation
            values, depending on the value of `lexpectation`.
        """
        if not all(check in ("phys", "bath")  for check in [type_orb1, type_orb2]): 
            raise ValueError("write_prod only accepts physical or bath orbitals")
    
        outMat = csr_matrix((self.hsize_half,self.hsize_half), dtype=dtype_params) if not lexpectation else np.zeros((n_orb1,n_orb2))

        # Define prefix of file name based on orbital types, impurity number and type
        prefix = f"{type_orb1}-{type_orb2}_imp-{str(self.impurity_nr)}_{self.impurity_type}"

        for i in range(n_orb1):
            for j in range(n_orb2):
                idx = i + i_start
                jdx = j + j_start

                fname = f"{prefix}_op+{str(i)}-{str(j)}"
                prod = scipy.sparse.load_npz(fname+".npz")

                if not lexpectation:
                    outMat += facMat[i,j]*prod
                else:
                    outMat[i,j] = facMat.T.dot( prod.dot(facMat) )

        return outMat



    def build_Hemb(self, D, H1, Lc, U):
        """
        Build the impurity Hamiltonian (embedding Hamiltonian) for the system.

        This method constructs the impurity Hamiltonian by reading one- and two-electron
        operator products from files, and adding a spin penalty term if required.

        Parameters
        ----------
        D : numpy.ndarray
            Matrix representing the hybridization between the impurity and bath orbitals.
        H1 : numpy.ndarray
            One-body Hamiltonian matrix for the physical block of the embedding Hamiltonian.
        Lc : numpy.ndarray
            One-body Hamiltonian matrix for the bath block of the embedding Hamiltonian.
        U : float
            Interaction strength for the two-body term.

        Returns
        -------
        scipy.sparse.csr_matrix
            The embedding Hamiltonian for the impurity solver.

        Raises
        ------
        ValueError
            If lspin_sym is False, since the impurity solver is not yet adapted for
            non spin-symmetric systems.
        """
        if not self.lspin_sym: raise ValueError("Impurity solver not adapted yet for non spin-symmetric systems")

        D, H1E, LAMBDA, V2E = self.set_parameters(D, H1, Lc, U)
    
        Hemb  = csr_matrix((self.hsize_half,self.hsize_half), dtype=dtype_params)
        H1mat = csr_matrix((self.hsize_half,self.hsize_half), dtype=dtype_params)
        U2loc = csr_matrix((self.hsize_half,self.hsize_half), dtype=dtype_params)

        Hemb += self.read_prod_1e(H1E, 0, 0, self.n_phys_orb, self.n_phys_orb, 'phys', 'phys')
        H1mat = Hemb.copy()
        Hemb += self.read_prod_1e(D, self.n_phys_orb, 0, self.n_bath_orb, self.n_phys_orb, 'bath', 'phys')
        Hemb += self.read_prod_1e(D.T, 0, self.n_phys_orb, self.n_phys_orb, self.n_bath_orb, 'phys', 'bath')
        Hemb += self.read_prod_1e(LAMBDA, self.n_phys_orb, self.n_phys_orb, self.n_bath_orb, self.n_bath_orb, 'bath', 'bath')
    
        #build local two-body part
        prefix = f"phys-phys_V2E_imp-{str(self.impurity_nr)}_{self.impurity_type}"
        indices = range(self.n_phys_orb)
        for i, j, k, l in it.product(indices, repeat=4):               
            fname = f"{prefix}_op+{str(i)}-{str(j)}-{str(k)}-{str(l)}"
            prod = scipy.sparse.load_npz(fname+".npz")
            Hemb += 0.5*V2E[i,k,j,l]*prod
            U2loc+= 0.5*V2E[i,k,j,l]*prod

        # Load S^2 operator and add to Hemb for incorporating spin penalty
        S2 = scipy.sparse.load_npz("S2-imp" + str(self.impurity_nr) + self.impurity_type + ".npz")
        Hemb += self.spin_pen*S2
 
        # Save H1mat and U2loc for use in energy evaluation
        scipy.sparse.save_npz("H1mat-imp"+str(self.impurity_nr) + self.impurity_type, H1mat) 
        scipy.sparse.save_npz("U2loc-imp"+str(self.impurity_nr) + self.impurity_type, U2loc) 
    
        return Hemb



    def solve_Hemb(self, X, U, lspin_sym=True):
        """
        Solve the embedding Hamiltonian.

        This method builds the embedding Hamiltonian, solves it, and computes
        one-particle density matrix, local energy, and double occupancy.

        Parameters
        ----------
        D : numpy.ndarray
            Matrix representing the hybridization between the impurity and bath orbitals.
        H1 : numpy.ndarray
            One-body Hamiltonian matrix for the physical block of the embedding Hamiltonian.
        Lc : numpy.ndarray
            One-body Hamiltonian matrix for the bath block of the embedding Hamiltonian.
        U : float
            Interaction strength for the two-body term.
        lspin_sym : bool, optional
            If True, assumes the system is spin-symmetric. Default is True.

        Returns
        -------
        None
        """ 
        # Build embedding Hamiltonian
        H1 = np.array( [[-U/2.]] )
        Lc = np.diag(X[:self.n_bath_orb//2])
        D = np.array(X[self.n_bath_orb//2:]).reshape((self.n_bath_orb//2, self.n_phys_orb//2))

        self.Hemb = self.build_Hemb(D, H1, Lc, U)
   
        # Solve Hemb 
        num_eig = 2
        eig_vals, eig_vecs = primme.eigsh(self.Hemb, num_eig, tol=1e-12, which='SA')

        self.eig_vec = eig_vecs[:,0]
        self.eig_val = eig_vals[0]
    
        # Compute 1 particle density matrix and double occupany
        self.calc_density_matrix()
        self.calc_Eloc()

        # Check if we found the correct state
        self.sanity_check()

        if self.l1s: self.calc_double_docc(0)



    def sanity_check(self):    
        """
        Performs sanity checks on the impurity solver results to ensure the correct state is found.
        It checks for the following conditions:
        1. The correct state at half-filling
        2. The singlet state
        3. The ground/binding state

        Raises:
            ValueError: If any of the sanity checks fail, a ValueError is raised with a description of the failed check.
        """
        # Read total number and spin operators
        S2 = scipy.sparse.load_npz("S2-imp" + str(self.impurity_nr) + self.impurity_type + ".npz")
    
        # Sanity check 1: Did we find the correct state at half-filling?
        if np.abs( np.trace(self.dm) - self.n_half) > self.tol:
            raise ValueError('Incorrect number of 1s or 2s particles')
    
        # Sanity check 2: Did we find the singlet state?
        if np.abs(self.eig_vec.conj().T.dot(S2.dot( self.eig_vec))) > self.tol:
            raise ValueError('Incrrect spin state')
    
        # Sanity check 3: Did we find the ground/binding state?
#        if self.eig_val > 0.00:
#            raise ValueError('Eigenvalue of Hemb is larger than 0')



    def update_eig_vec(self, Phi):
        self.eig_vec = Phi
        self.eig_val = self.eig_vec.T.dot( self.Hemb.dot(self.eig_vec) )
        self.calc_density_matrix()

        self.sanity_check()



    def calc_density_matrix(self):
        """
        Calculates the 1-particle density matrix for the impurity solver using the eigenvector
        obtained from solving the embedding Hamiltonian. The density matrix is stored in the
        'self.dm' attribute of the class instance.

        The function computes the following matrices:
        - cdaggerc: expectation value of the product of physical creation and annihilation operators
        - fdaggerc: expectation value of the product of bath creation and physical annihilation operators
        - ffdagger: expectation value of the product of bath creation and annihilation operators, subtracted from the identity matrix

        The density matrix is constructed by combining these matrices as follows:
        - Upper-left block: cdaggerc
        - Lower-left block: fdaggerc
        - Upper-right block: transpose of fdaggerc
        - Lower-right block: ffdagger
        """ 
        self.dm = np.zeros((self.n_tot_orb, self.n_tot_orb),dtype=dtype_params)


        self.cdaggerc = self.read_prod_1e(self.eig_vec, 0, 0, 
                                          self.n_phys_orb, self.n_phys_orb, 'phys', 'phys', lexpectation=True)
        self.fdaggerc = self.read_prod_1e(self.eig_vec, self.n_phys_orb, 0, 
                                          self.n_bath_orb, self.n_phys_orb, 'bath', 'phys', lexpectation=True)
        self.ffdagger = self.read_prod_1e(self.eig_vec, self.n_phys_orb, self.n_phys_orb, 
                                          self.n_bath_orb, self.n_bath_orb, 'bath', 'bath', lexpectation=True)
        self.fdaggerf = np.eye(self.n_bath_orb) - self.ffdagger

        # Compute number of correlated particles
        self.nc = np.trace(self.cdaggerc)

        # Set up density matrix
        self.dm[:self.n_phys_orb,:self.n_phys_orb] = self.cdaggerc 
        self.dm[self.n_phys_orb:,:self.n_phys_orb] = self.fdaggerc 
        self.dm[:self.n_phys_orb,self.n_phys_orb:] = self.fdaggerc.T
        self.dm[self.n_phys_orb:,self.n_phys_orb:] = self.fdaggerf

        if self.lspin_sym:
            self.cdaggerc = cr.spin_symmetrize(self.cdaggerc, tol=1e-8)
            self.fdaggerc = cr.spin_symmetrize(self.fdaggerc, tol=1e-8)
            self.ffdagger = cr.spin_symmetrize(self.ffdagger, tol=1e-8)

        # Also compute y-vector for convenience
        self.y = [[self.eig_val] + np.diag(2.*self.ffdagger).tolist() 
                 + (4.*self.fdaggerc).flatten().tolist()]



    def calc_double_docc(self, idx):
        """
        Calculates the double occupancy of the impurity orbital with index 'idx' using the
        eigenvector obtained from solving the embedding Hamiltonian. The result is stored in
        the 'self.docc' attribute of the class instance.

        The function performs the following steps:
        1. Construct the file name prefix based on the impurity number and type.
        2. Load the stored sparse matrix 'prod1' corresponding to the product of the creation
           and annihilation operators for the given impurity orbital index 'idx'.
        3. Load the stored sparse matrix 'prod2' corresponding to the product of the creation
           and annihilation operators for the impurity orbital index 'idx+1'.
        4. Compute the double occupancy by taking the expectation value of the product of 'prod1'
           and 'prod2' with respect to the eigenvector obtained from solving the embedding Hamiltonian.

        Args:
            idx (int): Index of the impurity orbital for which the double occupancy will be calculated.
        """

        prefix = f"phys-phys_imp-{str(self.impurity_nr)}_{self.impurity_type}" 
         
        fname = f"{prefix}_op+{str(idx)}-{str(idx)}"
        prod1 = scipy.sparse.load_npz(fname+".npz")

        fname = f"{prefix}_op+{str(idx+1)}-{str(idx+1)}"
        prod2 = scipy.sparse.load_npz(fname+".npz")   
 
        self.docc =  self.eig_vec.conj().T.dot( prod1.dot( prod2.dot( self.eig_vec ) ) )



    def calc_Eloc(self):
        """
        Calculates the local one- and two-particle energies using the eigenvector obtained from
        solving the embedding Hamiltonian. The results are stored in the 'self.E1loc' and 'self.E2loc'
        attributes of the class instance.

        The function performs the following steps:
        1. Load the stored sparse matrix 'H1mat', which represents the one-particle operators
           for the embedding Hamiltonian.
        2. Load the stored sparse matrix 'U2loc', which represents the local two-particle interaction.
        3. Compute the local one-particle energy by taking the expectation value of 'H1mat'
           with respect to the eigenvector obtained from solving the embedding Hamiltonian.
        4. Compute the local two-particle energy by taking the expectation value of 'U2loc'
           with respect to the eigenvector obtained from solving the embedding Hamiltonian.
        """ 
    
        # Load 1 and 2 particle operators
        H1mat   = scipy.sparse.load_npz("H1mat-imp"+str(self.impurity_nr) + self.impurity_type + ".npz")
        U2loc   = scipy.sparse.load_npz("U2loc-imp"+str(self.impurity_nr) + self.impurity_type + ".npz")
    
        self.E1loc = self.eig_vec.conj().T.dot( H1mat.dot( self.eig_vec ) )
        self.E2loc = self.eig_vec.conj().T.dot( U2loc.dot( self.eig_vec ) )
