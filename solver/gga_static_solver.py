import numpy as np
import convenience_routines as cr
import scipy
from scipy import optimize, integrate
from scipy.linalg import eigh, expm, det
import ed_solver as ed
#import fci_solver as fci
import sys
import os
import matplotlib
from pylab import *
from functools import reduce
import itertools as it
import sys
from lattice import *
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 日本語フォント設定（Mac用・強制適用版）
# -----------------------------------------------------------
try:
    # システム内の「ヒラギノ」を含むフォントファイルを検索
    fonts = [f for f in fm.findSystemFonts() if 'Hiragino' in f]
    
    if len(fonts) > 0:
        # 見つかった最初のヒラギノフォントをMatplotlibに登録
        target_font = fonts[0] 
        fm.fontManager.addfont(target_font)
        
        # 登録したフォントの正確な名前を取得して設定
        font_name = fm.FontProperties(fname=target_font).get_name()
        plt.rcParams['font.family'] = font_name
        print(f"成功: 日本語フォント '{font_name}' をロードしました。")
    else:
        # ヒラギノがない場合（稀ですが）、AppleGothicを試行
        plt.rcParams['font.family'] = 'AppleGothic'
        print("警告: ヒラギノが見つからなかったため、AppleGothicを設定しました。")
except Exception as e:
    print(f"フォント設定エラー: {e}")

# マイナス記号の文字化け防止
plt.rcParams['axes.unicode_minus'] = False 
# -----------------------------------------------------------

################### Class for the Anderson Lattice Model (ALM) ###################

class GA:
    def __init__(self,U,nghost,nphysorb,T=0.0002,lcanonical=True,n=0.5,tolconv=1e-7, eks=0):
        self.nphysorb   = nphysorb                   # Physical spin orbitals
        self.nghost     = nghost
        self.nquasiorb  = nphysorb + nghost          # Spin quasi orbs (Physical + Ghost)
        self.nqspo      = self.nquasiorb//2          # Spatial quasi orbs (Physical + Ghost)

        self.U          = U                          # Interaction potential
        self.T          = T                          # Fermi temperature
        self.lcanonical = lcanonical
        self.tolconv    = tolconv                    # If F_i>tolconv, print warning.
        self.mu = 0.0                                 
        self.eks = np.real(eks)                      # non-local hopping energy

        # Initialize with silly values
        self.n = n if self.lcanonical else -1

        self.H_list = cr.generate_orthonormal_basis(self.nqspo)

        self.imp_solver = ed.edSolver(self.nphysorb,nghost,0,'s')

        self.docc  = 0.
        self.E1loc = 0.
        self.E2loc = 0.
        self.Eqp   = 0.
        self.Etot  = 0.

        self.niter = 0 # Number of iterations
        self.ned   = 0 # Number of EDs

        self.dos = self.dos_sc
        if lksum: self.dos = lambda x: 1.0


    def dos_sc(self,x):
    # Semicircular density of states
        D = (2.0/np.pi)*np.sqrt(1.0-x**(2.0))
        if np.abs(x)>1:
          D = 0.0
        return D

    def root_mu_Hemb(self,mu):
        self.mu_eh = mu[0]
        self.Lmbdac = self.Lmbdac_orig - np.eye(self.nqspo)*mu

        self.solve_Hemb()
        self.mu_list.append(mu[0])
        self.muiter += 1

        Trace = self.imp_solver.nc/2.
        if nghost==2:
          Trace -= 1

        self.microitF3 += 1
        print(f"    F3 Micro-It. {self.microitF3}: F3= ", Trace-self.n)

        return (Trace-self.n)


    def calc_mu_Hemb(self):
        # Determine chemical of Hemb for a given filliing
        self.mu_list = []
        self.muiter = 0
        self.Lmbdac_orig = np.copy(self.Lmbdac+np.eye(self.nqspo)*self.mu_eh)

        chem_pot = optimize.root(self.root_mu_Hemb,self.mu)

        return chem_pot.x


    def calc_Lmbdac(self,Lmbda,Delta):
        # Compute Lagrange multipliers Lambda^c
        lmbda = cr.inverse_realHcombination(Lmbda,self.H_list)
        lmbdac = np.zeros((len(lmbda)))
        DxR = np.dot(self.D,self.R.T)

        for imat in range(len(lmbda)):
          deriv_Delta = cr.dF(Delta,self.H_list[imat].T,cr.denRm1,cr.ddenRm1)
          deriv_DeltaxDR = np.trace(np.dot(DxR,deriv_Delta))
          lmbdac[imat] = -lmbda[imat] - (deriv_DeltaxDR + np.conjugate(deriv_DeltaxDR))

        return cr.realHcombination(lmbdac,self.H_list)


    def solve_Hemb(self):
        #H1 = np.array( [[-U/2.]] )

        D, Lmbdac, phasemat, permmat, transmat = self.fix_gauge(self.D, self.Lmbdac, lfor_D=True, lreturn_mats=True)
        X = np.diag(Lmbdac).tolist() + D.flatten().tolist()

        self.imp_solver.solve_Hemb(X, self.U)

        # Transform <cd> and <dd> (actually <cg> and <gg>) back to original basis (Transformation order is crucial)
        # Transform back using perutation matrix
        self.imp_solver.fdaggerc = np.dot(permmat.T, self.imp_solver.fdaggerc)
        self.imp_solver.ffdagger = np.dot(permmat.T,np.dot(self.imp_solver.ffdagger,permmat))
        # Transform back using matrix of phase factors
        self.imp_solver.fdaggerc = np.dot(phasemat, self.imp_solver.fdaggerc)
        self.imp_solver.ffdagger = np.dot(phasemat,np.dot(self.imp_solver.ffdagger,phasemat.T))
        # Transform back using eigenbasis of Lambda^c
        self.imp_solver.fdaggerc = np.dot(transmat, self.imp_solver.fdaggerc)
        self.imp_solver.ffdagger = np.dot(transmat,np.dot(self.imp_solver.ffdagger,transmat.T))
        self.ned += 1
        #print(U, self.U)
     





    def fix_gauge(self,R,Lmbda, lfor_D=False, lreturn_mats=False):
        """
           Transform R and Lambda to basis where Lambda is diagonal and R_1 > R_2 > R_3 >= 0
            
           Args:
             R and Lambda in original basis

           return:
             R and Lambda in eigenbasis of Lambda and potentially resorted and with fixed phase
        """

        transmat = scipy.linalg.eigh(Lmbda)[1]
        Lmbda = np.dot(transmat.T,np.dot(Lmbda,transmat))
        R = np.dot(transmat.T,R)

        d_fac = 1. if not lfor_D else -1.

        # Check if R has any negative elements and make them positive and the same for phasemat
        # which transforms <cd> and <dd> according to positive entries in R
        phasemat = np.eye(self.nqspo)
        for i in range(self.nqspo):
          if d_fac*R[i,0]<0.0:
            R[i,0] *= -1.0
            phasemat[i,i] *= -1.0

        if self.nqspo>1:
          Lmbda = np.dot(phasemat,np.dot(Lmbda,phasemat.T))

        # Get the indices of the elements of R in descending order
        idx = np.argsort(-R[:,0])
        # Create permutation matrix
        permmat = np.zeros((self.nqspo, self.nqspo))
        for i in range(self.nqspo):
            permmat[i, idx[i]] = 1
    

        # Transform D and Lambda^c by permutation matrix
        R = np.dot(permmat,R)
        Lmbda = np.dot(permmat,np.dot(Lmbda,permmat.T))

        if locdbg:
          print(f"transmat\n{transmat}")
          print(f"phasemat\n{phasemat}")
          print(f"permmat\n{permmat}")
          print(f"Lmbda1 in fix\n{Lmbda}")
          print(f"R1 in fix\n{R}")
        
        if not lreturn_mats:
            return R, Lmbda
        else:
            return R, Lmbda, phasemat, permmat, transmat



    def optimize_selfc(self,rinit=None,lambdainit=None,muinit=None):
        """
          Explcitly self-consistent way of solving GA equations:

          Internally optimize R, Lambda, mu in each iteration and check convergence based on 
          differences between initial and optimized R and Lambda (or Z and total energy)
        """

        def root_GA(xinit):
            tolscRL = 1e-3     # Convergence threshold for R and Lambda
            tolscE = 1e-4      # Convergence threshold for total energy 
            tolscZ = 1e-4      # Convergence threshold for Z
            useconv = "RL"     # "RL": Use R and Lambda for convergence check; "EZ": Use total energy and Z
           

            diff = 99.9        # Initialize diff with unreasonably large number
            lmu_Hemb = True    # Chemical potential in Hemb or Hqp?
            maxiter = 30       # Maximum number of iterations

            self.mu_fermi=0.0   # Chemical potential on Hqp
            self.mu = xinit[-1] # Global chemical potential

            self.lconv = False      # Convergence flag
            
            while not self.lconv:
                self.niter += 1
                print ("####### ITERATION ", self.niter, "#######")

                lmbda0 = xinit[0:self.nqspo*(self.nqspo+1)//2]
                r0     = xinit[self.nqspo*(self.nqspo+1)//2:self.nqspo*(self.nqspo+1)//2 + self.nqspo]


                # Construct R and Lambda from vector
                self.R = np.reshape(np.array(r0),(self.nqspo,1))
                self.Lmbda = cr.realHcombination(lmbda0,self.H_list)

                # Save R and Lambda separately for convergence check
                R_orig = np.copy(self.R)
                Lmbda_orig = np.copy(self.Lmbda)

                # Fix gauge of R and Lambda
                R_orig, Lmbda_orig = self.fix_gauge(R_orig,Lmbda_orig)

                if self.niter==1:
                  print ("Initial mu", self.mu)
                  print ("Initial R")
                  print (self.R)
                  print ("Initial Lambda")
                  print (self.Lmbda)
                  print(lmbda0)

                if lmu_Hemb:
                  self.mu_fermi = 0.0
                  self.mu_eh    = self.mu
                else:
                  self.mu_fermi = self.mu = self.calc_mu()
                  self.mu_eh    = 0.0

                # Compute Delta
                self.calc_Delta()
                self.Delta = self.Delta[0:self.nqspo,0:self.nqspo]

                # Compute Lagrange multipliers D
                self.calc_D()

                # Compute Lagrange multipliers Lambda^c
                self.Lmbdac = self.calc_Lmbdac(self.Lmbda,self.Delta)


                # Save Z and Etot for convergence check
                self.Z_old = self.calc_Z(Lmbda_orig,R_orig)
                self.Etot_old = np.copy(self.Etot)


                # Diagonalize embeeding Hamiltonian
                self.solve_Hemb()

                # [metallic fix] ゴーストモードの縮退を処理してn_el=Δ(=0.5)に固定
                if getattr(self, '_metallic_mode', False) and self.nghost > 0:
                    _delta_sp = self.Delta[:self.nqspo, :self.nqspo]
                    _f2_check = float(np.linalg.norm(
                        self.imp_solver.ffdagger - _delta_sp, 'fro'))
                    if _f2_check > 0.05:
                        self._fix_degenerate_phi(_delta_sp)

                # If number of particles in Hemb not what we want: Modify mu
                if not lmu_sweep:
                  if np.abs(self.imp_solver.nc-self.n)>1e-3:
                    self.microitF3 = 0
                    self.mu = self.calc_mu_Hemb()
                    print(f"F3 Microiterations completed in {self.microitF3} iterations")

                if locdbg:
                  print('n, nc, nu',self.n,self.imp_solver.nc,self.nu)
                  print('mu, mu_fermi, mu_eh',self.mu,self.mu_fermi,self.mu_eh)


                # Try minimizing, if not successful break current loop or continue if F1, F2 sufficiently small (< 1e-5)
                # Minimize F1, F2 with respect to R and Lambda using least-squares minimizer
                uvec = np.hstack((cr.inverse_realHcombination(self.Lmbda,self.H_list),self.R[:,0]))
                self.microitF1F2 = 0
                try:
                  result = optimize.least_squares(self.cost_func, uvec)
                  print(f"F1, F2 Microiterations converged in {self.microitF1F2} iterations")
                except:
                  if  self.F1_max>1e-5 and self.F2_max>1e-5:
                      self.R_new = np.copy(R_orig)
                      self.Lmbda_new = np.copy(Lmbda_orig)
                      self.Z = self.calc_Z(Lmbda_orig,R_orig)
                      print(f"F1, F2 Microiterations not converged in 7000 iterations and not below threshold: Stop calculation")
                      break
                  else:
                      print(f"F1, F2 Microiterations not converged in iterations but below thresold: Continue calculation")
                      result = self.xnosol
    
    
                if locdbg:
                  print(f'Delta after  microit\n {self.Delta[0:self.nqspo,0:self.nqspo]}')
                  print(f'cd after    microit\n {self.imp_solver.fdaggerc}')
                  print(f'dd after    microit\n {self.imp_solver.ffdagger}')
                  print(f'R after     microit\n {self.R}')
                  print(f'Lambda after microit\n {self.Lmbda}')
    
    
                # Construct new R and Lambda from optimized solution
                lmbda0_new = result.x[0:self.nqspo*(self.nqspo+1)//2]
                r0_new     = result.x[self.nqspo*(self.nqspo+1)//2:self.nqspo*(self.nqspo+1)//2 + self.nqspo]
                self.R_new = np.reshape(np.array(r0_new),(self.nqspo,1))
                self.Lmbda_new = cr.realHcombination(lmbda0_new,self.H_list)

                # disk基底の Lmbda と R を保存 (TD-gGA で calc_Z に使う)
                self.Lmbda_disk = np.copy(self.Lmbda_new)
                self.R_disk     = np.copy(self.R_new)

                # Also fix gauge of new R and Lambda
                self.R_new, self.Lmbda_new = self.fix_gauge(self.R_new,self.Lmbda_new)

                # Compute quasi-particle energy
                self.calc_Eqp()

                # Explicit anti-symmetrization of Lambda in Mott phase (only N=1, only for nqspo==3)
                if self.nghost>0 and self.nqspo==3 and (self.n==0.5 or (self.mu==0. and lmu_sweep)):
                  if np.abs(self.Lmbda_new[0,0]) > np.abs(self.Lmbda_new[2,2]) and  np.abs(self.Lmbda_new[1,1]) > np.abs(self.Lmbda_new[2,2]):
                    l1 = (self.Lmbda_new[0,0] - self.Lmbda_new[1,1])/2.
                    l2 = (self.Lmbda_new[1,1] - self.Lmbda_new[0,0])/2.
                    self.Lmbda_new[0,0] = l1
                    self.Lmbda_new[1,1] = l2
                  elif  np.abs(self.Lmbda_new[1,1]) > np.abs(self.Lmbda_new[0,0]) and  np.abs(self.Lmbda_new[2,2]) > np.abs(self.Lmbda_new[0,0]):
                    l1 = (self.Lmbda_new[1,1] - self.Lmbda_new[2,2])/2.
                    l2 = (self.Lmbda_new[2,2] - self.Lmbda_new[1,1])/2.
                    self.Lmbda_new[1,1] = l1
                    self.Lmbda_new[2,2] = l2


                # Decompose old and new R and Lambda into basis vectors
                x_new  = np.hstack((cr.inverse_realHcombination(self.Lmbda_new,self.H_list),self.R_new[:,0]))
                x_old  = np.hstack((cr.inverse_realHcombination(Lmbda_orig,self.H_list),R_orig[:,0]))       

                # Compute error/difference between old and new R and Lambda for convergence check 
                error = np.abs(np.array(x_new)) - np.abs(np.array(x_old))
                RLdiff = np.abs(error).max()
                lconvRL = RLdiff < tolscRL


                # Compute differences in Z and Etot for alternative convergence check
                self.Z = self.calc_Z(self.Lmbda_new,self.R_new)
                self.E1loc = (self.U/2.0)*self.imp_solver.nc
                self.Etot = self.Eqp + self.U*self.docc - self.E1loc +self.U/2.0
                Ediff = np.abs(self.Etot_old-self.Etot)
                lconvE = Ediff < tolscE
                lconvZ = np.abs(self.Z_old-self.Z) <  tolscZ

                # Converged?
                self.lconv = (lconvE and lconvZ and useconv=="EZ") or (lconvRL and useconv=="RL")

                Rdiff = np.abs( np.abs(self.R_new) - np.abs(R_orig) )
                Ldiff = np.abs( np.abs(self.Lmbda_new) - np.abs(Lmbda_orig) )

                if locdbg:
                  print (f"Lambda new\n{self.Lmbda_new}")
                  print (f"R new\n{self.R_new}")
                  print (f"Lambda orig\n{Lmbda_orig}")
                  print (f"R orig\n{R_orig}")
                  print (f"Lambda diff\n{Ldiff}")
                  print (f"R diff\n{Rdiff}")
                  print("mu: ",self.mu,self.mu_eh,self.mu_fermi)
                  print(f"RLdiff",RLdiff)
                  print("Zdiff",np.abs(self.Z_old-self.Z), self.Z_old, self.Z)
                  print(f"Nc: {self.imp_solver.nc}")
                  print(f"Max. error {self.niter},   {RLdiff}")
                  print(f"Z in main {self.Z}")
                print(f"Max. error {self.niter},   {RLdiff}")

                if self.niter>maxiter: break

                xinit = x_new


            self.R = self.R_new
            self.Lmbda = self.Lmbda_new


        self.rinit = rinit
        self.lambdainit = lambdainit
        xinit = np.hstack((lambdainit,rinit,muinit))

        # Call GA solver and find saddle point of Lagrangian
        root_GA(xinit)
   
        # Check if everything is converged 
        self.lconverged_root = True
        if self.F1nrm > self.tolconv:
            print ("WARNING: F1 not converged", self.F1nrm)
            self.lconverged_root = False
        if self.F2nrm > self.tolconv:
            print ("WARNING: F2 not converged", self.F2nrm)
            self.lconverged_root = False
        self.lconverged_root = self.lconverged_root and self.lconv


    def optimize_selfc_metallic(self, rinit=None, lambdainit=None, muinit=None):
        """
        全 gGA 静的ソルバー（金属的解探索版）。

        optimize_selfc と同一だが、solve_Hemb 後にゴーストモードの縮退を
        _fix_degenerate_phi で処理し n_el_ghost → Δ_ghost (≈0.5) に固定する。

        これにより「PH 凍結解」ではなく「金属的解（Λ≈0, n_el_ghost≈0.5）」に
        収束することを狙う。lambdainit=0（Λ=0 出発）で呼ぶこと。
        """
        nhl = self.nqspo * (self.nqspo + 1) // 2
        if lambdainit is None:
            lambdainit = np.zeros(nhl)            # Λ=0 出発が必須
        if rinit is None:
            rinit = np.zeros(self.nqspo); rinit[0] = 1.0
        if muinit is None:
            muinit = 0.0

        self._metallic_mode = True                # solve_Hemb 内でフィックスを有効化
        try:
            self.optimize_selfc(rinit=rinit, lambdainit=lambdainit, muinit=muinit)
        finally:
            self._metallic_mode = False           # 必ず元に戻す


    def cost_func(self,xinit):
           lmbda0 = xinit[0:self.nqspo*(self.nqspo+1)//2]
           r0     = xinit[self.nqspo*(self.nqspo+1)//2:self.nqspo*(self.nqspo+1)//2 + self.nqspo]

           self.R = np.reshape(np.array(r0),(self.nqspo,1))
           self.Lmbda = cr.realHcombination(lmbda0,self.H_list)


           self.calc_Delta()
           self.Delta = self.Delta[0:self.nqspo,0:self.nqspo]

           if locdbg: 
             print(f'<cd>    in microit \n{self.imp_solver.fdaggerc.T}')
             print(f'<dd>    in microit \n{self.imp_solver.ffdagger}')
             print(f'R        in microit \n{self.R}')
             print(f'Lambda   in microit \n{self.Lmbda}')
             print(f'Delta    in microit \n{self.Delta}')
             print(f'Trace    in microit \n{np.trace(self.Delta)-1}')
             print(f'mu_fermi in microit \n{self.mu_fermi}')

           # Compute F1 and F2
           self.F1 = self.imp_solver.fdaggerc.T - np.dot(self.R.T,cr.funcMat(self.Delta, cr.denRm1))
           self.F2 = self.imp_solver.ffdagger - self.Delta

           self.F1_max = np.abs(self.F1).max()
           self.F2_max = np.abs(self.F2).max()

           # Compute norm of F1 and F2 for convergence check 
           self.F1nrm = np.linalg.norm(np.power(self.F1,1),'fro')
           self.F2nrm = np.linalg.norm(np.power(self.F2,1),'fro')

           if locdbg:
             print(f'F1    in microit\n{self.F1}')
             print(f'F2    in microit\n{self.F2}')


           if self.microitF1F2>6000:
             self.xnosol = type("X",(object,),{'x':xinit})()
             return("la") # Return non-sense such that the solver crashes and we can catch that using an exception


           self.microitF1F2 += 1
           print(f"    F1, F2 Micro-It. {self.microitF1F2}: F1, F2= ", self.F1_max, self. F2_max)
           #sys.exit("LA fct")

           return np.hstack((cr.inverse_realHcombination(self.F2,self.H_list),self.F1[0]))




#mf new Yongxin solver
    def optimize_selfc_new(self,rinit=None,lambdainit=None,muinit=None):
        """
          Explcitly self-consistent way of solving GA equations:

          Internally optimize R, Lambda, mu in each iteration and check convergence based on 
          differences between initial and optimized R and Lambda (or Z and total energy)
        """

        def root_GA(xinit):
            tolscRL = 1e-3     # Convergence threshold for R and Lambda
            tolscE = 1e-4      # Convergence threshold for total energy 
            tolscZ = 1e-4      # Convergence threshold for Z
            useconv = "RL"     # "RL": Use R and Lambda for convergence check; "EZ": Use total energy and Z
           

            diff = 99.9        # Initialize diff with unreasonably large number
            maxiter = 30       # Maximum number of iterations

            self.mu_fermi=0.0  # Chemical potential on Hqp
            self.mu = xinit[-1] #0.28987805 #0.0      # Global chemical potential
            lmu_Hemb = True

            self.lconv = False      # Convergence flag
            
            while not self.lconv:
                self.niter += 1
                print ("####### ITERATION ", self.niter, "#######")

                lmbda0 = xinit[0:self.nqspo*(self.nqspo+1)//2]
                r0     = xinit[self.nqspo*(self.nqspo+1)//2:self.nqspo*(self.nqspo+1)//2 + self.nqspo]


                # Construct R and Lambda from vector
                self.R = np.reshape(np.array(r0),(self.nqspo,1))
                self.Lmbda = cr.realHcombination(lmbda0,self.H_list)

                # Save R and Lambda separately for convergence check
                R_orig = np.copy(self.R)
                Lmbda_orig = np.copy(self.Lmbda)

                # Fix gauge of R and Lambda
                R_orig, Lmbda_orig = self.fix_gauge(R_orig,Lmbda_orig)

                if self.niter==1:
                  print ("Initial mu", self.mu)
                  print ("Initial R")
                  print (self.R)
                  print ("Initial Lambda")
                  print (self.Lmbda)
                  print(lmbda0)

                if lmu_Hemb:
                  self.mu_fermi = 0.0
                  self.mu_eh    = self.mu
                else:
                  self.mu_fermi = self.mu = self.calc_mu()
                  self.mu_eh    = 0.0

                # Compute Delta
                self.calc_Delta()
                self.Delta = self.Delta[0:self.nqspo,0:self.nqspo]

                # Compute Lagrange multipliers D
                self.calc_D()

                # Compute Lagrange multipliers Lambda^c
                self.Lmbdac = self.calc_Lmbdac(self.Lmbda,self.Delta)


                # Save Z and Etot for convergence check
                self.Z_old = self.calc_Z(Lmbda_orig,R_orig)
                self.Etot_old = np.copy(self.Etot)


                # Diagonalize embeeding Hamiltonian
                self.solve_Hemb()

                # If number of particles in Hemb not what we want: Modify mu
                if not lmu_sweep:   
                  if np.abs(self.imp_solver.nc-self.n)>1e-3:
                    self.microitF3 = 0
                    self.mu = self.calc_mu_Hemb()
                    print(f"F3 Microiterations completed in {self.microitF3} iterations")

                self.R     = ( self.imp_solver.fdaggerc.T.dot( cr.funcMat(self.imp_solver.ffdagger, cr.denR) ) ).T
                self.R_new = np.copy(self.R)
#                self.Delta = np.copy(self.imp_solver.ffdagger)


                if locdbg:
                  print('n, nc, nu',self.n,self.imp_solver.nc,self.nu)
                  print('mu, mu_fermi, mu_eh',self.mu,self.mu_fermi,self.mu_eh)



                # Minimize F1, F2 with respect to R and Lambda using least-squares minimizer
                uvec = np.hstack((cr.inverse_realHcombination(self.Lmbda,self.H_list) )) #,self.mu))
                self.microitF1F2 = 0
    
                # Try minimizing, if not successful break current loop or continue if F1, F2 sufficiently small (< 1e-5)
                result = optimize.least_squares(self.cost_func_Delta, uvec)
                print(f"F1, F2 Microiterations converged in {self.microitF1F2} iterations")
    
    
#                print(result.x)
#                sys.exit()

                if locdbg:
                  print(f'Delta after  microit\n {self.Delta[0:self.nqspo,0:self.nqspo]}')
                  print(f'cd after    microit\n {self.imp_solver.fdaggerc}')
                  print(f'dd after    microit\n {self.imp_solver.ffdagger}')
                  print(f'R after     microit\n {self.R}')
                  print(f'Lambda after microit\n {self.Lmbda}')
    
    
                # Construct new R and Lambda from optimized solution
                lmbda0_new = result.x[0:self.nqspo*(self.nqspo+1)//2]
                self.Lmbda_new = cr.realHcombination(lmbda0_new,self.H_list)
#                self.mu = 0. #result.x[-1]

                # disk基底の Lmbda と R を保存 (TD-gGA で calc_Z に使う)
                self.Lmbda_disk = np.copy(self.Lmbda_new)
                self.R_disk     = np.copy(self.R_new)

                # Also fix gauge of new R and Lambda
                self.R_new, self.Lmbda_new = self.fix_gauge(self.R_new,self.Lmbda_new)

                # Compute quasi-particle energy
                self.calc_Eqp()

                # Explicit anti-symmetrization of Lambda in Mott phase (only N=1, only for nqspo==3)
                if self.nghost>0 and self.nqspo==3 and (self.n==0.5 or (self.mu==0. and lmu_sweep)):
                  if np.abs(self.Lmbda_new[0,0]) > np.abs(self.Lmbda_new[2,2]) and  np.abs(self.Lmbda_new[1,1]) > np.abs(self.Lmbda_new[2,2]):
                    l1 = (self.Lmbda_new[0,0] - self.Lmbda_new[1,1])/2.
                    l2 = (self.Lmbda_new[1,1] - self.Lmbda_new[0,0])/2.
                    self.Lmbda_new[0,0] = l1
                    self.Lmbda_new[1,1] = l2
                  elif  np.abs(self.Lmbda_new[1,1]) > np.abs(self.Lmbda_new[0,0]) and  np.abs(self.Lmbda_new[2,2]) > np.abs(self.Lmbda_new[0,0]):
                    l1 = (self.Lmbda_new[1,1] - self.Lmbda_new[2,2])/2.
                    l2 = (self.Lmbda_new[2,2] - self.Lmbda_new[1,1])/2.
                    self.Lmbda_new[1,1] = l1
                    self.Lmbda_new[2,2] = l2


                # Decompose old and new R and Lambda into basis vectors
                x_new  = np.hstack((cr.inverse_realHcombination(self.Lmbda_new,self.H_list),self.R_new[:,0]))
                x_old  = np.hstack((cr.inverse_realHcombination(Lmbda_orig,self.H_list),R_orig[:,0]))       

                # Compute error/difference between old and new R and Lambda for convergence check 
                error = np.abs(np.array(x_new)) - np.abs(np.array(x_old))
                RLdiff = np.abs(error).max()
                lconvRL = RLdiff < tolscRL


                # Compute differences in Z and Etot for alternative convergence check
                self.Z = self.calc_Z(self.Lmbda_new,self.R_new)
                self.E1loc = (self.U/2.0)*self.imp_solver.nc
                self.Etot = self.Eqp + self.U*self.docc - self.E1loc +self.U/2.0
                Ediff = np.abs(self.Etot_old-self.Etot)
                lconvE = Ediff < tolscE
                lconvZ = np.abs(self.Z_old-self.Z) <  tolscZ



                # Converged?
                self.lconv = (lconvE and lconvZ and useconv=="EZ") or (lconvRL and useconv=="RL")



                Rdiff = np.abs( np.abs(self.R_new) - np.abs(R_orig) )
                Ldiff = np.abs( np.abs(self.Lmbda_new) - np.abs(Lmbda_orig) )

                if locdbg:
                  print (f"Lambda new\n{self.Lmbda_new}")
                  print (f"R new\n{self.R_new}")
                  print (f"Lambda orig\n{Lmbda_orig}")
                  print (f"R orig\n{R_orig}")
                  print (f"Lambda diff\n{Ldiff}")
                  print (f"R diff\n{Rdiff}")
                  print("mu: ",self.mu,self.mu_eh,self.mu_fermi)
                  print(f"RLdiff",RLdiff)
                  print("Zdiff",np.abs(self.Z_old-self.Z), self.Z_old, self.Z)
                  print(f"Nc: {self.imp_solver.nc}")
                  print(f"Max. error {self.niter},   {RLdiff}")
                  print(f"Z in main {self.Z}")
                print(f"Max. error {self.niter},   {RLdiff}")

                if self.niter>maxiter: break

                xinit = x_new


            self.R = self.R_new
            self.Lmbda = self.Lmbda_new


        self.rinit = rinit
        self.lambdainit = lambdainit
        xinit = np.hstack((lambdainit,rinit,muinit))

        # Call GA solver and find saddle point of Lagrangian
        root_GA(xinit)
   
        # Check if everything is converged 
        self.lconverged_root = True
        if self.F1nrm > self.tolconv:
            print ("WARNING: F1 not converged", self.F1nrm)
            self.lconverged_root = False
        if self.F2nrm > self.tolconv:
            print ("WARNING: F2 not converged", self.F2nrm)
            self.lconverged_root = False
        self.lconverged_root = self.lconverged_root and self.lconv




    def optimize_selfc_routeA(self, rinit=None, muinit=None, maxiter=50, tol=1e-8):
        """
        Route A (Λ=0 ゲージ) 静的ソルバー。
        Λ を常にゼロに固定した自己無撞着方程式を解く。
        Λ の最適化ステップを省き、R と μ のみを反復する。
        論文 Eq.(15) の「setting Λ=0」に対応。
        """
        nqspo = self.nqspo

        self.Lmbda    = np.zeros((nqspo, nqspo))
        self.mu_fermi = 0.0
        self.mu_eh    = muinit if muinit is not None else self.mu
        self.mu       = self.mu_eh

        if rinit is not None:
            self.R = np.reshape(np.array(rinit), (nqspo, 1))

        self.lconv = False
        self.niter = 0

        for it in range(maxiter):
            self.niter += 1
            self.Lmbda = np.zeros((nqspo, nqspo))   # Λ=0 を全反復で強制

            R_old = np.copy(self.R)

            # Δ = ∫ρ(ω) f(ωRR†) dω  (Λ=0 → H_qp = ωRR† − μI)
            self.calc_Delta()
            self.Delta = self.Delta[:nqspo, :nqspo]

            # D (Λ=0 で計算)
            self.calc_D()

            # Λ^c = −deriv(D·R, Δ)  (Λ=0 なのでオフセット項は消える)
            self.Lmbdac = self.calc_Lmbdac(self.Lmbda, self.Delta)

            # H_emb を対角化
            self.mu_eh = self.mu
            self.solve_Hemb()

            # 粒子数を目標値に合わせて μ を調整
            if abs(self.imp_solver.nc - self.n) > 1e-3:
                self.microitF3 = 0
                self.mu    = float(self.calc_mu_Hemb())
                self.mu_eh = self.mu

            # 縮退部分空間で F2 を最小化する |Φ⟩ を選ぶ
            F2_cur = float(np.linalg.norm(self.imp_solver.ffdagger - self.Delta, 'fro'))
            if F2_cur > 0.05:
                self._fix_degenerate_phi(self.Delta)

            # R を更新: R = [Δ(1−Δ)]^{-1/2} ⟨f†c⟩
            self.R     = (self.imp_solver.fdaggerc.T.dot(
                              cr.funcMat(self.imp_solver.ffdagger, cr.denR))).T
            self.R_new = np.copy(self.R)

            dR = np.max(np.abs(self.R - R_old))
            print(f"  RouteA SC iter {it+1:3d}: dR={dR:.3e}  nc={self.imp_solver.nc:.5f}")

            if dR < tol:
                self.lconv = True
                print(f"  Converged in {it+1} iterations (dR={dR:.2e})")
                break

        # 最終的な後処理
        self.Lmbda      = np.zeros((nqspo, nqspo))
        self.Lmbda_disk = np.zeros((nqspo, nqspo))
        self.R_disk     = np.copy(self.R_new)
        self.Z          = self.calc_Z(self.Lmbda, self.R_new)
        self.calc_Eqp()
        self.E1loc = (self.U / 2.0) * self.imp_solver.nc
        self.Etot  = self.Eqp + self.U * self.docc - self.E1loc + self.U / 2.0

        # 拘束条件確認
        self.calc_Delta()
        self.Delta = self.Delta[:nqspo, :nqspo]
        F2_mat      = self.imp_solver.ffdagger - self.Delta
        self.F2nrm  = np.linalg.norm(F2_mat, 'fro')
        self.F1nrm  = abs(self.n - (np.trace(self.Delta) - 1))

        self.lconverged_root = self.lconv and (self.F2nrm < self.tolconv)
        if not self.lconv:
            print(f"  WARNING: RouteA SC 未収束 (dR={dR:.2e})")
        if self.F2nrm > self.tolconv:
            print(f"  WARNING: F2 未収束 (F2={self.F2nrm:.2e})")


    def _fix_degenerate_phi(self, Delta, tol_degen=0.05, n_tries=10):
        """
        H_emb の縮退部分空間で ||⟨f†f⟩ − Δ|| を最小化する |Φ⟩ を選ぶ。
        縮退が検出された場合のみ imp_solver.update_eig_vec を呼んで状態を更新する。
        """
        import primme
        import scipy.sparse as sp_mod
        from scipy.optimize import minimize

        nqspo = self.nqspo
        imp   = self.imp_solver

        k = min(8, imp.hsize_half - 1)
        try:
            evals, evecs = primme.eigsh(imp.Hemb, k, tol=1e-10, which='SA')
        except Exception:
            return

        idx   = np.argsort(evals)
        evals = evals[idx];  evecs = evecs[:, idx]

        E0         = evals[0]
        degen_mask = (evals - E0) < tol_degen
        V          = evecs[:, degen_mask]   # (dim_Phi, n_deg)
        n_deg      = V.shape[1]

        if n_deg <= 1:
            return

        next_gap = float(evals[np.sum(degen_mask)] - E0) if np.sum(degen_mask) < k else float('inf')
        print(f"    [degen] {n_deg} 縮退状態, next gap={next_gap:.4f}")

        # 縮退部分空間の bath-bath 行列要素
        # M[a,b] = 0.5*(V.T @ op_{2a,2b} @ V + V.T @ op_{2a+1,2b+1} @ V)
        imp_nr   = imp.impurity_nr
        imp_type = imp.impurity_type
        M = np.zeros((nqspo, nqspo, n_deg, n_deg))
        for a in range(nqspo):
            for b in range(nqspo):
                for spin in range(2):
                    fname    = f"bath-bath_imp-{imp_nr}_{imp_type}_op+{2*a+spin}-{2*b+spin}.npz"
                    op_dense = sp_mod.load_npz(fname).toarray()
                    M[a, b] += 0.5 * (V.T @ op_dense @ V)

        Delta_target = np.real(Delta[:nqspo, :nqspo])

        def residual(x):
            norm = np.linalg.norm(x)
            c    = x / norm if norm > 1e-300 else x
            ffd  = np.einsum('abkl,k,l->ab', M, c, c)
            return float(np.linalg.norm(ffd - Delta_target) ** 2)

        rng      = np.random.default_rng(42)
        best_val = np.inf
        best_c   = V[:, 0]

        for _ in range(n_tries):
            x0  = rng.standard_normal(n_deg)
            x0 /= np.linalg.norm(x0)
            res  = minimize(residual, x0, method='COBYLA',
                            options={'maxiter': 1000, 'rhobeg': 0.3})
            if res.fun < best_val:
                best_val = res.fun
                best_c   = res.x / (np.linalg.norm(res.x) + 1e-300)

        best_phi  = V @ best_c
        best_phi /= np.linalg.norm(best_phi)
        imp.update_eig_vec(best_phi)

        F2_after = float(np.linalg.norm(imp.ffdagger - Delta_target, 'fro'))
        print(f"    [degen] F2: {np.sqrt(best_val):.3e} → (update後) {F2_after:.3e}")


    def cost_func_Delta(self,xinit):
           lmbda0 = xinit[0:self.nqspo*(self.nqspo+1)//2]
#           self.mu_fermi = 0. #xinit[-1]

           self.Lmbda = cr.realHcombination(lmbda0,self.H_list)


           self.calc_Delta()
           self.Delta = self.Delta[0:self.nqspo,0:self.nqspo]

           if locdbg:
             print(f'<cd>    in microit \n{self.imp_solver.fdaggerc.T}')
             print(f'<dd>    in microit \n{self.imp_solver.ffdagger}')
             print(f'R        in microit \n{self.R}')
             print(f'Lambda   in microit \n{self.Lmbda}')
             print(f'Delta    in microit \n{self.Delta}')
             print(f'Trace    in microit \n{np.trace(self.Delta)-1}')
             print(f'mu_fermi in microit \n{self.mu_fermi}')

           # Compute F1 and F2
           self.F1 = self.n - (np.trace(self.Delta)-1)
           self.F2 = self.imp_solver.ffdagger - self.Delta

           self.F1_max = np.abs(self.F1).max()
           self.F2_max = np.abs(self.F2).max()

           # Compute norm of F1 and F2 for convergence check 
           self.F1nrm = self.F1_max #np.linalg.norm(np.power(self.F1,1),'fro')
           self.F2nrm = np.linalg.norm(np.power(self.F2,1),'fro')

           if locdbg:
             print(f'F1    in microit\n{self.F1}')
             print(f'F2    in microit\n{self.F2}')

           self.microitF1F2 += 1
           print(f"    F1, F2 Micro-It. {self.microitF1F2}: F1, F2= ", self.F1_max, self. F2_max)

           # Issue④修正: F1（粒子数拘束）をコスト残差に含める
           # F1 = n - (Tr(Δ)-1) はスカラー。含めることで Λ の最適化が
           # F2=0 だけでなく粒子数保存も同時に満たす解を探す。
           return np.hstack((cr.inverse_realHcombination(self.F2,self.H_list), [self.F1]))


    def optimize_selfc_offdiag(self, muinit=0.0, n_offdiag=5, verbose=True):
        """
        Issue③修正: 物理–ゴースト off-diagonal Λ を含む複数の初期条件で
        optimize_selfc を試し、最小 F2 の解を返すマルチスタートソルバー。

        n_offdiag: off-diagonal 初期化を何種類試すか (1〜5)
        """
        import copy

        nqspo = self.nqspo
        nhl   = nqspo * (nqspo + 1) // 2

        # ---------- 候補初期値リスト ----------
        li_diag = np.zeros(nhl); li_diag[0] = 1.0
        ri_phys = np.zeros(nqspo); ri_phys[0] = 1.0
        ri_eq   = np.array([0.8124, 0.4124, 0.4124]) if nqspo == 3 else ri_phys.copy()
        ri_eq  /= np.linalg.norm(ri_eq)

        candidates = [('diagonal', li_diag.copy(), ri_phys.copy())]

        if nqspo >= 3 and n_offdiag > 0:
            # H_list index 3 = Λ_{01}/√2 (物理-ゴースト1結合)
            # H_list index 4 = Λ_{02}/√2 (物理-ゴースト2結合)
            # H_list index 5 = Λ_{12}/√2 (ゴースト1-ゴースト2結合)
            for strength in [0.5, 1.0]:
                li = np.zeros(nhl); li[3] = strength; li[4] = strength
                candidates.append((f'phys-ghost±{strength}', li.copy(), ri_phys.copy()))
            if n_offdiag >= 3:
                # 対角 + off-diagonal 混合
                li = np.zeros(nhl); li[0] = 0.5; li[2] = -0.5
                li[3] = 0.4; li[4] = 0.4
                candidates.append(('mixed-diag+off', li.copy(), ri_phys.copy()))
            if n_offdiag >= 4:
                # ゴースト成分を持つ R と off-diagonal Λ
                li = np.zeros(nhl); li[3] = 0.6; li[4] = 0.6
                candidates.append(('ghost-R+off', li.copy(), ri_eq.copy()))
            if n_offdiag >= 5:
                # 符号反転
                li = np.zeros(nhl); li[3] = -0.5; li[4] = -0.5
                candidates.append(('phys-ghost-neg', li.copy(), ri_phys.copy()))

        best_F2nrm = np.inf
        best_attrs = None

        for label, li, ri in candidates[:1 + n_offdiag]:
            if verbose:
                print(f'\n  [multistart] 初期化: {label}  '
                      f'li={np.round(li[:min(nhl,6)], 3)}')
            try:
                self.optimize_selfc(rinit=ri.copy(), lambdainit=li.copy(), muinit=muinit)
            except Exception as e:
                if verbose:
                    print(f'  [multistart] 収束失敗: {e}')
                continue

            f2 = float(self.F2nrm)
            eig_L = np.linalg.eigvalsh(np.real(self.Lmbda))
            if verbose:
                print(f'  [multistart] F2nrm={f2:.3e}  Λ_eig={np.round(eig_L, 3)}')

            if f2 < best_F2nrm:
                best_F2nrm = f2
                best_attrs = {k: copy.deepcopy(v)
                              for k, v in vars(self).items()
                              if isinstance(v, (np.ndarray, float, int, bool))
                              and not k.startswith('_')}

        if best_attrs is not None:
            for k, v in best_attrs.items():
                setattr(self, k, v)
            if verbose:
                print(f'\n  [multistart] 採用: F2nrm={best_F2nrm:.3e}  '
                      f'Λ_eig={np.round(np.linalg.eigvalsh(np.real(self.Lmbda)), 3)}')

        return best_F2nrm


#mf new Yongxin solver







    def calc_Z(self,Lmbda,R):
        # Compute quasi-particle weight Z
        # General formula for arbitrary nqspo:
        #   numerator = (Σ_i r_i^2 * Π_{j≠i} λ_j)^2
        #   denominator = Σ_i r_i^2 * Π_{j≠i} λ_j^2
        # For nqspo=1: Z = R^2 (recovered automatically via empty product = 1)
        R_orig, Lmbda_orig = self.fix_gauge(R,Lmbda)

        if locdbg:
          print(f"R in Z: \n{R_orig}")
          print(f"Lmbda in Z: \n{Lmbda_orig}")

        lam = np.array([Lmbda_orig[i,i] - self.mu_fermi for i in range(self.nqspo)])
        r   = np.array([R_orig[i,0] for i in range(self.nqspo)])

        numerator   = 0.0
        denominator = 0.0
        for i in range(self.nqspo):
          prod1 = np.prod([lam[j]    for j in range(self.nqspo) if j != i])
          prod2 = np.prod([lam[j]**2 for j in range(self.nqspo) if j != i])
          numerator   += r[i]**2 * prod1
          denominator += r[i]**2 * prod2

        # Λ→0 極限（弱相関・U≈0）: 分母が数値的にゼロになる場合は Z = Tr(RR†) = Σ rᵢ²
        # Mott 絶縁体 (真の Z=0) と区別するため、r が有意かどうかで判断
        if abs(denominator) < 1e-30:
          r_norm2 = float(np.sum(r**2))
          if r_norm2 < 1e-10:
            return 0.0   # 真の Mott 絶縁体: R≈0
          return r_norm2  # 弱相関極限: Z = Tr(RR†)

        return numerator**2 / denominator



    def make_Hqp(self,x):
        Roprod = np.dot(self.R,self.R.T)
        Hqp      = np.zeros((self.nqspo,self.nqspo))
        Hqp      = Roprod*x*alpha + self.Lmbda - self.mu_fermi*np.eye(self.nqspo)

        return Hqp


    # Compute uncorrelated one-electron density Delta
    def calc_Delta(self):
        def integrand(x,idx,jdx):
                Hqp = self.make_Hqp(x)

                f = self.dos(x)*cr.calc_C(Hqp,T=self.T)[idx,jdx]

                return f
    
        Roprod = np.zeros((self.nqspo,self.nqspo))
        Roprod[0:self.nqspo,0:self.nqspo] = np.dot(self.R,self.R.T)
    
        Delta = np.zeros((self.nqspo,self.nqspo))


        if lksum:
            Roprod = np.dot(self.R,self.R.T)
            Hqp_tmp = self.Lmbda - self.mu_fermi*np.eye(self.nqspo)
            for ek in self.eks:
              Hqp = Roprod*ek*alpha + Hqp_tmp
              f = cr.calc_C(Hqp,T=self.T)
              for i in range(self.nqspo):
                  for j in range(i,self.nqspo):
                    Delta[i,j] += f[i,j]

            for i in range(self.nqspo):
              for j in range(i,self.nqspo):
                Delta[j,i] = Delta[i,j]
            Delta /= len(self.eks)
        else:
            for i in range(self.nqspo):
                for j in range(i,self.nqspo):

                    if lksum:
                      integral = 0.0
                      for  ek in self.eks:
                        integral += integrand(ek, i, j)/len(self.eks)
                    else:
                      integral,error = integrate.quad(integrand, -1, 1,args=(i,j),epsabs=1e-14)

                    Delta[i,j] = integral
                    Delta[j,i] = integral
        self.Delta  = Delta
        self.nu     = 0.0   # In the Hubbard model we do not have an uncorrelated band


    # Compute Lagrange multipliers D
    def calc_D(self):
        def integrand(x,idx):
            Hk       = np.zeros((self.nqspo,self.nqspo))
            Hqp      = np.zeros((self.nqspo,self.nqspo))
            Rt       = np.zeros((self.nqspo,self.nqspo))
            Lambdat  = np.zeros((self.nqspo,self.nqspo))

            Rt[0:self.nqspo,0]         = np.copy(self.R[:,0])

            # Set up Hk
            Hk[0,0]                              = alpha*x

            # Compute Hqp
            Hqp = self.make_Hqp(x)

            resl = Hk.dot(Rt.T)
            resr = cr.calc_C(Hqp,T=self.T)
            f = self.dos(x)*resl.dot(resr)[0,idx]

            return f

        Left = np.zeros((self.nqspo))
        for i in range(self.nqspo):

            if lksum:
              integral = 0.0
              for  ek in self.eks:
                integral += integrand(ek,i)/len(self.eks)
            else:
              integral,error = integrate.quad(integrand, -1, 1,args=(i,),epsabs=1e-14)

            Left[i] = integral

        Left = np.reshape(Left,(self.nqspo,1))
        Right = cr.funcMat(self.Delta, cr.denR)

        self.D = np.dot(Right,Left)
       

    def calc_Eqp(self):
    # Compute quasi-particle energy
        def integrand(x):
            Hk       = np.zeros((self.nqspo,self.nqspo))
            Rt       = np.zeros((self.nqspo,self.nqspo))
            Lambdat  = np.zeros((self.nqspo,self.nqspo))


            # Set up R-tilde
            Rt[0:self.nqspo,0]         = np.copy(self.R[:,0])

            # Set up Hk
            Hk[0,0]                    = alpha*x

            # Compute Hqp
            Hqp = self.make_Hqp(x)

            resl = Rt.dot(Hk.dot(Rt.T))
            resr = cr.calc_C(Hqp,T=self.T)

            f = self.dos(x)*np.trace(resl.dot(resr))

            return f

        if lksum:
          integral = 0.0
          for  ek in self.eks:
            integral += integrand(ek)
          integral /= len(self.eks)
        else:
          integral,error = integrate.quad(integrand, -1, 1,epsabs=1e-14)

        self.Eqp = 2.0*integral







######### Set up tight-binding and lattice to be used ########
clat = "Inf"
lksum = clat != "Inf"
eks = -99

if not lksum: print("Use semicircular DOS (infinite lattice)")

# ... (Lattice setup code omitted as it is skipped for clat="Inf") ...

######### Hamiltonan settings #########
nphysorb    = 2                 
nghost      = 4                 
shift_list  = (-1.0,)            
V           = 1.0               
alpha       = 0.0               
tij         = 1.0               
T           = 0.002             

######## Setting for Hubbard model ########
lhubbard  = True                          
if lhubbard:
  alpha = 1.0                             
  T     = 0.003                             

lmicroit_mu = False 
locdbg   = False    



print("################### PARAMETERS/SETTINGS OF THE CALCULATION #######################")

ntot_list   = (0.5,)
lmu_sweep = np.abs(ntot_list[0]-0.5)<1e-7

def calc_phase_diagram():
  U_start = 0.01
  U_end   = 2.0
  U_step  = 0.01 

  mu_start = 0.0
  mu_end = 0.0
  mu_step = 1.0 

  print("U_start, U_end, U_step", U_start, U_end, U_step)

  # For half-filling start from Mott phase (irep_start = 1) and perform a forward and backward
  irep_start = 1
  nrep = 3

  if nghost==0 or np.abs(ntot_list[0]-0.5) > 1e-7: 
    irep_start = 2
    nrep = 3

  # Loop over particle numbers
  for ntot in ntot_list:
    icnt = 0
    GA_list = []

    # Main Loop (Forward/Backward)
    for reploop in range(irep_start,nrep):

      print(f"------- REPETITION LOOP Nr. {reploop} --------")

      # Determine Sweep Direction
      is_backward = (reploop % 2 != 0)
      
      if not is_backward:
          U_list = np.arange(U_start,U_end+U_step,U_step) # Forward
      else:
          U_list = np.arange(U_start,U_end+U_step,U_step)[::-1] # Backward

      if np.abs(U_start-U_end) <= 1e-7: U_list = (U_start,)

      z_list = []
      nc_list = []

      # Loop over interaction strengths
      for U in U_list:

        if is_backward: mu_list = np.arange(mu_start,mu_end,mu_step)
        else: mu_list = np.arange(mu_start,mu_end,mu_step)[::-1]
        if np.abs(mu_start-mu_end) <= 1e-7: mu_list = (mu_start,)

        for mu in mu_list:
              calc_nqspo = (nphysorb + nghost) // 2

              # Initial Guess Strategy
              if icnt==0 or nghost==0:
                if is_backward and nghost>0 and np.abs(ntot-0.5)<1e-7:
                  # Mott insulator guess (generalized for arbitrary calc_nqspo)
                  if calc_nqspo == 1:
                      rinit = np.zeros(1)
                      lambdainit = np.zeros(1)
                  else:
                      rinit = np.zeros(calc_nqspo)
                      # 0.95 の重みをゴースト軌道数（calc_nqspo - 1）で等分して物理的に正規化
                      weight_per_ghost = 0.95 / (calc_nqspo - 1)
                      rinit[1:] = np.sqrt(weight_per_ghost)
                      nlambda = calc_nqspo * (calc_nqspo + 1) // 2
                      lambdainit = np.zeros(nlambda)
                      if nlambda > 2:
                          lambdainit[1] =  U/2
                          lambdainit[2] = -U/2
                else:
                  # Metallic guess
                  rinit = np.zeros(calc_nqspo); rinit[0] = 1.0
                  lambdainit = np.zeros(calc_nqspo * (calc_nqspo + 1) // 2)
                  muinit = 0.0

                if lmu_sweep: muinit = np.copy(mu)
              else:
                # Adiabatic switching (Use previous result)
                rinit = GA_list[icnt-1].R[:,0]
                lambdainit = cr.inverse_realHcombination(GA_list[icnt-1].Lmbda,GA_list[icnt-1].H_list)
                if not lmu_sweep: muinit = (GA_list[icnt-1].mu)

              # Log
              if not lmu_sweep:
                print ("Starting calculation for U, N = ",str("{:.4f}".format(U)),str("{:.4f}".format(ntot)))
              else:
                print ("Starting calculation for U, mu = ",str("{:.4f}".format(U)),str("{:.4f}".format(muinit)))
  
              # Initialize GA object
              GA_list.append(GA(U,nghost,nphysorb,n=ntot,T=T,lcanonical=True,tolconv=1e-3, eks=eks))
              if icnt>0: GA_list[icnt].mu = GA_list[icnt-1].mu  
              if lmu_sweep: muinit = mu

              # Solve GA equations
              GA_list[icnt].optimize_selfc_new(rinit,lambdainit,muinit)

              z_list.append(GA_list[icnt].Z)
              nc_list.append(GA_list[icnt].imp_solver.nc)

              # Results
              print ("Final Z:", GA_list[icnt].Z)
              
              if not lmu_sweep:
                if GA_list[icnt].lconverged_root:
                    print ("GA Converged for U, N = ",str("{:.4f}".format(U)),str("{:.4f}\n".format(ntot)))
                else:
                    print ("GA NOT converged for U, N = ",str("{:.4f}".format(U)),str("{:.4f}\n".format(ntot)))
              
              icnt += 1

      # --- プロット (ループの内側で実行することで、往路・復路の両方を描画) ---
      label_str = "U降順:絶縁体→金属" if is_backward else "U昇順:金属→絶縁体"
      plt.plot(U_list, z_list, marker='o', markersize=4, label=f"{label_str}")


  # --- ★修正点: 全ループ終了後に理論カーブを描画して show() する ---

  # Brinkman-Rice Theory (Bethe Lattice, Bandwidth W=2, D=1)

  D_code = 1.0
  Uc_exact2 = 2.94
  Uc_exact1 = 2.39

  # Theoretical curve: Z = 1 - (U/Uc)^2
  U_theory1 = np.linspace(0, Uc_exact1, 100)
  U_theory2 = np.linspace(0, Uc_exact2, 100)
  #Z_theory1 = 1.0 - (U_theory1 / Uc_exact1)**2 #
  #Z_theory2 = 1.0 - (U_theory2 / Uc_exact2)**2


  # Plot theory curve 
  # plt.plot(U_theory1,Z_theory1, 'k--', linewidth=2, label=f'Brinkman-Rice Theory ($U_c \\approx {Uc_exact1:.2f}$)')
  # plt.plot(U_theory2, Z_theory2, 'k--', linewidth=2, label=f'Brinkman-Rice Theory ($U_c \\approx {Uc_exact2:.2f}$)')
  plt.axvline(x=Uc_exact1, color='blue', linestyle=':', label='DMFTによる計算値$U_{c1}$')
  plt.axvline(x=Uc_exact2, color='red', linestyle=':', label='DMFTによる計算値$U_{c2}$')
  # Graph Formatting
  plt.xlabel("U")
  plt.ylabel("純粒子重み $Z$")
  plt.title("半バンド幅 $D$ ($D=1$),Bethe格子")
  plt.xlim(0, 5.0)
  plt.ylim(-0.05, 1.05)
  plt.legend()
  plt.grid(True)
  plt.show()
if __name__ == '__main__':
  calc_phase_diagram()