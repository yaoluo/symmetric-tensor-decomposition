
import numpy as np 

class rankOneCP_SymmetricT3:

   def __init__(self, A):
      self.A = A 
      self.n,_,_ = A.shape 
      return 
   #test case 
   def loss_func(self,z):
      df = self.A - np.tensordot(np.tensordot(z, z, axes=0), z, axes=0)
      return 0.5*np.sum(df**2) 
   def gradient(self,z):
      df = np.linalg.norm(z)**4 * z - np.tensordot(np.tensordot(self.A,z,axes=([2],[0])), z, axes=([1],[0]))
      return df 
   def optimal_init(self, L):
      n = self.n 
      w = np.random.random([L,n])-0.5
      z0 = 0
      for i in range(L):
         w[i] = w[i] / np.linalg.norm(w[i]) / np.sqrt(n)
         z0 = z0 + w[i] - n**2 * self.gradient(w[i])
      return z0 / L 

   def first_interation(self, z0):
      g0 = self.gradient(z0)
      z1 = z0 - g0 * 1e-5
      g1 = self.gradient(z1)
      self.g0 = g0 
      self.g1 = g1 
      self.z0 = z0 
      self.z1 = z1 
      return 
   
   def run(self, L = 100):
      z0 = self.optimal_init(L)
      self.first_interation(z0)
      loss = [] 
      for i in range(100):
         dz = self.z1 - self.z0 
         dg = self.g1 - self.g0 
         
         if( np.linalg.norm(dz)< 1e-14 and i>2):
            loss.append(self.loss_func(self.z1))
            return self.z1, loss  
         #long BB step 
         BB_stepsize = np.dot(dz,dz) / np.dot(dz,dg)
         #short BB step 
         #BB_stepsize = np.dot(dz,dg) / np.dot(dg,dg)

         z0 = self.z1 + 0 
         g0 = self.g1 + 0 
         z1 = z0 - g0 * BB_stepsize
         g1 = self.gradient(z1)

         #update g0, g1, z0, z1
         self.g0 = g0 + 0 
         self.g1 = g1 + 0 
         self.z0 = z0 + 0 
         self.z1 = z1 + 0 

         loss.append(self.loss_func(self.z1))
      return self.z1, loss  
   def update_A(self):
      #substract dominate mode 
      z = self.z1 
      self.A = self.A -  np.tensordot(np.tensordot(z, z, axes=0), z, axes=0)
      return 