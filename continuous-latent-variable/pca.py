import numpy as np

class PCA():
    def __init__(self):
        self.dataset = []

    def compute_principal(self,  x: np.ndarray):
        '''
        1. compute mean, mu
        2. compute covariance matrix, s
        3. spectral deocmposition
        4. analytically maximize variance by finding largest eigenvalue, i
        5. compute principal component, u  
        6. project data onto principal component and save
        '''
        mu = np.mean(x, axis=0)
        S = sum(np.outer(x[i]-mu , x[i]-mu) for i in range(x.shape[0])) / x.shape[0]
        eigval, eigvec = np.linalg.eigh(S)
        i = np.argmax(eigval)
        u = eigvec[:,i]
        y = x @ u

        return u, y

    def forward(self, x: np.ndarray, n_components: int=1):
        ''' 
        1. compute 1st principal
        2. remove 1st principal
        3. loop for 2nd to nth principal
            a. compute ith principal and save the vector
            b. prepare dataset for next principal
        ''' 
        components = []
        data = []
        
        u, y = self.compute_principal(x)
        u = u / np.linalg.norm(u, ord=2)
        components.append(u)

        for i in range(x.shape[0]):
            x[i] = x[i] - np.dot(x[i], u) * u
        
        for i in range(1, n_components):
            u, y = self.compute_principal(x)
            u = u / np.linalg.norm(u, ord=2)
            components.append(u)

            for i in range(x.shape[0]):
                x[i] = x[i] - np.dot(x[i], u) * u
        
        return [self.dataset @ component for component in components]