"""
Annotated implementation of Manifold Inference from Neural Dynamics (MIND)
as introduced in:

Low, R. J., Lewallen, S., Aronov, D., Nevers, R. & Tank, D. W. 
Probing variability in a cognitive map using manifold inference from neural dynamics. 
Biorxiv 418939 (2018) doi:10.1101/418939.

@author Quentin RV. Ferry
"""

#/////////////////////////////////////////////////
# Imports
#/////////////////////////////////////////////////

# miscellaneous
import time
# linear algebra
import numpy as np
# gradient descent optimization
import autograd.numpy as agnp
from autograd import grad
from scipy.optimize import minimize
# mutilvariate normal & shortest path algorithm
from scipy.stats import multivariate_normal
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
# Embedding (dimensionality reduction)
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
# state -> embedding and reverse mapping
from sklearn.model_selection import train_test_split

#/////////////////////////////////////////////////
# Helper functions
#/////////////////////////////////////////////////

def probabilistic_pca(X, M, axis_feature, generate_samples = 0):
    
    """
    Computes probabilistic PCA as introduced in Tipping and Bishop 1997.
    The code fits the data with a multivariate normal generative model with mean ppca_mu and covariance ppca_C.  

    INPUTS:
    - X: NxD (axis_feature = 1) or DxN (axis_feature = 0) matrix of N D-dimensional observation x_i.
    - M: dimensionality of the principal component basis used for modeling (M <= D).
        If M is None, it is infered form the eigen values of the cov. matrix to capture 95% of variance.
    - axis_feature: indicates which axis encodes the features. 
        If X is NxD then axis_feature = 1.
        If X is DxN then axis_feature = 0.
    - generate_samples: will generate Xsyn, k samples generated using the ppca generative model.
        Set to 0 to skip Xsyn generation (default = 0).

    OUTPUTS:
    - ppca_mu mean for p(x)
    - ppca_C covariance matrix for p(x)
    - Xsyn if generate_samples > 0, generate generate_samples samples from the model.
    """
    
    # make sure that X is DxN
    if axis_feature == 1:
        X = np.transpose(X)
    D, N = X.shape
        
    # perform eigen decomposition of covariance matrix
    S = np.cov(X) # get covariance matrix
    eig_values, eig_vectors = np.linalg.eig(S) # eigen decomposition of S. eig_vectors[:,i] is unit eig vector corresponding to eig_values[i]
    # sort eigen values/vectors by decreasing eigen value
    idx_argsort = np.flip(np.argsort(eig_values)) 
    eig_values = eig_values[idx_argsort]
    eig_vectors = eig_vectors[:, idx_argsort]

    # determine M if not passed. M would at maximum be equal to D-1.
    if M is None:
        M = min(np.sum(np.cumsum(eig_values / np.sum(eig_values)) < 0.95) + 1, D - 1)
        #print(f'selecting M = {M} to cover 0.95% variance, eig_v = {eig_values}')
    
    # estimate W, mu, sigma square, and C
    ppca_mu = np.mean(X, axis = 1) # get mu
    ppca_sigma2 = np.sum(eig_values[M:])/(D-M) # get sigma square
    ppca_W = np.matmul(eig_vectors[:,:M],(np.diag(eig_values[:M]) - ppca_sigma2 * np.eye(M))**0.5) # get W
    ppca_C = np.matmul(ppca_W, np.transpose(ppca_W)) + ppca_sigma2 * np.eye(D) # get C
    
    # generate new samples from model
    if generate_samples > 0:
        Xsyn = np.random.multivariate_normal(ppca_mu, ppca_C, size = (generate_samples,))
        if axis_feature == 0: # make sure that the generated samples follow the same structure as X.
            X_syn = np.transpose(Xsyn)
    else:
        Xsyn = None
        
    return ppca_mu, ppca_C, Xsyn

def key_to_int(key):
    """
    Converts node keys (list of integers, e.g. [x4, x3, x2, x1]) into a unique integer using 
    Goedel's trick: key_int = sum(pi**xi), where pi is the ith prime number.
    """
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,
              73,79,83,89,97,101,103,107,109,113,
              127,131,137,139,149,151,157,163,167,173,
              179,181,191,193,197,199,211,223,227,229,
              233,239,241,251,257,263,269,271,277,281] # needs to be extended to accommodate deeper trees 
    key_int = 0
    for i, k in enumerate(reversed(key)):
        key_int += primes[i]**k
    return key_int

#/////////////////////////////////////////////////
# Classes
#/////////////////////////////////////////////////

class PPCA():
    """
    Object storing the mean and variance obtained from a specific ppca.

    ATTRIBUTES:
    - mu: mean of the mutivariate normal.
    - C: covariance matrix of the mutivariate normal.

    METHODS:
    - init
    """
    def __init__(self, mu, C):
        self.mu = mu
        self.C = C

class Node():
    """
    Object representing a single node in a MIND tree.

    ATTRIBUTES:
    - key: list of integers working as a unique key for this node. Assigned by parent node.
    - ppca: PPCA object for this node (computed in the parent node). Assigned by parent node.
    - v: best partition vector to create left/right children nodes. Computed by 'split' method.
    - c: best partition threshold to create left/right children nodes. Computed by 'split' method.
    - left_child/right_child: left/right children node objects. Created in 'split' method.

    METHODS:
    - init
    - split: Starting from a set of n_v partition vectors and n_c partition threshold,
        finds the optimal v and c to parition the data associated with this node, and
        generates the corresponding left and right children nodes.
    - partition_data: Recursively assigns each sample of X to its corresponding leaf in the tree.
    - fetch_ppca: Recursively fetches all leaf ppca models.
    - print_structure: Recursively display all nodes in the tree in a hierarchical fashion.
    """
    def __init__(self):
        self.key = None 
        self.ppca = None
        self.v = None
        self.c = None
        self.left_child = None
        self.right_child = None
        
    def split(self, X, Xp, nb_leaf, nb_v, nb_c):
        
        """
        Splits X (data corresponding to this node) into X_left and X_right based on partition parameters v, c
        such that X_left = {x_i, x_i . v < c}. Parameters v and c are optimize to maximize the combined log likelihood
        of ppca_left(X_left) and ppca_right(X_right), where ppca_x is a generative trained on x.

        INPUTS:
        - X: DxN matrix containing N D-dimensional states corresponding to x_{t}.
        - Xp: DxN matrix containing N D-dimensional matching successor states corresponding to x_{t+1}.
        - nb_leaf: min number of states in a node below which the node is considered a leaf and won't be split.
        - nb_v: number of random candidates {v}.
        - nb_c: number of random candidates {c}.

        OUTPUTS:
        - best_v: optimal v according to log likelihood.
        - best_c: optimal c according to log likelihood.
        - left/right nodes and their key, ppca.
        """

        assert X.shape == Xp.shape, "[Node.split] X and Xp have incompatible shapes."
        D, N = X.shape # get dimension and number of samples in the current node.
        
        if N > 2 * nb_leaf: # only split if there are enough samples in the current node.
            
            # make {v}, nb_v unitary vectors
            V = np.random.uniform(-1., 1., (nb_v, D))
            V_norm = V / np.linalg.norm(V, axis = 1).reshape((-1,1)) # ensure unitary vectors

            # compute min quantile allowed ({c} are taken to represent certain quantile of X's projection onto v)
            min_quantile =  nb_leaf / N

            best_v = None # stores best v so far
            best_c = None # stores best c so far
            best_left_ppca = None # stores best ppca for left node so far
            best_right_ppca = None # stores best ppca for right node so far
            max_llhd = - np.Inf # stores maximum log likelihood so far

            for idx_v in range(nb_v): # for each v in {v}

                v = V_norm[idx_v,:] # get current v
                x_vProj = (v.reshape((1,-1)) @ X).reshape((-1,)) # project data X along v                
                C = np.quantile(x_vProj, np.linspace(min_quantile, 1-min_quantile, nb_c)) # create set of {c} using quantiles

                for idx_c in range(nb_c): # for each c in {c}
                    c = C[idx_c] # get current c
                    # partition Xp data
                    left_idx = x_vProj < c
                    Xp_left = Xp[:,left_idx]
                    Xp_right = Xp[:,~left_idx]
                    # compute PPCA over Xp data for each half
                    ppca_mu_left, ppca_C_left, _ = probabilistic_pca(Xp_left, M = None, axis_feature = 0)
                    ppca_mu_right, ppca_C_right, _ = probabilistic_pca(Xp_right, M = None, axis_feature = 0)                    
                    # compute log likelihoods over Xp data for each half
                    llhd_left = np.sum(multivariate_normal.logpdf(np.transpose(Xp_left), 
                                                                  mean = ppca_mu_left, 
                                                                  cov = ppca_C_left))
                    llhd_right = np.sum(multivariate_normal.logpdf(np.transpose(Xp_right), 
                                                                  mean = ppca_mu_right, 
                                                                  cov = ppca_C_right))
                    # save v, c yielding best log lilekihood
                    llhd = llhd_left + llhd_right
                    if llhd > max_llhd:
                        max_llhd = llhd
                        best_v = v
                        best_c = c
                        best_left_ppca = PPCA(ppca_mu_left, ppca_C_left)
                        best_right_ppca = PPCA(ppca_mu_right, ppca_C_right)

            # split X and Xp based on best v and c
            x_vProj = (best_v.reshape((1,-1)) @ X).reshape((-1,))
            left_idx = x_vProj < best_c
            X_left = X[:,left_idx]
            X_right = X[:,~left_idx]
            Xp_left = Xp[:,left_idx]
            Xp_right = Xp[:,~left_idx]

            # set v, c for current node
            self.v = best_v
            self.c = best_c

            # create left and right node. Set their key, ppca.
            self.left_child = Node()
            self.left_child.key = [1] + self.key # create new key by appending '1' (for left) to parent's key
            self.left_child.ppca = best_left_ppca # add corresponding ppca

            self.right_child = Node()
            self.right_child.key = [2] + self.key  # create new key by appending '2' (for right) to parent's key
            self.right_child.ppca = best_right_ppca # add corresponding ppca

            # and run 'split' method on children
            self.left_child.split(X_left, Xp_left, nb_leaf, nb_v, nb_c)
            self.right_child.split(X_right, Xp_right, nb_leaf, nb_v, nb_c)
            
        else: # node is a leaf
            pass
            
    def partition_data(self, X):

        """
        Recursively assigns each sample of X to its corresponding leaf in the tree.

        INPUTS:
        - X: DxN matrix containing N D-dimensional states corresponding to x_{t}.

        OUTPUTS:
        - keys: list of key_int matching every sample to a corresponding leaf.
        """
        
        D, N = X.shape
        key_int = key_to_int(self.key)
        #print(f'node {self.key} => {key_int}')
        keys = key_to_int(self.key) * np.ones((N,))
        # use v and c to partition data
        if self.left_child: # the node has children
            
            x_vProj = (self.v.reshape((1,-1)) @ X).reshape((-1,))
            left_idx = x_vProj < self.c
            X_left = X[:,left_idx]
            X_right = X[:,~left_idx]
            
            #print(f'X_left:{X_left.shape}, X_right:{X_right.shape}')
            
            keys[left_idx] = self.left_child.partition_data(X_left)
            keys[~left_idx] = self.right_child.partition_data(X_right)
        
        return keys
    
    def fetch_ppca(self, dict_ppca):

        """
        Recursively fetches all leaf ppca models. They are stored in a dictionary 
        indexed by the key_int of the corresponding leaf.

        INPUTS:
        - dict_ppca: dict with {key_int:PPCA} pairs.

        OUTPUTS:
        - new_dict_ppca: dict with {key_int:PPCA} pairs updated with the info of the current node (if leaf).
        """
        
        new_dict_ppca = dict_ppca
        
        if self.left_child: # node is not a leaf
            # run the method on children
            new_dict_ppca = self.left_child.fetch_ppca(new_dict_ppca)
            new_dict_ppca = self.right_child.fetch_ppca(new_dict_ppca)
            
        else: # node is a leaf
            # update dictionary with the info of this node
            new_dict_ppca.update({key_to_int(self.key):self.ppca})
            
        return new_dict_ppca
        
    def print_structure(self, offset):

        """
        Recursively display all nodes in the tree in a hierarchical fashion.

        INPUTS:
        - offset: print offset. Get incremented at each level of the tree hierarchy.

        OUTPUTS: None.
        """
        
        print('.' * offset + f' key: {key_to_int(self.key)}')
        
        if self.left_child:
            self.left_child.print_structure(offset = offset + 3)
        if self.right_child:
            self.right_child.print_structure(offset = offset + 3)
            
class Tree():

    """
    Object representing a single MIND tree.

    ATTRIBUTES:
    - root: Node object represented the root node of the tree.
    - dict_ppca: dictionary containing all ppca models index by corresponding leaf's key_int.

    METHODS:
    - init
    - train_tree: partition state space and derive local generative models.
    - partition_data: Recursively assigns each sample of X to its corresponding leaf in the tree.
    - print_structure: Recursively display all nodes in the tree in a hierarchical fashion.
    """

    def __init__(self):
        self.root = None
        self.dict_ppca = None
        self._init_tree()
        
    def _init_tree(self):
        self.root = Node() # set root as node
        self.root.key = [0] # set root key
        self.dict_ppca = {} # clear dict_ppca

    def _fetch_ppca(self):
        """
        Uses the 'fetch_ppca' method from the Node class to recursively fill in dict_ppca.
        """
        self.dict_ppca = self.root.fetch_ppca(self.dict_ppca)
        
    def train_tree(self, X, Xp, nb_leaf, nb_v, nb_c):
        """
        Calls the 'split' method from the Node class to recursively partition space 
        and derive local ppca generative models. Once the split is completed, 
        the method '_fetch_ppca' is used to populate dict_ppca.

        INPUTS:
        - X: DxN matrix containing N D-dimensional states corresponding to x_{t}.
        - Xp: DxN matrix containing N D-dimensional matching successor states corresponding to x_{t+1}.
        - nb_leaf: min number of states in a node below which the node is considered a leaf and won't be split.
        - nb_v: number of random candidates {v}.
        - nb_c: number of random candidates {c}.

        OUTPUTS: None
        """
        self.root.split(X, Xp, nb_leaf, nb_v, nb_c)
        self._fetch_ppca()
        
    def partition_data(self, X):
        """
        Uses the 'partition_data' method from the Node class to 
        recursively assigns each sample of X to its corresponding leaf in the tree.
        """
        partition = self.root.partition_data(X)
        return partition.astype(int)
    
    def print_structure(self):
        """
        Uses the 'print_structure' method from the Node class to 
        recursively display all nodes in the tree in a hierarchical fashion.
        """
        if self.root:
            self.root.print_structure(offset = 3)
        else:
            print('root not yet assigned.')
            
class Forest():

    """
    Object representing a MIND forest. This contain high level computations at the forest level.

    ATTRIBUTES:
    - trees: list of Tree instances making up the forest.

    METHODS:
    - init
    - fit: train nb_trees Tree instances to make up a trained forest.
    - project: projects X onto a low dimensional space (dimensionality = n_components). 
    designed to match the pairwise distance computed in the state space.
    - mapping_parameter_tunning: determines the optimal K (number of nearest neighbors) and 
    alpha parameters for LLE mapping through gridsearch/cross-validation.
    - mapping: computes the image Yq (target space) of Xq (source space) using LLE.
    """

    def __init__(self, nb_trees):
        """
        Initialize Forest object. 

        INPUTS:
        - nb_trees: number of trees making up the forest.

        OUTPUTS: None
        """
        self.trees = [Tree() for i in range(nb_trees)]

    def __len__(self):
        """
        Implements the magic __len__ function for the Forest class.
        Call len(forest) to get the number of trees.
        """
        return len(self.trees)

    def _compute_distances(self, X, p_threshold = 1.e-10, min_nb_vote = 0.5):

        """
        Computes local and global distances between pairs of states in X.
        1) For each tree in the forest:
            - Partition X so as to assign each x to its corresponding leaf in the current tree.
            - For each pair (x1, x2), compute the probability of x2 being a successor of x1 as p(x2|x1) by using the 
            probability density function of the ppca of the leaf x1 belongs to (p(x2|leaf i)). All conditional probabilities
            computed through one generative model (same leaf) are normalized to sum up to 1. Note that p(x2|x1) is set to 0. if
            it is smaller than 'p_threshold'. All probabilities are stored in a nb_trees * nb_samples * nb_samples
            local distance matrix (ldm)
        2) A final measure of the local distances p(x2|x1) is obtained by taking the median over the values for each tree.
        Note that p(x2|x1) is set to 0. if there are less than min_nb_vote * nb_trees non negative entries for p(x2|x1).
        All final probability are converted into distances using the formula sqrt(-log(p)) and stored in a nb_samples * nb_samples
        local distance matrix (ldm). This matrix encode a directed graph between the samples.
        3) A global distance between x1 and x2 (d(x1->x2)) is computed as the shortest path linking x1 to x2 in the directed graph
        encoded by ldm. The final distance between x1 and x2 is computed as the average of d(x1->x2) and d(x2->x1).
        
        INPUTS:
        - X: DxN matrix containing N D-dimensional states corresponding to x_{t}.
        - p_threshold: min probability under which probability are considered equal to 0.
        default value = 1.e-10
        - min_nb_vote: min number of non-zero probaiblities across trees for p(x2|x2) to not be considered null (see explanation above).
        default value = 0.5

        OUTPUTS:
        - D: a NxN matrix containing the global pairwise distances between every two samples of X.
        """

        N = X.shape[1] # get number of samples
        ldm = np.nan * np.ones((len(self.trees), N, N)) # initialize local distance matrix

        for idx_tree, tree in enumerate(self.trees): # cycle through all trees of the forest
            
            print(f'... computing proba on tree #{idx_tree+1}     ', end = '\r')
            
            # partition X1 based on first tree
            partition = tree.partition_data(X)
            unique_keys = np.unique(partition)

            for idx_key in range(unique_keys.shape[0]):
                
                curr_key = unique_keys[idx_key]
                curr_ppca = tree.dict_ppca[curr_key]

                # get the proba for each X1
                probas = multivariate_normal.pdf(np.transpose(X), mean = curr_ppca.mu, cov = curr_ppca.C)
                probas = probas / np.sum(probas)
                probas[probas < p_threshold] = 0.

                idx = np.arange(N)
                idx_inLeaf = idx[partition == curr_key]
                idx_successors = idx[probas > 0.]
                proba_successor = probas[probas > 0.]
                    
                # populate ldm
                ii, jj = np.meshgrid(idx_inLeaf,idx_successors)
                ldm[idx_tree, ii, jj] = proba_successor.reshape((-1,1))
        
        #print('... computing proba completed.'+ ' ' * 50)

        # zero out all p(x2|x1) for which less than min_nb_vote * nb_trees entries are non-zero
        mask = ~(np.sum(~np.isnan(ldm), axis = 0) >= len(self.trees) * min_nb_vote)
        ldm[:, mask] = 0. 
        # compute an 'average' p(x2|x1) by taking the median over corresponding non-zero entries
        ldm = np.nanmedian(ldm, axis = 0)
        # convert probas to local distances
        ldm[ldm > 0.] = np.sqrt(-np.log(ldm[ldm > 0.]))
        # turn ldm into a graph and compute global distance using the dijkstra shortest path algorithm 
        graph = csr_matrix(ldm)
        D = dijkstra(csgraph=graph, directed=True)
        # make distances symmetrical
        D = (D + np.transpose(D))/2
        return D

    def fit(self, X, Xp, axis_feature, nb_leaf, nb_v, nb_c):
        """
        Trains nb_trees Tree instances to make up a trained forest to fit the data. 
        Calls the 'train_tree' methods on all Tree instances.

        INPUTS:
        - X: DxN or NxD matrix containing N D-dimensional states corresponding to x_{t}.
        - Xp: DxN or NxD matrix containing N D-dimensional matching successor states corresponding to x_{t+1}.
        - axis_feature: indicates which is the feature dimension: If X is NxD then axis_feature = 1, 0 corresponds to DxN.
        - nb_leaf: min number of states in a node below which the node is considered a leaf and won't be split.
        - nb_v: number of random candidates {v}.
        - nb_c: number of random candidates {c}.

        OUTPUTS: None.
        """
        if axis_feature == 1: # make sure that X and Xp have DxN shape (axis_feature = 0)
            X = np.transpose(X)
            Xp = np.transpose(Xp)

        for i, tree in enumerate(self.trees): # train each tree of the forest
            print(f'... training tree #{i+1}/{len(self.trees)}     ', end = '\r')
            tree.train_tree(X, Xp,
                            nb_leaf = nb_leaf, 
                            nb_v = nb_v,
                            nb_c = nb_c)
        print('... training completed'+ ' ' * 50)

    def project(self, X, axis_feature, n_components, 
        p_threshold = 1.e-10, 
        min_nb_vote = 0.5,
        mds_only = True):
        
        """
        Projects X onto a low dimensional space with n_components dimensions 
        designed to match the pairwise distance computed in the state space.
        
        INPUTS:
        - X: DxN or NxD matrix containing N D-dimensional states corresponding to x_{t}.
        - axis_feature: indicates which is the feature dimension: If X is NxD then axis_feature = 1, 0 corresponds to DxN.
        - n_components: dimensionality of the resulting embedding.
        - p_threshold: min probability under which probability are considered equal to 0.
        default value = 1.e-10
        - min_nb_vote: min number of non-zero probaiblities across trees for p(x2|x2) to not be considered null (see explanation above).
        default value = 0.5
        - mds_only: boolean indicating whether restricting the calculation of the coordinate to multidimensional scaling (default).
        if False, additional gradient-based optimization is performed.

        OUTPUTS:
        - X_emb: n_componentsxN or Nxn_components matrix containing N n_components-dimensional states corresponding to embedded x_{t}.
        """

        if axis_feature == 1: # ensuring that X is DxN
            X = np.transpose(X)
        
        time_start = time.time()
        
        # estimate global pairwise distances
        print('... computing D matrix...', end = '\r')
        D = self._compute_distances(X, p_threshold, min_nb_vote)
        print('... computing D matrix completed.')

        # computing embedding coordinates with multidimensional scaling
        print('... computing MDS...', end = '\r')
        mds = MDS(n_components = n_components,  dissimilarity='precomputed')
        X_emb = mds.fit_transform(D)
        print('... computing MDS completed.')

        if not mds_only: # perform further coordinate optimization with autograd and minimize
            
            print('... improving coordinates with gradient-based optimization:')
            def loss(x): # loss function to minimize
                eps = 1e-100
                n = D.shape[0]
                x = x.reshape((n,-1))
                dif = agnp.sqrt(agnp.sum((x[:,agnp.newaxis] - x[agnp.newaxis,:])**2, axis=-1)+eps)
                error = 1/(D+agnp.eye(n)) * (D-dif)**2 # diag prevents 1/0, not counted in triu
                return agnp.sum(agnp.triu(error, k=1))
            
            loss_grad = grad(loss) # get the derivative of the loss function

            # perform optimization
            # compute loss before
            loss_before = loss(X_emb)

            # i am not satisfied with minimize. It's taking too long
            # I will create my own function with adjustable lr.
            max_nb_iter = 100
            count_iter = 0
            lrs = [1e-1, 1e-2, 1e-3, 1e-4]
            curr_loss = np.Inf
            curr_loss_delta = - np.Inf

            while (count_iter < max_nb_iter and curr_loss_delta < -1e-2 ):

                count_iter += 1
                print(f'{count_iter}/{max_nb_iter} - delta loss = {curr_loss_delta:0.3f}', end = '\r')
                curr_gradient = loss_grad(X_emb)

                losses = np.zeros((len(lrs),))

                for idx_lr, lr in enumerate(lrs):
                    losses[idx_lr] = loss(X_emb - lr * curr_gradient)
                
                if losses.min() < curr_loss:
                    curr_loss_delta = losses.min() - curr_loss
                    curr_loss = losses.min()
                    X_emb = X_emb - lrs[np.argmin(losses)] * curr_gradient

            # x0 = agnp.copy(X_emb).flatten()
            # res = minimize(loss, x0=x0, jac=loss_grad)
            # X_emb = np.copy(res['x'].reshape(X_emb.shape))

            # compute loss after
            loss_after = loss(X_emb)
            print(f'...... new loss is {100 * loss_after / loss_before:0.2f}% of original loss.' + ' ' * 50)

        # add PCA to maximize variance around embedding dimensions
        X_emb = PCA().fit_transform(X_emb)

        print(f'... embedding completed in {time.time() - time_start:0.1f}s')

        if axis_feature == 0: # ensure that X_emb is return with the same convention as input X
            X_emb = np.transpose(X_emb)

        return X_emb
    
    def mapping_parameter_tunning(self, X, Y, axis_feature, fraction_test = 0.05, nb_sampling = 20, Ks = None, alphas = None):
        
        """
        Determines the optimal K (number of nearest neighbors) and alpha parameters for LLE mapping through gridsearch/cross-validation.

        INPUTS:
        - X: DxN or NxD matrix containing N D-dimensional x_i in state space.
        - Y: DxN or NxD matrix containing N D-dimensional corresponding y_i in embedding space.
        - axis_feature: indicates which is the feature dimension: If X is NxD then axis_feature = 1, 0 corresponds to DxN.
        - fraction_test: fraction of N to use as test set.
        - nb_sampling: number of time training and test sets are randomly sampled to estimate LLE accuracy for a given (K,alpha) tuple.
        - Ks: list of [K] to test for the gridsearch. If None, default set is used.
        - alphas: list of [alpha] to test for the gridsearch. If None, default set is used.

        OUTPUTS:
        - opt_K: optimal K for the LLE mapping.
        - opt_alpha: optimal alpha for the LLE mapping.
        """

        if axis_feature == 0: # ensure that X and Y are NxD (axis_feature = 1)
            X = np.transpose(X)
            Y = np.transpose(Y)

        # if Ks and/or alphas set are not provided, use default sets
        if Ks is None:  
            Ks = [5, 10, 15, 20, 25, 30]
        if alphas is None: 
            alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.]

        # place holders matrices for losses' means (optim_means), losses stds (optim_std)
        optim_means = np.zeros((len(Ks), len(alphas)))
        optim_std = np.zeros((len(Ks), len(alphas)))
        # place holders matrices to keep track of corresponding K (optim_Ks) and alpha (optim_alphas) being used
        optim_Ks = np.zeros((len(Ks), len(alphas))).astype('int')
        optim_alphas = np.zeros((len(Ks), len(alphas)))

        for idx_K, K in enumerate(Ks): # cycle through Ks

            print(f'... Tunning LLE parameters: {100 * (idx_K + 1)/len(Ks):0.2f}%      ', end = '\r')

            for idx_alpha, alpha in enumerate(alphas): # cycle through alphas
                
                cv_losses = np.Inf * np.ones(nb_sampling) # initiate cross validation losses

                for idx_cv in range(nb_sampling): # repeat sampling nb_sampling times

                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= fraction_test) # randomly split X and Y between train and test sets
                    N_train, N_test = X_train.shape[0], Y_test.shape[0] # get nb of samples in each set
                    dim_x, dim_y = X.shape[1], Y.shape[1] # get dimensionality in each space

                    # find knn of X_test in X_train
                    knn = NearestNeighbors(n_neighbors= K, algorithm='kd_tree').fit(X_train)
                    _, indices = knn.kneighbors(X_test)

                    # compute LLE Y predictions on test set 
                    Y_test_predicted = np.zeros((N_test, Y_train.shape[1]))

                    for i in range(N_test):

                        # get the x coordinates of knn for current query
                        knn_X = X_train[indices[i,:],:]
                        knn_Y = Y_train[indices[i,:],:]
                        # compute gram matrix
                        g = X_test[i,:] * np.ones((K, dim_x)) - knn_X
                        G = g @ g.T
                        # find optimal W
                        w = np.linalg.inv(G + alpha * np.eye(K)) @ np.ones((K,1))
                        w = w / np.sum(w)
                        # infer y
                        Y_test_predicted[i,:] = w.T @ knn_Y

                    # compute the test set loss
                    cv_losses[idx_cv] = np.sum(np.linalg.norm(Y_test - Y_test_predicted, axis = 1))

                # record stats for this (K, alpha) tuple
                optim_means[idx_K, idx_alpha] = np.mean(cv_losses)
                optim_std[idx_K, idx_alpha] = np.std(cv_losses)
                optim_Ks[idx_K, idx_alpha] = K
                optim_alphas[idx_K, idx_alpha] = alpha
        
        # select best parameter
        optim_means = optim_means.reshape((-1,))
        optim_std = optim_std.reshape((-1,))
        optim_Ks = optim_Ks.reshape((-1,))
        optim_alphas = optim_alphas.reshape((-1,))

        opt_K = optim_Ks[np.argmin(optim_means + optim_std)].item()
        opt_alpha = optim_alphas[np.argmin(optim_means + optim_std)].item()

        print(f'... Tunning LLE parameters completed. Selected K = {opt_K}, alpha = {opt_alpha}')
        return opt_K, opt_alpha
        
    def mapping(self, Xq, X, Y, axis_feature, K, alpha):

        """
        Compute the image Yq (target sapce) of Xq (source space) using LLE.

        INPUTS:
        - Xq: DxN or NxD matrix containing N D-dimensional queries in the source space.
        - X: DxN or NxD matrix containing N D-dimensional x_i in the source space.
        - Y: DxN or NxD matrix containing N D-dimensional corresponding y_i in target space.
        - axis_feature: indicates which is the feature dimension: If X is NxD then axis_feature = 1, 0 corresponds to DxN.
        - K: number of nearest neighbors to use for the LLE mapping.
        - alpha: alpha parameter to use for the LLE mapping.

        OUTPUTS:
        - Yq_predicted: mapping of Xq in the target space.
        """
        
        if axis_feature == 0: # ensuring Xq, X, Y have NxD shape
            Xq = np.transpose(Xq)
            X = np.transpose(X)
            Y = np.transpose(Y)

        N = Xq.shape[0] # get number of queries
        dim_x, dim_y = X.shape[1], Y.shape[1] # get source and target space dimensionality

        # find Knn of Xq in X
        knn = NearestNeighbors(n_neighbors= K, algorithm='kd_tree').fit(X)
        _, indices = knn.kneighbors(Xq)

        # infer Yq
        Yq_predicted = np.zeros((N, dim_y))

        for i in range(N):

            # get the x coordinates of knn for current query
            knn_X = X[indices[i,:],:]
            knn_Y = Y[indices[i,:],:]
            # compute gram matrix
            g = Xq[i,:] * np.ones((K, dim_x)) - knn_X
            G = g @ g.T
            # find optimal W
            w = np.linalg.inv(G + alpha * np.eye(K)) @ np.ones((K,1))
            w = w / np.sum(w)
            # infer y
            Yq_predicted[i,:] = w.T @ knn_Y

        if axis_feature == 0:
            Yq_predicted = np.transpose(Yq_predicted)
        
        return Yq_predicted

class MIND():

    """
    Wrapper class for the Forest Object. This is the class that most users will be interacting with
    when creating MIND embedding from data.

    ATTRIBUTES:
    - tree_nb_leaf: min number of states in a node below which the node is considered a leaf and won't be split.
    - tree_nb_v: number of random candidates {v}.
    - tree_nb_c: number of random candidates {c}.
    - X: NxD matrix containing N D-dimensional states corresponding to x_{t} (used to train MIND random forest).
    - Xp: NxD matrix containing N D-dimensional states corresponding to x_{t+1} (used to train MIND random forest).
    - emb_X: NxD matrix containing N D-dimensional states corresponding to x_{t} (used to compute embedding).
    - emb_Y: NxD matrix containing N D-dimensional states corresponding to x_{t} .
    - mapping_XY_params: [K, alpha] parameters for the forward mapping as derived through parameter tunning.
    - mapping_YX_params: [K, alpha] parameters for the reverse mapping as derived through parameter tunning.
    - forest: random forest of nb_tress trees each fitted to the data.

    METHODS:
    - init: initializes the MIND object. Creates the random forest and set key parameters.
    - fit: learns MIND embedding from data. 
        - Fits random forest to the data
        - Uses random forest and shortest path algorithm to computes global distances between state pairs.
        - Embeds data in a lower dimensional space.
        - Finds optimal LLE parameters to map novel data to the manifold and back.
    - transform: maps data from the state space onto the manifold (low-dimensional embedding space).
    - reverse_transform: maps data from manifold (low-dimensional embedding space) to the original state space.

    """

    def __init__(self, nb_trees, nb_leaf, nb_v, nb_c) -> None:
        """
        Initializes the MIND object. Creates the random forest and set key parameters.

        INPUTS:
        - nb_trees: number of trees making up the forest.
        - nb_leaf: min number of states in a node below which the node is considered a leaf and won't be split.
        - nb_v: number of random candidates {v} considered when partitioning the space at each node.
        - nb_c: number of random candidates {c} considered when partitioning the space at each node.

        OUTPUTS: None
        """
        # parameters for the tree
        self.tree_nb_leaf = nb_leaf # min nb of data points in each leaf
        self.tree_nb_v = nb_v # nb of random partition vectors to sample at each split
        self.tree_nb_c = nb_c # nb of random partition threshold to sample at each split
        # data
        self.X = None # states data
        self.Xp = None # next states data
        # data for embedding
        self.emb_X = None # states data
        self.emb_Y = None # states data
        # mapping
        self.mapping_XY_params = []
        self.mapping_YX_params = []
        # forest
        self.forest = Forest(nb_trees)
    
    def fit(self, X, Xp, 
        emb_dim, emb_fraction = 1.,
        p_threshold = 1e-10, fraction_vote = 0.5, mds_only = False):
        """
        learns MIND embedding from data: 
        - Fits random forest to the data.
        - Uses random forest and shortest path algorithm to computes global distances between state pairs.
        - Embeds data in a lower dimensional space.
        - Finds optimal LLE parameters to map novel data to the manifold and back.

        INPUTS:
        - X: NxD matrix containing N D-dimensional states corresponding to x_{t}.
        - Xp: NxD matrix containing N D-dimensional matching states corresponding to x_{t+1}.
        Note that if X is such that X[i,:] = x_{t} and X[i+1,:] = x_{t+1}, 
        then pass X = X[:-1,:] and Xp = [1:,:] as inputs.
        - emb_dim: dimensionality of the embedding space.
        - emb_fraction: fraction of all states (X) used to create the embedding (default = 1.).
         - p_threshold: min probability under which probability are considered equal to 0.
        default value = 1.e-10.
        - min_nb_vote: min number of non-zero probaiblities across trees for p(x2|x2) to not be considered null (see explanation above).
        default value = 0.5.
        - mds_only: boolean indicating whether restricting the calculation of the coordinate to multidimensional scaling (default).
        if False, additional gradient-based optimization is performed.

        OUTPUTS: 
        modifies the following class attributes:
        - self.X
        - self.Xp
        - self.emb_X
        - self.emb_Y
        - self.mapping_XY_params
        - self.mapping_YX_params
        """
        assert X.shape == Xp.shape, "[learning embedding] X and Xp do not have matching shapes."
        
        self.X, self.Xp = X, Xp # store X and Xp as class attributes
        N, D = self.X.shape # get number of state (N) and dimensionality (D) of state space.

        # fit forest to X and Xp
        print(f'>> Creating {len(self.forest.trees)} partitions (i.e. trees) of the state space.')
        self.forest.fit(self.X, self.Xp, axis_feature = 1, 
            nb_leaf = self.tree_nb_leaf, 
            nb_v = self.tree_nb_v, 
            nb_c = self.tree_nb_c)
        
        # subsample X to N * emb_fraction and embed into a emb_dim space
        print(f'>> Creating a {emb_dim}D embedding from {D}D data.')
        N_emb = int(N * emb_fraction)
        print(f'... using {100*emb_fraction}% of X ({N_emb} states)')
        self.emb_X = self.X[np.sort(np.random.choice(N, N_emb, replace = False)),:]
        self.emb_Y = self.forest.project(self.emb_X, axis_feature = 1,
            n_components = emb_dim,
            p_threshold = p_threshold,
            min_nb_vote = fraction_vote,
            mds_only = mds_only)

        # Find optimal LLE parameters for forward and reverse mappings
        print(f'>> Optimizing LLE mapping parameters')
        Ks = [5, 10, 15, 20, 25, 30]
        alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.]
        print(f'... gridsearch over default [K]: {Ks}')
        print(f'... gridsearch over default [alpha]: {alphas}')
        print('... optimizing for state (x) -> embedding (y) forward mapping')
        K, alpha = self.forest.mapping_parameter_tunning(self.emb_X, self.emb_Y, axis_feature = 1)
        self.mapping_XY_params = [K, alpha]
        print('... optimizing for embedding (y) -> state (x) reverse mapping')
        K, alpha = self.forest.mapping_parameter_tunning(self.emb_Y, self.emb_X, axis_feature = 1)
        self.mapping_YX_params = [K, alpha]

    def transform(self, X):

        """
        Maps data from the state space onto the manifold (low-dimensional embedding space).

        INPUTS:
        - X: NxD matrix containing N D-dimensional states (state space).

        OUTPUTS:
        - Y_predicted: Nxd matrix containing N d-dimensional corresponding state embeddings (states projected in the low-dimensional embedding),
        predicted by LLE.
        """

        Y_predicted = None

        if (self.emb_X is not None) and (self.emb_Y is not None) and (len(self.mapping_XY_params) > 0):
            Y_predicted = self.forest.mapping(X, self.emb_X, self.emb_Y, axis_feature =1, 
                K = self.mapping_XY_params[0], 
                alpha = self.mapping_XY_params[1])
        else:
            print('No embedding learnt yet. Please learn embedding first.')
        
        return Y_predicted

    def reverse_transform(self, Y):

        """
        Maps data from manifold (low-dimensional embedding space) to the original state space.

        INPUTS:
        - Y: Nxd matrix containing N d-dimensional state embeddings (states projected in the low-dimensional embedding).

        OUTPUTS:
        - X_predicted: NxD matrix containing N d-dimensional corresponding states (state space),
        predicted by LLE.
        """
        
        X_predicted = None

        if (self.emb_X is not None) and (self.emb_Y is not None) and (len(self.mapping_YX_params) > 0):
            X_predicted = self.forest.mapping(Y, self.emb_Y, self.emb_X, axis_feature =1, 
                K = self.mapping_YX_params[0], 
                alpha = self.mapping_YX_params[1])
        else:
            print('No embedding learnt yet. Please learn embedding first.')
        
        return X_predicted


           





        
        