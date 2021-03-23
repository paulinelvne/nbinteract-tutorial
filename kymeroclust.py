import numpy as np
import pandas as pd
from numpy import argmin
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class KymeroClust:
    def __init__(self, nb_children, D, Gmax, H, nb_iter, distance):
        # parameters
        self.nb_children = nb_children
        self.Gmax = Gmax
        self.distance = distance
        self.H = H
        self.nb_iter = nb_iter
        # models
        self._reset_model(D)
        # dataset
        self.S = {}

    def _reset_model(self,D=None):
        if D is not None:
            self.G = D
            self.L = pd.DataFrame([np.zeros(D)],index=[0])
            self.W = pd.DataFrame([np.ones(D)],index=[0])
            self.components = []
        self.fitness = -1e1000

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1,-1)
        distances = pairwise_distances(X,
                                       self.L,
                                       metric=self.distance)
        membership = argmin(distances,axis=1)
        y = self.L.index[membership]
        return(y)

    def compute_fitness(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1,-1)
        distances = pairwise_distances(X,
                                       self.L,
                                       metric=self.distance)
        membership = argmin(distances,axis=1)
        return(-distances[range(distances.shape[0]), membership].sum())

    def _choose_random_cluster(self):
        cluster_sizes = self.W.sum(axis=1)
        if np.random.random() > self.G/(self.G+1):
            new_cluster = cluster_sizes.index.max()+1
        else:
            new_cluster = np.random.choice(cluster_sizes.index, p=cluster_sizes/self.G)
        return(new_cluster)

    def _insert(self,c,d,x):
        if c not in self.W.index:
            self.W.loc[c] = np.zeros(self.W.shape[1])
            self.L.loc[c] = np.zeros(self.L.shape[1])
        x_old = self.L.loc[c,d]
        self.L.loc[c,d] = x
        self.W.loc[c,d] += 1
        self.G += 1
        self.components.append((c,d))
        reverse_change = (c,d,x_old)
        return(reverse_change)

    def _delete(self,c,d,x=None):
        self.components.remove((c,d))
        self.G -= 1
        self.W.loc[c,d] -= 1
        x_del = self.L.loc[c,d]
        if x is not None:
            self.L.loc[c,d] = x
        if not self.W.loc[c,d]:
            self.L.loc[c,d] = 0
        if not self.W.loc[c].sum():
            self.W.drop(c,inplace=True)
            self.L.drop(c,inplace=True)
        reverse_change = (c,d,x_del)
        return(reverse_change)

    def _compute_child(self, X):
        reverse_del = None
        deletion = None
        if self.G >= self.Gmax:
            # remove component
            i_rand =  np.random.choice(len(self.components))
            c_del,d_del = self.components[i_rand]
            deletion = (c_del,d_del)
            reverse_del = self._delete(*deletion)
        # include new component
        x_id = np.random.choice(X.shape[0])
        d_add = np.random.choice(X.shape[1])
        xd = X[x_id][d_add]
        c_add = self._choose_random_cluster()
        insertion = (c_add, d_add, xd)
        reverse_ins = self._insert(*insertion)
        return(deletion,reverse_del,insertion,reverse_ins)

    def _compute_generation(self):
        deletion,reverse_del,insertion,reverse_ins = self._compute_child(self.S)
        new_fitness = self.compute_fitness(self.S)
        if reverse_ins is not None:
            self._delete(*reverse_ins)
        if reverse_del is not None:
            self._insert(*reverse_del)
        return(new_fitness,deletion,insertion)

    def fit(self, X, warm_start = False):
        if not warm_start:
            self._reset_model(X.shape[1])
        if X.shape[0] < self.H:
            self.H = X.shape[0]-1
        h = 0
        self.S = X[:self.H,:].copy()
        self.fitness = self.compute_fitness(self.S)
        self._fitnesses = []
        self._W_history = []
        self._L_history = []
        lazy = False
        i = self.H
        for _ in tqdm(range(self.nb_iter)):
            best_fitness = self.fitness
            best_changes = (None,None)
            for ch in range(self.nb_children):
                new_fitness,deletion,insertion = self._compute_generation()
                if new_fitness >= best_fitness:
                    best_changes = (deletion,insertion)
                    best_fitness = new_fitness
            if best_changes[0] is not None:
                self._delete(*best_changes[0])
            if best_changes[1] is not None:
                self._insert(*best_changes[1])
            self.fitness = best_fitness
            self._fitnesses.append(self.fitness)
            self._W_history.append(self.W.copy())
            self._L_history.append(self.L.copy())
            old_fitness = self.fitness
            self.fitness -= self.compute_fitness(self.S[h,:])
            self.S[h,:] = X[i,:]
            self.fitness += self.compute_fitness(self.S[h,:])
            h = (h+1)%self.H
            i = (i+1)%X.shape[0]

    def _plot_projection(self,X,x=0,y=1):
        y_pred = self.predict(X)
        X_full = pd.DataFrame(X)
        X_full["cluster"] =  ["clus. "+str(v) for v in y_pred]
        X_full["cluster"] = X_full["cluster"].astype('category')
        sns.scatterplot(x=x, y=y, hue='cluster', marker=".", data=X_full)
        sns.scatterplot(x=x, y=y, marker="o", data=c.L)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    def _cross_tab(self,X,y_true):
        y_pred = self.predict(X)
        res = pd.DataFrame([y_true,y_pred]).T
        return(pd.crosstab(y_true,y_pred))



c = KymeroClust(nb_children=5, D=448, Gmax=1000, H=200, nb_iter=5000, distance="l1")
c.fit(X)
