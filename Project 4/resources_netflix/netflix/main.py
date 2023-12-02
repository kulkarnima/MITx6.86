import numpy as np
import kmeans
import naive_em
import numpy as np
import common
import em

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")

mixture, posts = common.init(X, 12, seed=1)

mixture, posts, new_ll = em.run(X, mixture, posts)
X_pred = em.fill_matrix(X, mixture)

print(common.rmse(X_gold, X_pred))
