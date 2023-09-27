import matplotlib.pyplot as plt
import numpy as np

def plotter3d(X,Y,Z):
    """Given vectors X,Y,Z, of size 1xn plot 3d plot of the 3 vectors

    :param X: 1st Vector
    :type X: np.array (1,n)
    :param Y: 2nd Vector
    :type Y: np.array (1,n)
    :param Z: 3rd Vector
    :type Z: np.array (1,n)
    """
    ax=plt.figure(figsize=(10,10)).add_subplot(projection='3d')
    ax.scatter(X[0], Y[0], Z[0])
    ax.plot3D(X, Y, Z)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.show()
