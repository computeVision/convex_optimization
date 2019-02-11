import numpy as np
import matplotlib.pyplot as plt
import pdb, sys, scipy.misc
rng = np.random.RandomState(12354)
sys.setrecursionlimit(10000)

from scipy import spatial
from scipy.sparse import identity, csr_matrix, vstack, hstack
import skimage.io as io

def plot_optimal_transport(Xs, Ys, PI, indices, energy, title='Primal Dual'):
    # plt.ion()
    plt.figure(10)
    plt.title("PI Matrix")
    plt.imshow(PI)
    plt.colorbar()
    plt.savefig('pi_mat.png')

    print('lol1')
    plt.figure(20)
    plt.subplot(2,1,1)
    PI_max = np.argmax(PI, axis=1)
    for i in range(PI.shape[0]):
        plt.plot([Xs[i, 0], Ys[PI_max[i],0]],[Xs[i,1], Ys[PI_max[i],1]], 'g--')
    plt.scatter(Xs[:,0], Xs[:,1], label='Source')
    plt.scatter(Ys[:,0], Ys[:,1], label='Target')
    plt.title('Optimal Transport - Argmax')
    plt.legend()
    plt.savefig('argmax.png')
    print('lol2')
    
    plt.subplot(2,1,2)
    plt.title('Optimal Transport - PI*Y')
    plt.scatter(Xs[:,0], Xs[:,1], label='Source')
    plt.scatter(Ys[:,0], Ys[:,1], label='Target')
    # for i in xrange(PI.shape[0]):
    #     for j in xrange(PI.shape[0]):
    #         plt.plot([Xs[i, 0], Ys[j,0]],[Xs[i,1], Ys[j,1]], 'g--', linewidth=PI[i,j]*40)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('piy.png')
    print('lol3')

    plt.figure(40)
    plt.plot(energy)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Energy')
    plt.grid()
    plt.savefig('energy.png')
    print('lol4')
    
    if not np.all(indices) == 0.0:
        plt.figure(1000)
        plt.imshow(Ys[PI_max[indices]])
        plt.savefig('transfered.png')
    plt.figure(50)
    plt.imshow(Ys[PI_max[indices]])
    plt.savefig('transfered.png')

    plt.show()


def create_points1(N):
    Xs = rng.multivariate_normal([-2,-2], [[1.0, 0],[0, 1.0]], size=N)
    Ys = rng.multivariate_normal([3,4], [[10.0, 0],[0, 1.0]], size=N)

    C = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            C[i,j] = np.linalg.norm(Xs[i] - Ys[j], 2)**2

    p = np.ones(N) / N
    q = np.ones(N) / N
    
    return Xs, Ys, C, p, q


def create_points(N, im_src, im_tar):

    h,w,c = im_src.shape
    Xs_cords = rng.randint(h*w, size=N)
    im_src_flatten = im_src.reshape(h*w,3)
    h,w,c = im_tar.shape
    Ys_cords = rng.randint(h*w, size=N)
    im_tar_flatten = im_tar.reshape(h*w,3)

    Xs = im_src_flatten[Xs_cords]
    Ys = im_tar_flatten[Ys_cords]
    
    C = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            C[i,j] = np.linalg.norm(Xs[i] - Ys[j], 2)**2
    
    p = np.ones(N) / N
    q = np.ones(N) / N
    
    return Xs, Ys, C, p, q


def primal_dual(N=50, iters=100, color_transfer=False):

    if color_transfer:
        # im_src = plt.imread('blue.png')[...,:3]
        # im_tar = plt.imread('orange.png')[...,:3]
        im_src = plt.imread('green.jpg')
        im_tar = plt.imread('orange.jpg')

        percent = 1.0
        im_src = scipy.misc.imresize(im_src, percent) / 255.
        im_tar = scipy.misc.imresize(im_tar, percent) / 255.
        Xs, Ys, C, p, q = create_points(N, im_src, im_tar)
    else:
        Xs, Ys, C, p, q = create_points1(N)

    def prox_sf(x, sigma, w):
        return x - sigma * w
    
    def prox_tg(x, tau, Cf):
        return np.maximum(x-tau*Cf, 0)

    PI = np.ones((N,N)) / N**2
    xk = PI.flatten()
    Cf = C.flatten()
    
    A = csr_matrix((N, N**2))
    for i in xrange(N):
        A[i, i*N:(i+1)*N] = 1.0

    B = csr_matrix((N, N**2))
    for j in xrange(0, N**2, N):
        B[:,j:(j+N)] = identity(N)

    K = vstack([A,B])
    L = 2*N**2 #scipy.linalg.norm(K)

    tau = 1 / scipy.sqrt(L)
    sigma = 1 / scipy.sqrt(L)
    
    y = np.zeros(2*N)
    w = np.hstack((p,q))
    
    print 'xk: ', xk.shape
    print 'K.T: ', K.T.shape
    print 'y: ', y.shape
    
    eps = 0.000001
    energy = []
    for i in xrange(iters):
        xk_1 = prox_tg(xk - tau*K.T.dot(y), tau, Cf)
        y = prox_sf(y + sigma*K.dot(2.*xk_1 - xk), sigma, w)
        xk = xk_1    
        energy.append((np.dot(Cf.T, xk_1)).sum())
        if i % 10000 == 0:
            print 'Iter: {}, Energy: '.format(i), energy[-1]
            print "PI: ", np.sum(xk_1)

        # early stopping
        tmp = xk.reshape((N,N))
        rows = np.abs(np.sum(tmp, axis=0) - 0.02) <= eps
        cols = np.abs(np.sum(tmp, axis=1) - 0.02) <= eps
        # print 'rows: ', rows
        # print 'cols: ', cols
        if i > 10 and np.all(rows) == True and np.all(cols) == True:
            print "<<<<<<<<<<<<< Early Stopping >>>>>>>>>>>>>>>>"
            break

    PI = xk_1.reshape(PI.shape)
    
    if color_transfer:
        tree = spatial.KDTree(Xs)
        nn, indices = tree.query(im_src)
        plot_optimal_transport(Xs, Ys, PI, indices, energy)
    else:
        plot_optimal_transport(Xs, Ys, PI, 0, energy)

#################### Sinkhorn-Knopp Algorithm ####################
def sinkhorn_knop(N=50, iters=100, lamda=0.001, color_transfer=True):
    if color_transfer:
        # im_src = plt.imread('blue.png')[...,:3]
        # im_tar = plt.imread('orange.png')[...,:3]
        im_src = plt.imread('green.jpg')
        im_tar = plt.imread('orange.jpg')

        percent = 1.0
        im_src = scipy.misc.imresize(im_src, percent) / 255.
        im_tar = scipy.misc.imresize(im_tar, percent) / 255.
        Xs, Ys, C, p, q = create_points(N, im_src, im_tar)
    else:
        Xs, Ys, C, p, q = create_points1(N)
    print Xs.shape

    M = np.exp(-(C / np.sum(C)) / lamda)
    vk = np.ones(N)
    
    print "lol: ", C.dtype, M.dtype, vk.dtype, p.dtype, type(q)

    eps = 0.000001
    energy = []
    for k in range(iters):
        uk = p / M.dot(vk)
        if  np.all(np.isnan(uk)):
            pdb.set_trace()
        vk = q / (M.T.dot(uk))
        if  np.all(np.isnan(vk)):
            pdb.set_trace()
        PI = np.diag(uk).dot(M).dot(np.diag(vk))
        
        
        energy.append((C * PI).sum() + lamda * (PI* np.log(PI)).sum())
        if k % 100:
            print energy[-1]

    if color_transfer:
        tree = spatial.KDTree(Xs)
        nn, indices = tree.query(im_src)
        plot_optimal_transport(Xs, Ys, PI, indices, energy, title="Sinkhorn Knopp")
    else:
        plot_optimal_transport(Xs, Ys, PI, 0, energy, title="Sinkhorn Knopp")

# primal_dual(N=50, iters=600000, color_transfer=False)
# primal_dual(N=10, iters=10000, color_transfer=False)
primal_dual(N=200, iters=400000, color_transfer=True)
# sinkhorn_knop(N=5, iters=100, lamda=0.0001, color_transfer=False)
# sinkhorn_knop(N=1000, iters=200, lamda=0.000000001)
# sinkhorn_knop(N=50, iters=80, lamda=0.00001,  color_transfer=False)
# sinkhorn_knop(N=30, iters=80, lamda=0.001,  color_transfer=False)