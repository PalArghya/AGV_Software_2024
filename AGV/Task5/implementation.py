import numpy as np
import cv2
from helper import _singularize, displayEpipolarF
import scipy as sp
class Reconstruction:
    img1 = cv2.imread('3D/im1.png')
    img2 = cv2.imread('3D/im2.png')
    def eight_point(pts1, pts2, M):
        data = np.load("3D/some_corresp.npz")
        pts1 = data['pts1']
        pts2 = data['pts2']
        norm_pts1 = pts1/M
        norm_pts2 = pts2/M
        X1 = norm_pts1[:, 0]
        Y1 = norm_pts1[:, 1]
        X2 = norm_pts2[:, 0]
        Y2 = norm_pts2[:, 1]
        A = np.column_stack((X1*X2,X1*Y2,X1,Y1*X2,Y1*Y2,Y1,X2,Y2,np.ones(len(pts1))))
        _,_,V = np.linalg.svd(A)#finding the right singular vector transpose of A   
        F = np.reshape(V[-1, :], (3, 3))#fundamental matrix
        _F = _singularize(F)
        T = np.identity(3) / M
        T[2, 2] = 1
        _F = np.dot(np.dot(np.transpose(T), F), T)
        pts1 = pts1 * M
        pts2 = pts2 * M
        '''displayEpipolarF(img1, img2, _F)'''
        return _F, pts1, pts2
    def epipolar_correspondences(im1, im2, F, pts1):
        size = 45
        sigma = 25
        filter = sp.ndimage.gaussian_laplace(np.ones((size, size)), sigma) #creating a filter kernel for processing image
        filter = filter/np.sum(filter) #normalizing filter kernel
        kernel = np.dstack((kernel, kernel, kernel)) #length, breadth, channels
        # Applying Laplacian of Gaussian filter
        height, width, _ = im2.shape
        vect = np.array([width, height, 1])
        l = F.dot(vect)/(np.sqrt(l[0] ** 2 + l[1] ** 2))
        x1 = -(l[1] * y1 + l[2]) / l[0]
        x2 = -(l[1] * y2 + l[2]) / l[0]
        y1 = 0
        y2 = height - 1
        delta = 0.0
        max = max(x1-x2, y1-y2)
        x2_list = np.rint(np.linspace(x1, x2, max))
        y2_list = np.rint(np.linspace(y1, y2, max))
        k_half = size //2
        k_half__ = (size-1) // 2
        x2_delta=0.0
        y2_delta=0.0
        if x1 >= k_half and y1 >= k_half and x1 <= y2-k_half__ and y1 <= y2-k_half__:
            patch_1 = im1[y1 - k_half: y1 - k_half + size, x1 - k_half: x1 - k_half + size, :]
            patch_1 = np.asarray(patch_1)
            for i in range(x2_list.shape[0]):
                x2 = x2_list[i]
                y2 = y2_list[i]
                if x2 >= k_half and y2 >= k_half and x2 <= sx-1-k_half__ and y2 <= sy-1-k_half__:
                    diff_gaussian = np.multiply(kernel, diff)
                    error = np.linalg.norm(diff_gaussian)
                    patch_2 = im2[y2-k_half: y2-k_half+size, x2-k_half: x2-k_half+size, :]
                    patch_2 = np.asarray(patch_2)
                    diff = patch_1 - patch_2
                    if delta>error:
                        delta = error
                        x2_delta = x2
                        y2_delta = y2
        return x2_delta, y2_delta
    def EssentialMatrix(F, K1, K2):
        E= K1.transpose().dot(F).dot(K2)
        return E
    def Triangulate(P1,P2, pts1, pts2)
    