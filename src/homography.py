import numpy as np
import random

class HomographyEstimator:
    def __init__(self, threshold=5.0, iterations=1000):
        self.threshold = threshold
        self.iterations = iterations

    def compute_homography(self, src_pts, dst_pts):
        """Compute homography matrix from point correspondences.
        
        Args:
            src_pts (numpy.ndarray): Source points of shape (N,2)
            dst_pts (numpy.ndarray): Destination points of shape (N,2)
            
        Returns:
            numpy.ndarray: 3x3 homography matrix
        """
        assert src_pts.shape[0] >= 4, "At least four correspondences are required."
        N = src_pts.shape[0]
        A = []
        for i in range(N):
            x, y = src_pts[i]
            xp, yp = dst_pts[i]
            A.append([-x, -y, -1,  0,  0,  0, x*xp, y*xp, xp])
            A.append([ 0,  0,  0, -x, -y, -1, x*yp, y*yp, yp])
        A = np.array(A)
        
        # Solve Ah = 0 using SVD
        U, S, Vt = np.linalg.svd(A)
        h = Vt[-1, :]
        H = h.reshape(3, 3)
        H = H / H[2, 2]  # Normalize so that H[2,2]==1
        return H

    def estimate_homography(self, src_pts, dst_pts):
        """Estimate homography using RANSAC.
        
        Args:
            src_pts (numpy.ndarray): Source points of shape (N,2)
            dst_pts (numpy.ndarray): Destination points of shape (N,2)
            
        Returns:
            tuple: (H, inlier_indices) - Best homography matrix and indices of inliers
        """
        num_points = src_pts.shape[0]
        best_inlier_count = 0
        best_inliers = None
        best_H = None

        for _ in range(self.iterations):
            # Randomly sample 4 unique correspondences
            indices = random.sample(range(num_points), 4)
            src_sample = src_pts[indices]
            dst_sample = dst_pts[indices]

            try:
                H_candidate = self.compute_homography(src_sample, dst_sample)
            except np.linalg.LinAlgError:
                continue

            # Project all src_pts using the candidate homography
            src_homog = np.hstack([src_pts, np.ones((num_points, 1))])
            projected = (H_candidate @ src_homog.T).T
            projected = projected / projected[:, [2]]  # Normalize homogeneous coordinates

            # Compute reprojection errors
            errors = np.linalg.norm(projected[:, :2] - dst_pts, axis=1)
            inliers = np.where(errors < self.threshold)[0]
            inlier_count = len(inliers)

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_H = H_candidate

        # Re-estimate H using all inliers
        if best_inliers is not None and len(best_inliers) >= 4:
            best_H = self.compute_homography(src_pts[best_inliers], dst_pts[best_inliers])
        else:
            print("Warning: Not enough inliers found. Returning best candidate.")

        return best_H, best_inliers 