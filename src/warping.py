import cv2
import numpy as np

class ImageWarper:
    def __init__(self):
        pass

    def warp_images(self, img1, img2, H):
        """Warp and blend two images using the homography matrix.
        
        Args:
            img1 (numpy.ndarray): First image (source)
            img2 (numpy.ndarray): Second image (destination)
            H (numpy.ndarray): 3x3 homography matrix
            
        Returns:
            numpy.ndarray: Blended panoramic image
        """
        # Define output mosaic size
        out_height = max(img1.shape[0], img2.shape[0])
        out_width = img1.shape[1] + img2.shape[1]
        
        # Compute the inverse homography
        H_inv = np.linalg.inv(H)
        
        # Warp second image using inverse homography
        warped_img = cv2.warpPerspective(img2, H_inv, (out_width, out_height))
        
        # Overlay first image onto warped image
        panorama = warped_img.copy()
        h_src, w_src = img1.shape[:2]
        panorama[0:h_src, 0:w_src] = img1
        
        return panorama 