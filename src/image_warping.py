import numpy as np
import cv2

class ImageWarper:
    def __init__(self):
        pass

    def _get_corners(self, image):
        """Get corner coordinates of an image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of four (x, y) tuples in the order [top-left, bottom-left, top-right, bottom-right]
        """
        h, w = image.shape[:2]
        return [(0, 0), (0, h-1), (w-1, 0), (w-1, h-1)]

    def _warp_image(self, image, H, output_shape):
        """Warp an image using homography matrix.
        
        Args:
            image (numpy.ndarray): Input image
            H (numpy.ndarray): 3x3 homography matrix
            output_shape (tuple): Shape of output image (height, width)
            
        Returns:
            numpy.ndarray: Warped image
        """
        warped = np.zeros(output_shape, dtype=np.uint8)
        H_inv = np.linalg.inv(H)
        
        for y in range(output_shape[0]):
            for x in range(output_shape[1]):
                # Apply inverse homography
                p = np.array([x, y, 1], dtype=np.float64)
                p_prime = np.dot(H_inv, p)
                p_prime = p_prime / p_prime[2]
                
                # Get source coordinates
                src_x = int(round(p_prime[0]))
                src_y = int(round(p_prime[1]))
                
                # Check bounds
                if (0 <= src_x < image.shape[1] and 
                    0 <= src_y < image.shape[0]):
                    warped[y, x] = image[src_y, src_x]
        
        return warped

    def _blend_images(self, img1, img2, mask):
        """Blend two images using a mask.
        
        Args:
            img1 (numpy.ndarray): First image
            img2 (numpy.ndarray): Second image
            mask (numpy.ndarray): Blending mask
            
        Returns:
            numpy.ndarray: Blended image
        """
        return cv2.addWeighted(img1, 1-mask, img2, mask, 0)

    def create_panorama(self, img1, img2, H):
        """Create panorama by warping and blending two images.
        
        Args:
            img1 (numpy.ndarray): First image
            img2 (numpy.ndarray): Second image
            H (numpy.ndarray): Homography matrix
            
        Returns:
            numpy.ndarray: Panorama image
        """
        # Get corners of both images
        corners1 = self._get_corners(img1)
        corners2 = self._get_corners(img2)
        
        # Transform corners of second image
        transformed_corners = []
        for corner in corners2:
            p = np.array([corner[0], corner[1], 1], dtype=np.float64)
            p_prime = np.dot(H, p)
            p_prime = p_prime / p_prime[2]
            transformed_corners.append((int(round(p_prime[0])), int(round(p_prime[1]))))
        
        # Calculate output dimensions
        all_corners = corners1 + transformed_corners
        x_coords = [c[0] for c in all_corners]
        y_coords = [c[1] for c in all_corners]
        
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        # Create output image
        output_shape = (max_y - min_y + 1, max_x - min_x + 1, 3)
        panorama = np.zeros(output_shape, dtype=np.uint8)
        
        # Warp second image
        warped = self._warp_image(img2, H, output_shape)
        
        # Create mask for blending
        mask = np.zeros(output_shape[:2], dtype=np.float32)
        mask[0:img1.shape[0], 0:img1.shape[1]] = 1
        
        # Blend images
        for c in range(3):
            panorama[:, :, c] = self._blend_images(
                warped[:, :, c],
                img1[:, :, c],
                mask
            )
        
        return panorama 