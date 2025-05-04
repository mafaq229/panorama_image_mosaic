import numpy as np
import cv2

class FeatureDetector:
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def detect_and_compute(self, image):
        """Detect keypoints and compute descriptors using SIFT.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        return self.sift.detectAndCompute(image, None)

    def match_features(self, des1, des2):
        """Match features using BFMatcher with cross-check.
        
        Args:
            des1 (numpy.ndarray): Descriptors from first image
            des2 (numpy.ndarray): Descriptors from second image
            
        Returns:
            list: List of matches
        """
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)

    def get_matched_points(self, kp1, kp2, matches):
        """Get corresponding points from matches.
        
        Args:
            kp1 (list): Keypoints from first image
            kp2 (list): Keypoints from second image
            matches (list): List of matches
            
        Returns:
            tuple: (src_pts, dst_pts) - Arrays of corresponding points
        """
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
        return src_pts, dst_pts 