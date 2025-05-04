import cv2
import numpy as np

def load_image(path):
    """Load an image from the given path.
    
    Args:
        path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Loaded image in BGR format
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    return img

def save_image(path, image):
    """Save an image to the given path.
    
    Args:
        path (str): Path where to save the image
        image (numpy.ndarray): Image to save
    """
    cv2.imwrite(path, image) 