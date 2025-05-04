import argparse
import cv2

from src.feature_detection import FeatureDetector
from src.homography import HomographyEstimator
from src.warping import ImageWarper
from src.utils.image_utils import load_image, save_image

def main():
    parser = argparse.ArgumentParser(description='Create panoramic image mosaic from two images')
    parser.add_argument('--image1', required=True, help='Path to first image')
    parser.add_argument('--image2', required=True, help='Path to second image')
    parser.add_argument('--output', default='output_panorama.jpg', help='Output panorama path')
    parser.add_argument('--threshold', type=float, default=5.0, help='RANSAC threshold')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of RANSAC iterations')
    args = parser.parse_args()

    # Load images
    img1 = load_image(args.image1)
    img2 = load_image(args.image2)

    # Convert to grayscale for feature detection
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize components
    detector = FeatureDetector()
    homography_estimator = HomographyEstimator(threshold=args.threshold, iterations=args.iterations)
    warper = ImageWarper()

    # Detect features and compute descriptors
    print("Detecting features...")
    kp1, des1 = detector.detect_and_compute(gray1)
    kp2, des2 = detector.detect_and_compute(gray2)
    
    print(f"Found {len(kp1)} features in image 1 and {len(kp2)} features in image 2")

    # Match features
    print("Matching features...")
    matches = detector.match_features(des1, des2)
    print(f"Found {len(matches)} matches")

    if len(matches) < 4:
        print("Error: Not enough matches found between images")
        return

    # Get matched points
    src_pts, dst_pts = detector.get_matched_points(kp1, kp2, matches)

    # Estimate homography
    print("Estimating homography...")
    H, inliers = homography_estimator.estimate_homography(src_pts, dst_pts)

    if H is None:
        print("Error: Failed to estimate homography")
        return

    print(f"Found {len(inliers) if inliers is not None else 0} inliers")

    # Create panorama
    print("Creating panorama...")
    try:
        panorama = warper.warp_images(img1, img2, H)
        # Save result
        save_image(args.output, panorama)
        print(f"Panorama saved to {args.output}")
    except Exception as e:
        print(f"Error during warping: {str(e)}")
        return

if __name__ == '__main__':
    main()
