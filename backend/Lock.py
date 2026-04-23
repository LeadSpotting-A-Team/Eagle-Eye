import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 

 

def The_Lock(reference_image : np.ndarray, new_image : np.ndarray, keypoints_detector = cv.ORB_create()):
    '''
    Aligns a new image to a reference image's coordinate system using ORB features and Homography.

    Args:
        reference_image (np.ndarray): The static grayscale image used as the coordinate baseline.
        new_image (np.ndarray): The grayscale image to be transformed (warped) to match the reference.
        keypoints_detector: OpenCV feature detector object (default is ORB).

    Returns:
        aligned_image (np.ndarray): The 'new_image' transformed to perfectly overlay the 'reference_image'.
        M (np.ndarray): The 3x3 perspective transformation matrix used for the alignment.
    '''
    # 1. Feature Extraction: Find keypoints (points of interest) and descriptors (their "ID cards")
    # reference_image: The static image we want to match to.
    # new_image: The image that might have different zoom/angle.
    kp_ref, des_ref = keypoints_detector.detectAndCompute(reference_image, None)
    kp_new, des_new = keypoints_detector.detectAndCompute(new_image, None)

    # 2. Matching: Create a BFMatcher (Brute-Force Matcher)
    # cv.NORM_HAMMING: The distance metric for ORB (compares binary strings).
    # crossCheck=True: Ensures that point A in image 1 matches point B in image 2, AND vice versa.
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # 3. Find matches between the two sets of descriptors
    matches = bf.match(des_ref, des_new)

    # 4. Sort matches by distance (shorter distance = better/more reliable match)
    matches = sorted(matches, key=lambda x: x.distance)

    # 5. Coordinate Extraction: Convert match indexes into actual (x, y) coordinates
    # m.queryIdx: Index of the keypoint in the reference image.
    # m.trainIdx: Index of the keypoint in the new image.
    # .pt: Extracts the (x, y) floating point coordinates.
    # .reshape(-1, 1, 2): Formats the array as required by OpenCV (N rows, 1 channel, 2 coords).
    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    new_pts = np.float32([kp_new[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 6. Homography Mapping: Compute the transformation matrix (M)
    # new_pts: Points to be transformed (Source).
    # ref_pts: Target points (Destination).
    # cv.RANSAC: An algorithm that ignores "bad" matches (outliers) to ensure a precise lock.
    # 5.0: The threshold (in pixels) for a match to be considered valid by RANSAC.
    M, mask = cv.findHomography(new_pts, ref_pts, cv.RANSAC, 5.0)

    # 7. Warping: Apply the matrix to align the new image to the reference coordinates
    # reference_image.shape: Ensures the output has the same height (h) and width (w).
    h, w = reference_image.shape
    aligned_image = cv.warpPerspective(new_image, M, (w, h))

    return aligned_image, M