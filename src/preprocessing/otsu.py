import numpy as np
import cv2
from typing import Tuple, List, Optional

def calculate_otsu_threshold(image: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Apply Otsu's thresholding to a grayscale image.
    
    Args:
        image: Grayscale image as numpy array
        
    Returns:
        tuple: (threshold value, binary thresholded image)
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Calculate Otsu's threshold
    threshold_value, binary_image = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return threshold_value, binary_image

def calculate_content_area_ratio(binary_image: np.ndarray) -> float:
    """
    Calculate the ratio of content area to total image area.
    
    Args:
        binary_image: Binary image as numpy array
        
    Returns:
        float: Ratio of content area to total area
    """
    total_pixels = binary_image.size
    content_pixels = np.count_nonzero(binary_image)
    
    return content_pixels / total_pixels

def filter_patches(patches: List[np.ndarray], min_content_ratio: float = 0.3) -> List[np.ndarray]:
    """
    Filter image patches based on content area ratio using Otsu's thresholding.
    
    Args:
        patches: List of image patches as numpy arrays
        min_content_ratio: Minimum required ratio of content area to total area
        
    Returns:
        list: Filtered list of patches that meet the content ratio requirement
    """
    filtered_patches = []
    
    for patch in patches:
        # Apply Otsu's thresholding
        _, binary_image = calculate_otsu_threshold(patch)
        
        # Calculate content area ratio
        content_ratio = calculate_content_area_ratio(binary_image)
        
        # Keep patch if content ratio exceeds threshold
        if content_ratio > min_content_ratio:
            filtered_patches.append(patch)
            
    return filtered_patches

def process_single_patch(patch: np.ndarray, min_content_ratio: float = 0.3) -> Optional[np.ndarray]:
    """
    Process a single patch and return it only if it meets the content ratio requirement.
    
    Args:
        patch: Image patch as numpy array
        min_content_ratio: Minimum required ratio of content area to total area
        
    Returns:
        numpy.ndarray or None: The patch if it meets the requirement, None otherwise
    """
    # Apply Otsu's thresholding
    _, binary_image = calculate_otsu_threshold(patch)
    
    # Calculate content area ratio
    content_ratio = calculate_content_area_ratio(binary_image)
    
    # Return patch if content ratio exceeds threshold
    if content_ratio > min_content_ratio:
        return patch
    return None

