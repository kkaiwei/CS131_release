from __future__ import print_function

import random
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from scipy.ndimage.filters import convolve
from scipy.spatial.distance import cdist
from skimage import color, filters, io
from skimage.feature import corner_peaks
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float
import os
from skimage import transform

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        prev_assignments = np.copy(assignments)
        
        for i in range(N):
            distances = np.sqrt(np.sum((features[i] - centers) ** 2, axis=1))
            assignments[i] = np.argmin(distances)
            
        if np.all(assignments == prev_assignments):
            break
            
        for j in range(k):
            cluster_points = features[assignments == j]
            if len(cluster_points) > 0:
                centers[j] = np.mean(cluster_points, axis=0)
        ### END YOUR CODE

    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find cdist (imported from scipy.spatial.distance) and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        distances = cdist(features, centers, 'euclidean')
        new_assignments = np.argmin(distances, axis=1)
        
        if np.all(assignments == new_assignments):
            break
            
        assignments = new_assignments
        
        for j in range(k):
            cluster_points = features[assignments == j]
            if len(cluster_points) > 0:
                centers[j] = np.mean(cluster_points, axis=0)
        ### END YOUR CODE

    return assignments

def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Hints
    - You may find pdist (imported from scipy.spatial.distance) useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N, dtype=np.uint32)
    centers = np.copy(features)
    n_clusters = N

    # Initial distances array
    dists = squareform(pdist(centers))
    np.fill_diagonal(dists, float('inf'))
    assert dists.shape == (N, N)

    while n_clusters > k:
        ### YOUR CODE HERE
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        
        # Merge clusters i and j, keeping cluster i and removing cluster j
        assignments[assignments == j] = i
        # Reassign all clusters with indices greater than j
        assignments[assignments > j] -= 1
        
        mask_i = assignments == i
        centers[i] = features[mask_i].mean(axis=0)
        
        dists = np.delete(np.delete(dists, j, axis=0), j, axis=1)
        centers = np.delete(centers, j, axis=0)
        
        new_dists = pdist(centers)
        dists = squareform(new_dists)
        np.fill_diagonal(dists, float('inf'))
        
        n_clusters -= 1
    return assignments

def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
    features = img.reshape(-1, C)
    ### END YOUR CODE

    return features

def visualize_mean_color_image(img, segments):

    img = img_as_float(img)
    k = np.max(segments) + 1
    mean_color_img = np.zeros(img.shape)

    for i in range(k):
        mean_color = np.mean(img[segments == i], axis=0)
        mean_color_img[segments == i] = mean_color

    plt.imshow(mean_color_img)
    plt.axis('off')
    plt.show()

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    color = img_as_float(img)
    color = color.reshape(-1, C)
    color = (color - np.mean(color, axis=0)) / np.std(color, axis=0)
    
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    positions = np.dstack((x_coords, y_coords)).reshape(-1, 2)
    
    positions = (positions - np.mean(positions, axis=0)) / np.std(positions, axis=0)
    
    features[:, :C] = color
    features[:, C:] = positions
    ### END YOUR CODE

    return features


def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE
    from skimage import filters, color, img_as_float
    from scipy import ndimage as ndi
    import numpy as np
    
    H, W, C = img.shape
    
    ### YOUR CODE HERE
    img_float = img_as_float(img)
    
    # Color features
    img_lab = color.rgb2lab(img_float)
    color_features = img_lab.reshape(-1, C)    
    color_features = (color_features - np.mean(color_features, axis=0)) / np.std(color_features, axis=0)
    
    # Position features 
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    position_features = np.dstack((x_coords, y_coords)).reshape(-1, 2)
    position_features = (position_features - np.mean(position_features, axis=0)) / np.std(position_features, axis=0)   
    position_weight = 0.8
    position_features *= position_weight
    
    # Edge features
    grayscale = color.rgb2gray(img_float)
    gradient_x = filters.sobel_h(grayscale)
    gradient_y = filters.sobel_v(grayscale)
    gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)    
    gradient_mag = (gradient_mag - np.mean(gradient_mag)) / np.std(gradient_mag)
    gradient_features = gradient_mag.reshape(-1, 1)
    
    # Texture features
    window_size = 3  # Adjust based on the scale of textures
    texture_feature = ndi.uniform_filter(grayscale**2, size=window_size) - ndi.uniform_filter(grayscale, size=window_size)**2
    texture_feature = texture_feature.reshape(-1, 1)    
    texture_feature = (texture_feature - np.mean(texture_feature)) / np.std(texture_feature)
    
    # Combine all features
    features = np.hstack((color_features, position_features, gradient_features, texture_feature))
    ### END YOUR CODE
    return features


def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    accuracy = np.mean(mask_gt == mask)
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy

def load_dataset(data_dir):
    """
    This function assumes 'gt' directory contains ground truth segmentation
    masks for images in 'imgs' dir. The segmentation mask for image
    'imgs/aaa.jpg' is 'gt/aaa.png'
    """

    imgs = []
    gt_masks = []

    # Load all the images under 'data_dir/imgs' and corresponding
    # segmentation masks under 'data_dir/gt'.
    for fname in sorted(os.listdir(os.path.join(data_dir, 'imgs'))):
        if fname.endswith('.jpg'):
            # Load image
            img = io.imread(os.path.join(data_dir, 'imgs', fname))
            imgs.append(img)

            # Load corresponding gt segmentation mask
            mask_fname = fname[:-4] + '.png'
            gt_mask = io.imread(os.path.join(data_dir, 'gt', mask_fname))
            gt_mask = (gt_mask != 0).astype(int) # Convert to binary mask (0s and 1s)
            gt_masks.append(gt_mask)

    return imgs, gt_masks

def compute_segmentation(img, k,
        clustering_fn=kmeans_fast,
        feature_fn=color_position_features,
        scale=0):
    """ Compute a segmentation for an image.

    First a feature vector is extracted from each pixel of an image. Next a
    clustering algorithm is applied to the set of all feature vectors. Two
    pixels are assigned to the same segment if and only if their feature
    vectors are assigned to the same cluster.

    Args:
        img - An array of shape (H, W, C) to segment.
        k - The number of segments into which the image should be split.
        clustering_fn - The method to use for clustering. The function should
            take an array of N points and an integer value k as input and
            output an array of N assignments.
        feature_fn - A function used to extract features from the image.
        scale - (OPTIONAL) parameter giving the scale to which the image
            should be in the range 0 < scale <= 1. Setting this argument to a
            smaller value will increase the speed of the clustering algorithm
            but will cause computed segments to be blockier. This setting is
            usually not necessary for kmeans clustering, but when using HAC
            clustering this parameter will probably need to be set to a value
            less than 1.
    """

    assert scale <= 1 and scale >= 0, \
        'Scale should be in the range between 0 and 1'

    H, W, C = img.shape

    if scale > 0:
        # Scale down the image for faster computation.
        img = transform.rescale(img, scale)

    features = feature_fn(img)
    assignments = clustering_fn(features, k)
    segments = assignments.reshape((img.shape[:2]))

    if scale > 0:
        # Resize segmentation back to the image's original size
        segments = transform.resize(segments, (H, W), preserve_range=True)

        # Resizing results in non-interger values of pixels.
        # Round pixel values to the closest interger
        segments = np.rint(segments).astype(int)

    return segments


