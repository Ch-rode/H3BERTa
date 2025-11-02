import numpy as np 

def normalize_distributions(data_p, data_q, num_bins=50):
    import scipy.stats as stats
    """
    Estimates the probability density functions (PDFs) of two datasets using Kernel Density Estimation (KDE)
    and normalizes them to obtain comparable probability distributions.
    
    Parameters:
        data_p (array-like): First dataset (e.g., samples from a distribution).
        data_q (array-like): Second dataset (e.g., samples from another distribution).
        num_bins (int): Number of bins for evaluating the KDE (default: 50).
    
    Returns:
        bins (numpy array): Common range of values for both distributions.
        p (numpy array): Normalized probability distribution of data_p.
        q (numpy array): Normalized probability distribution of data_q.
    """
    # Define a common range of values based on the min and max of both datasets
    bins = np.linspace(min(data_p.min(), data_q.min()), max(data_p.max(), data_q.max()), num_bins)
    
    # Estimate probability density functions (PDF) using KDE
    kde_p = stats.gaussian_kde(data_p)
    kde_q = stats.gaussian_kde(data_q)
    
    # Evaluate the densities at the defined bin points
    p = kde_p(bins)
    q = kde_q(bins)
    
    # Normalize to ensure they sum to 1 (convert to probability distributions)
    p /= p.sum()
    q /= q.sum()
    
    return bins, p, q



def compute_jensenshannon(p, q):
    from scipy.spatial.distance import jensenshannon
    """
    Computes the Jensen-Shannon divergence between two probability distributions.
    
    The Jensen-Shannon divergence is a symmetric and smoothed version of the Kullback-Leibler divergence.
    It measures the similarity between two probability distributions and is always between 0 and 1.
    
    Parameters:
        p (numpy array): First probability distribution.
        q (numpy array): Second probability distribution.
    
    Returns:
        float: The Jensen-Shannon divergence between p and q.
    """
    # Ensure the distributions are normalized (sum to 1)
    p = p / p.sum()
    q = q / q.sum()
    
    # Compute the Jensen-Shannon divergence
    jsd = jensenshannon(p, q)
    return jsd