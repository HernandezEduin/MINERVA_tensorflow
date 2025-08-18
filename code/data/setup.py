"""
Setup utilities for the MINERVA TensorFlow project.

This module provides utility functions for project setup and configuration,
including seed management for reproducible experiments and repository path resolution.
"""

import subprocess
from typing import Optional, Union

def set_seeds(seed: Union[int, None]) -> None:
    """
    Set random seeds for reproducible experiments across multiple libraries.
    
    Sets seeds for Python's random module, NumPy, and TensorFlow to ensure
    deterministic behavior in machine learning experiments. This is crucial
    for reproducible research and debugging.
    
    Args:
        seed: Random seed value. If None, no seeds are set.
        
    Note:
        This function imports the required libraries locally to avoid
        unnecessary dependencies if not needed.
    """
    if seed is None:
        return
        
    import numpy as np
    import random
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_git_root() -> Optional[str]:
    """
    Get the root directory path of the current Git repository.
    
    Uses the 'git rev-parse --show-toplevel' command to find the root directory
    of the Git repository containing the current working directory. This is useful
    for constructing absolute paths relative to the project root.
    
    Returns:
        The absolute path to the Git repository root, or None if not in a Git
        repository or if the Git command fails.
        
    Note:
        This function will return None if:
        - The current directory is not within a Git repository
        - The .git folder has been removed
        - Git is not installed or accessible
        
    Todo:
        Consider adding fallback logic to search for project-specific markers
        (e.g., specific folder names) if Git is unavailable.
    """
    try:
        # Run git command to get repository root directory
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            check=True
        )
        # Extract and return the cleaned path
        git_root: str = result.stdout.strip()
        return git_root
    except subprocess.CalledProcessError:
        # Handle cases where Git command fails (not in repo, Git not installed, etc.)
        return None