import subprocess

from typing import Dict, Optional, List, Tuple, Union, Optional

def set_seeds(seed):
    import numpy as np
    import random
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_git_root() -> Optional[str]:
    # NOTE: This will break if we remove the .git folder (could very well happen).
    # IN that case:
    # TODO: simply return the path until the `MultiHopKG` folder is found. 

    try:
        # Run the git command to get the top-level directory of the repository
        result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        # The output will be the path to the root of the repository
        git_root = result.stdout.strip()
        return git_root
    except subprocess.CalledProcessError:
        # Handle the case where the command fails (e.g., not in a git repository)
        return None