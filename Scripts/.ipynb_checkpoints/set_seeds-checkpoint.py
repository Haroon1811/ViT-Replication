
# Creating a helper function to set seeds
import torch
def set_seeds(seed: int=42):
    """ Sets random sets for torch operations.

    Args:
        seed(int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations 
    torch.manual_seed(seed)
    # Set teh seed for CUDA torch operations(for ones that happen on GPUs)
    torch.cuda.manual_seed(seed)
