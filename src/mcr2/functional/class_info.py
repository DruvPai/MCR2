import torch

def pi_to_y(Pi):
    """
    Turns a class membership matrix into a vector of class labels.
    Args:
        Pi: A one-hot encoded class membership matrix (N, K)

    Returns:
        A vector y of class labels (N, )
    """
    return torch.argmax(
        Pi,  # (N, K)
        dim=1
    )  # (N, )

def y_to_pi(y, K=-1):
    """
    Turns a vector of class labels y into a one-hot encoding class membership matrix Pi.
    Note: Assuming classes are indices {0, 1, ..., K - 1}. If K is not provided,
    picks K = max(y) + 1.

    Args:
        y: The vector of class labels.
        K: The number of classes, if provided. Optional since we can estimate it from given labels.

    Returns:
        A class membership matrix Pi (N, K)
    """
    return torch.nn.functional.one_hot(y, K)  # (N, K)

__all__ = ["pi_to_y", "y_to_pi"]
