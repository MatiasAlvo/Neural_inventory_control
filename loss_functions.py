from shared_imports import *

class PolicyLoss(nn.Module):
    """
    Loss that returns the sum of the rewards.
    """
    
    def __init__(self):
        super(PolicyLoss, self).__init__()

    def forward(self, observation, action, reward):
        return reward.sum()