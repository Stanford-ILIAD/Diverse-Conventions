from typing import Optional, Tuple
from dataclasses import dataclass
import torch

@dataclass
class VectorObservation:
    """
    Representation of a batched observation provided by N environments in parallel.

    active: a mask specifying what environments are active as tensor of N bools
    obs: the partial observation that an agent receives
        - shape is N x (single_obs_shape)
    state: the full state information
        - shape is N x (single_state_shape)
    action_mask: a mask specifying what actions are legal
        - if it is None, all actions are permitted
        - shape is N x (num_discrete_actions)
    """
    active: torch.Tensor
    obs: torch.Tensor
    state: torch.Tensor
    action_mask: Optional[torch.Tensor] = None

    def __init__(self,
                 active: torch.Tensor,
                 obs: torch.Tensor,
                 state: Optional[torch.Tensor] = None,
                 action_mask: Optional[torch.Tensor] = None):
        self.active = active
        self.obs = obs
        self.state = (state if state is not None else obs)
        self.action_mask = action_mask
