import torch
from torch import nn

from labml_nn.sampling import Sampler
import torch.nn.functional as F

class NucleusSampler(Sampler):
    """
    ## Nucleus Sampler
    """
    def __init__(self, p: float, sampler: Sampler = None, top_k = 400, temp: int = 0.55):
        """
        :param p: is the sum of probabilities of tokens to pick $p$
        :param sampler: is the sampler to use for the selected tokens
        """
        self.p = p
        self.sampler = sampler
        # Softmax to compute $P(x_i | x_{1:i-1})$ from the logits_        self.softmax = nn.Softmax(dim=-1)
        self.top_k = top_k
        self.temp = temp

    def __call__(self, logits: torch.Tensor, mask):
        """
        Sample from logits_with Nucleus Sampling
        """
        
        # Get probabilities $P(x_i | x_{1:i-1})$
        if self.temp:
            logits = logits/self.temp
        probs = F.softmax(logits, dim = -1)
        probs *= mask.squeeze(dim = 0) # Tạm để như thế này vì batch_size bằng 1, fix như này dễ hơn

        
        # Sort probabilities in descending order
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        # Get the cumulative sum of probabilities in the sorted order
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find the cumulative sums less than $p$.
        nucleus = cum_sum_probs < self.p
        # Prepend ones so that we add one token after the minimum number
        # of tokens with cumulative probability less that $p$.
        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
        
        # Get log probabilities and mask out the non-nucleus
        # sorted_log_probs = torch.log(sorted_probs)
        # sorted_log_probs[~nucleus] = float('-inf')
        
        if torch.sum(nucleus) > self.top_k:
            nucleus_probs, nucleus_indices = torch.topk(
                sorted_probs,
                k = self.top_k)
            actual_nucleus_indices = indices.gather(
                dim = -1, 
                index = nucleus_indices# .unsqueeze(-1)
            )
            nucleus_probs = nucleus_probs.unsqueeze(0)
            actual_nucleus_indices = actual_nucleus_indices.unsqueeze(0)
            # Đây là trong trường hợp chỉ nhét vào batch_size = 1, nên unsqueeze = 0 để trở lại
            # shape 1 x top_p_probs_size, nếu không thì nó lại về top_p_probs_size mất
        else:
            nucleus_indices = nucleus.nonzero().view(-1)
            nucleus_probs = sorted_probs.gather(
                dim = -1, 
                index = nucleus_indices
            )
            # sorted_probs[nucleus_indices]
            actual_nucleus_indices = indices.gather(
                dim = -1, 
                index = nucleus_indices
            )
            nucleus_probs = nucleus_probs.unsqueeze(0)
            actual_nucleus_indices = actual_nucleus_indices.unsqueeze(0)
        return nucleus_probs, actual_nucleus_indices

        # Sample from the sampler
        # sampled_sorted_indexes = self.sampler(sorted_log_probs)

        # Get the actual indexes
        # res = indices.gather(-1, sampled_sorted_indexes.unsqueeze(-1))

        #
        # return res.squeeze(-1)