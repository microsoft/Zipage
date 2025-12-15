import torch
from torch import nn
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor


class Sampler(nn.Module):

    def __init__(self, repetition_penalty: float = 1.0):
        super().__init__()
        if repetition_penalty != 1.0:
            self.repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
                repetition_penalty
            )
        else:
            self.repetition_penalty_processor = None

    @torch.compile
    def calculate_probs(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        return probs

    @torch.compile
    def calculate_entropy(self, probs: torch.Tensor):
        return -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)

    @torch.compile
    def sample(self, probs: torch.Tensor):
        return probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        past_tokens: torch.Tensor = None,
    ):
        """
        logits: [batch, vocab_size]
        temperatures: [batch]
        past_tokens: [batch, prev_seq_len] or None
        """
        if past_tokens is not None and self.repetition_penalty_processor is not None:
            if past_tokens.numel() > 0:
                logits = self.repetition_penalty_processor(past_tokens, logits)
        probs = self.calculate_probs(logits, temperatures)
        sample_tokens = self.sample(probs)
        return sample_tokens
