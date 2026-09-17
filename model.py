"""Factor encoding and counterfactual intervention (paper Sections 2.1–2.3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

FACTOR_NAMES = ("sequence", "structure", "regulation", "expression")


@dataclass
class ModelOutput:
    factual_logits: Tensor
    final_logits: Tensor
    intervention_logits: Tensor
    effects: Tensor
    factor_weights: Tensor
    disentanglement: Tensor
    consistency: Tensor


class CounterfactualCDA(nn.Module):
    """Association model using biological features, without an association graph.

    The four interventions set one encoded factor to zero at a time. This
    deterministic reference state is an explicit implementation choice because
    the manuscript does not specify the intervened factor's replacement value.
    """

    def __init__(self, factor_dims: dict[str, int], disease_dim: int, hidden_dim: int = 256):
        super().__init__()
        if set(factor_dims) != set(FACTOR_NAMES):
            raise ValueError(f"Expected factor dimensions for {FACTOR_NAMES}")
        if min(*factor_dims.values(), disease_dim, hidden_dim) <= 0:
            raise ValueError("Input and hidden dimensions must be positive")
        self.factor_encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(factor_dims[name], hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim))
            for name in FACTOR_NAMES
        })
        self.factual_projection = nn.Linear(4 * hidden_dim, hidden_dim)
        self.disease_encoder = nn.Sequential(nn.Linear(disease_dim, hidden_dim), nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim))
        self.predictor = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, 1))

    def _logits(self, circ: Tensor, disease: Tensor) -> Tensor:
        return self.predictor(torch.cat((circ, disease), dim=-1)).squeeze(-1)

    def forward(self, factors: dict[str, Tensor], disease_features: Tensor) -> ModelOutput:
        if set(factors) != set(FACTOR_NAMES):
            raise ValueError(f"Expected factors {FACTOR_NAMES}")
        z = torch.stack([self.factor_encoders[name](factors[name]) for name in FACTOR_NAMES], dim=1)
        disease = self.disease_encoder(disease_features)

        # Eq. 3 and Eq. 5: concatenate the four independent factor encodings.
        factual_repr = self.factual_projection(z.flatten(start_dim=1))
        factual_logits = self._logits(factual_repr, disease)

        # Eq. 6–8: intervene on exactly one factor, retaining the other three.
        cf_logits = []
        for i in range(4):
            intervened = z.clone()
            intervened[:, i, :] = 0.0
            cf_repr = self.factual_projection(intervened.flatten(start_dim=1))
            cf_logits.append(self._logits(cf_repr, disease))
        intervention_logits = torch.stack(cf_logits, dim=1)
        effects = (torch.sigmoid(factual_logits).unsqueeze(1)
                   - torch.sigmoid(intervention_logits)).abs()

        # Eq. 9–11 and Eq. 15. The weighted sum is the invariant representation.
        weights = F.softmax(effects, dim=1)
        invariant_repr = (weights.unsqueeze(-1) * z).sum(dim=1)
        final_logits = self._logits(invariant_repr, disease)

        # Eq. 4: squared off-diagonal Gram elements, averaged over pairs/batch.
        gram = torch.bmm(z, z.transpose(1, 2))
        mask = ~torch.eye(4, device=z.device, dtype=torch.bool)
        disentanglement = gram[:, mask].square().mean()
        # Eq. 12: consistency between factual and counterfactual predictions.
        consistency = (torch.sigmoid(factual_logits) - torch.sigmoid(final_logits)).square().mean()
        return ModelOutput(factual_logits, final_logits, intervention_logits,
                           effects, weights, disentanglement, consistency)


def objective(output: ModelOutput, labels: Tensor, lambda_inv: float = 0.5,
              lambda_dis: float = 0.0) -> tuple[Tensor, dict[str, float]]:
    """Eq. 13–14, with an optional coefficient for the Eq. 4 constraint."""
    association = F.binary_cross_entropy_with_logits(output.final_logits, labels.float())
    total = association + lambda_inv * output.consistency + lambda_dis * output.disentanglement
    parts = {"association": float(association.detach()),
             "consistency": float(output.consistency.detach()),
             "disentanglement": float(output.disentanglement.detach())}
    return total, parts
