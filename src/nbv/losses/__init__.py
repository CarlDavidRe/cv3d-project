"""Losses for utility-map prediction."""

from nbv.losses.huber import masked_huber_loss
from nbv.losses.ranking import masked_pairwise_ranking_loss

__all__ = ["masked_huber_loss", "masked_pairwise_ranking_loss"]
