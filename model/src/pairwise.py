"""Two-stage pairwise readout: rank a stage-1 shortlist by pairwise duels.

Stage 1 is the raw backbone's own distribution over the full option set.  Its
top-m options form the shortlist.  Stage 2 asks the marker head, two options
at a time, which of the pair is right, and turns those duels into one score
per shortlisted option.

The head is the same marker head the listwise arm uses, restricted to
two-option sets: no new parameters, and the two arms stay comparable.  Choice
pairs are scored in both orders and averaged, because the head sees a
sequence and can otherwise learn a position bias.  Score items keep rubric
order: a pair of levels is not exchangeable, so there is one order and one
duel.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from itertools import combinations

import torch
import torch.nn.functional as F

from .decision_model import DecisionModel
from .listwise import ListwiseMarkerReadout


def stage1_shortlist(
    base: torch.Tensor, m: int, must_include: int | None = None
) -> list[int]:
    """Top-``m`` option indices by base probability, stable on ties.

    ``must_include`` adds the gold when the base ranking drops it: training
    cannot build a pair without the answer, and eval needs to know whether
    the shortlist was even capable of a hit.
    """
    if m < 1:
        raise ValueError("shortlist size must be at least 1")
    order = sorted(range(base.numel()), key=lambda index: (-float(base[index]), index))
    shortlist = order[:m]
    if must_include is not None and must_include not in shortlist:
        shortlist = shortlist[: m - 1] + [must_include] if m > 1 else [must_include]
    return sorted(shortlist)


class PairwiseMarkerReadout:
    """Pairwise duels over a shortlist with the marker head.

    Args:
        model: A :class:`DecisionModel` built with ``option_marker=True``.
    """

    def __init__(self, model: DecisionModel) -> None:
        self.readout = ListwiseMarkerReadout(model)
        self.model = model

    # -- one duel -------------------------------------------------------
    def duel(
        self,
        state: str,
        instructions: str,
        first: str,
        second: str,
    ) -> float:
        """Probability that ``first`` beats ``second``, in this order only."""
        logits = self.readout.score(state, instructions, [first, second])
        return float(torch.softmax(logits, dim=-1)[0, 0].detach())

    def pair_win(
        self,
        state: str,
        instructions: str,
        first: str,
        second: str,
        *,
        order_sensitive: bool,
    ) -> float:
        """Probability that ``first`` beats ``second``.

        ``order_sensitive`` averages the two presentation orders; otherwise
        the pair is scored once, in the given order, because rubric levels are
        not exchangeable.
        """
        forward = self.duel(state, instructions, first, second)
        if not order_sensitive:
            return forward
        backward = 1.0 - self.duel(state, instructions, second, first)
        return 0.5 * (forward + backward)

    # -- a whole shortlist ----------------------------------------------
    def option_scores(
        self,
        state: str,
        instructions: str,
        option_texts: Sequence[str],
        shortlist: Sequence[int],
        *,
        order_sensitive: bool,
    ) -> dict[int, float]:
        """Mean pairwise win log-probability per shortlisted option.

        Every pair inside the shortlist contributes one log-win to each side;
        an option's score is the mean over its duels.  A one-option shortlist
        has no duels and scores 0, so the fusion falls back to the base.
        """
        indices = list(shortlist)
        wins: dict[int, list[float]] = {index: [] for index in indices}
        for first, second in combinations(indices, 2):
            p_first = self.pair_win(
                state,
                instructions,
                option_texts[first],
                option_texts[second],
                order_sensitive=order_sensitive,
            )
            p_first = min(max(p_first, 1e-6), 1.0 - 1e-6)
            wins[first].append(math.log(p_first))
            wins[second].append(math.log(1.0 - p_first))
        return {
            index: (sum(values) / len(values) if values else 0.0)
            for index, values in wins.items()
        }

    def shortlist_probabilities(
        self,
        state: str,
        instructions: str,
        option_texts: Sequence[str],
        base: torch.Tensor,
        alpha: float,
        *,
        m: int,
        order_sensitive: bool,
    ) -> tuple[torch.Tensor, list[int]]:
        """Fused probabilities over the full option set.

        The shortlist is renormalised base probability times the pairwise
        score; every option outside it gets probability 0.
        """
        n_options = len(option_texts)
        shortlist = stage1_shortlist(base, m)
        probs = torch.zeros(n_options, dtype=torch.float32, device=base.device)
        if not shortlist:
            return probs, shortlist
        picked = base[torch.tensor(shortlist, device=base.device)].clamp_min(1e-6)
        picked = picked / picked.sum()
        if len(shortlist) == 1:
            probs[shortlist[0]] = 1.0
            return probs, shortlist
        scores = self.option_scores(
            state,
            instructions,
            option_texts,
            shortlist,
            order_sensitive=order_sensitive,
        )
        head = torch.tensor(
            [scores[index] for index in shortlist],
            dtype=torch.float32,
            device=base.device,
        )
        fused = torch.log(picked) + alpha * head
        probs[torch.tensor(shortlist, device=base.device)] = torch.softmax(fused, dim=-1)
        return probs, shortlist

    # -- training -------------------------------------------------------
    def pair_loss(
        self,
        state: str,
        instructions: str,
        option_texts: Sequence[str],
        gold_index: int,
        distractor_index: int,
        base: torch.Tensor,
        alpha: torch.Tensor,
        rng: random.Random,
        *,
        kl_weight: float,
        ce_weight: float,
        order_sensitive: bool,
    ) -> tuple[torch.Tensor, float, float]:
        """Fused CE and KL on one gold/distractor pair.

        The pair is presented in the gold-first order half the time, so a
        choice head cannot learn a position shortcut; evaluation averages
        both orders instead.
        """
        pair = [gold_index, distractor_index]
        if order_sensitive and rng.random() < 0.5:
            pair = pair[::-1]
        elif not order_sensitive:
            pair = sorted(pair)
        head = F.log_softmax(
            self.readout.score(
                state, instructions, [option_texts[index] for index in pair]
            ).reshape(-1),
            dim=-1,
        )
        picked = base[torch.tensor(pair, device=base.device)].clamp_min(1e-6)
        picked = picked / picked.sum()
        logits = torch.log(picked) + alpha * head
        log_p = F.log_softmax(logits, dim=-1)
        kl = torch.sum(picked * (torch.log(picked) - log_p))
        target = torch.tensor([pair.index(gold_index)], device=logits.device)
        ce = F.cross_entropy(logits.unsqueeze(0), target)
        return kl_weight * kl + ce_weight * ce, float(kl.detach()), float(ce.detach())
