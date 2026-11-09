"""The judgement objective: per-option yes/no prompts, one two-logit head.

Training and inference must share one prompt builder, so this module calls
``model/evaluation/judgement_prompt.py`` rather than re-deriving the template
(``#299``).  A row produces one prompt per option; the head reads the final
hidden state at a chosen position and emits two logits, ``no`` then ``yes``.
The head never fuses with a base logit: the fusion gradient is zero at the
fusion weight's initial value (``#324``).

Read positions
--------------
* ``answer``  — the final token, i.e. the ``Answer:`` cue.  Causal only in
  practice: with a full mask this position already sees the whole prefix.
* ``option``  — the last token of the option text.  Under a causal mask it
  cannot see the trailing question; under a full mask it can.

A criteria-less noul row has no option text (``render_polarity`` puts the
proposition in the question), so its read position is always ``answer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.evaluation.judgement_prompt import (
    candidates,
    polarity_gold,
    render,
    render_polarity,
)

#: The text that follows the option in ``JUDGEMENT_TEMPLATE``.  Its position
#: marks one past the last character of the option.
OPTION_END_MARKER = "\nDoes this option correctly answer the question?"

#: ``logits[..., 1]`` is yes.
YES_INDEX = 1


@dataclass(frozen=True)
class JudgementPrompt:
    """One prompt of the judgement objective, with its label."""

    row_id: str
    source: str
    split: str
    q_type: str
    text: str
    label: int
    option_index: int
    n_options: int
    #: Character offset one past the option text, or ``None`` when the row has
    #: no option span (a criteria-less noul).
    option_char_end: int | None

    @property
    def has_option_span(self) -> bool:
        return self.option_char_end is not None


def build_prompts(
    item: Any, row_id: str = "", source: str = "", split: str = ""
) -> list[JudgementPrompt]:
    """Return one prompt per option for ``item``, or an empty list.

    Returns an empty list when the item has neither an option set nor a
    polarity label — the same two conditions under which the readout skips it.
    """
    instructions = item.question.get("instructions", "")
    prepared = candidates(item)

    if prepared is None:
        gold = polarity_gold(item)
        if gold is None:
            return []
        return [
            JudgementPrompt(
                row_id=row_id or item.id,
                source=source,
                split=split,
                q_type=item.question.get("type", "noul"),
                text=render_polarity(item.state, instructions),
                label=int(bool(gold)),
                option_index=0,
                n_options=1,
                option_char_end=None,
            )
        ]

    texts, gold = prepared
    prompts: list[JudgementPrompt] = []
    for index, option_text in enumerate(texts):
        text = render(item.state, instructions, option_text)
        cut = text.find(OPTION_END_MARKER)
        end = len(text) if cut < 0 else cut
        prompts.append(
            JudgementPrompt(
                row_id=row_id or item.id,
                source=source,
                split=split,
                q_type=item.question.get("type", "noul"),
                text=text,
                label=int(index == gold),
                option_index=index,
                n_options=len(texts),
                option_char_end=end,
            )
        )
    return prompts


def read_token_index(tokenizer: Any, text: str, option_char_end: int | None) -> int:
    """Token index the head reads for ``text``.

    With ``option_char_end`` the last token overlapping the option's final
    character; without it the final token of the sequence.
    """
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if option_char_end is None:
        return len(offsets) - 1
    target = option_char_end - 1
    for index in range(len(offsets) - 1, -1, -1):
        start, end = offsets[index]
        if start <= target < end:
            return index
    return len(offsets) - 1


def full_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Additive 4D mask that lets every token see every other real token.

    ``attention_mask`` is the usual ``[batch, length]`` 0/1 tensor; padding
    positions are masked out with ``-inf`` so a padded batch matches a single
    sequence.
    """
    keep = attention_mask[:, None, None, :].bool()
    rows, length = attention_mask.shape
    keep = keep.expand(rows, 1, length, length)
    zeros = torch.zeros((), dtype=torch.float32, device=attention_mask.device)
    neg_inf = torch.full((), float("-inf"), dtype=torch.float32, device=attention_mask.device)
    return torch.where(keep, zeros, neg_inf)


def run_backbone(
    backbone: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    bidirectional: bool,
) -> torch.Tensor:
    """Return the last hidden state of a causal LM under one direction.

    ``backbone`` may be the HF causal LM or its ``.model`` tower.  The full
    direction passes the mask as a dict, which ``Qwen3Model.forward`` accepts
    as an already prepared causal-mask mapping and therefore does not replace
    with a causal one.
    """
    tower = getattr(backbone, "model", backbone)
    mask: Any = attention_mask
    if bidirectional:
        mask = {"full_attention": full_attention_mask(attention_mask)}
    output = tower(
        input_ids=input_ids,
        attention_mask=mask,
        use_cache=False,
    )
    return output.last_hidden_state


class TwoWayHead(nn.Module):
    """Two logits — ``no``, ``yes`` — over one read-position hidden state.

    One linear layer.  The yes bias starts positive so a corpus whose positive
    rate is about ``1/K`` does not drive the head to a near-constant no
    (``#299``); the check is that P(yes) exceeds 0.5 on some gold options.
    """

    def __init__(self, hidden_size: int, yes_bias: float = 2.0, std: float = 0.02):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 2)
        nn.init.normal_(self.proj.weight, std=std)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[YES_INDEX] = yes_bias

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)

    @staticmethod
    def score(logits: torch.Tensor) -> torch.Tensor:
        """The yes-minus-no score used to rank options."""
        return logits[..., YES_INDEX] - logits[..., 1 - YES_INDEX]


def row_normalised_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    rows: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Cross-entropy averaged within a row, then across rows (``#299``).

    Every row weighs the same regardless of how many options it has, so a
    77-option row cannot contribute 76 times the gradient of a 2-option row.
    """
    if not rows:
        raise ValueError("row_normalised_loss needs at least one row")
    per_row = [
        F.cross_entropy(logits[index], labels[index]) for index in rows
    ]
    return torch.stack(per_row).mean()


def contiguous_rows(row_ids: torch.Tensor) -> list[torch.Tensor]:
    """Split a batch of sorted row ids into index tensors, one per row."""
    if row_ids.numel() == 0:
        return []
    boundaries = torch.nonzero(row_ids[1:] != row_ids[:-1]).flatten() + 1
    edges = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=row_ids.device), boundaries,
         torch.tensor([row_ids.numel()], dtype=torch.long, device=row_ids.device)]
    )
    return [torch.arange(edges[i], edges[i + 1], device=row_ids.device)
            for i in range(len(edges) - 1)]
