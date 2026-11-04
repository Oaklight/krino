"""Per-option raw scoring with the float32 two-token readout.

Control for the shared-prefix parity check: it keeps ``_score_option``'s one
full forward per option and changes only the readout to the one
``score_many`` uses.  Comparing it with the stock per-option run isolates the
readout precision; comparing it with ``--shared-prefix`` isolates the cache
path.  Takes the same arguments as ``eval_gate.py``.
"""

import torch

from model.scripts import eval_gate as eg


@torch.inference_mode()
def _score_option(self, state, instructions, q_type, label, description):
    query = f"{state}\n\nQuestion: {instructions}"
    kind = "Rubric level" if q_type == "score" else "Answer option"
    text = (
        f"<Instruct>: {eg.RAW_RERANKER_INSTRUCTION}\n<Query>: {query}\n"
        f"<Document>: {kind} {label}: {description}"
    )
    tokens = self.tokenizer.encode(text, add_special_tokens=False)
    budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
    ids = self.prefix_tokens + tokens[:budget] + self.suffix_tokens
    hidden = self.model.model(
        input_ids=torch.tensor([ids], device=self.device)
    ).last_hidden_state[0, -1]
    head = self.lm_head if self.lm_head is not None else self.model.lm_head
    weight = head.weight[[self.false_id, self.true_id]].float()
    return torch.softmax(hidden.float() @ weight.T, -1)[1].item()


eg.RawReranker._score_option = _score_option
eg.RawReranker.scoring_path = property(
    lambda self: {"path": "per_option", "readout": "float32_two_token"}
)

if __name__ == "__main__":
    eg.main()
