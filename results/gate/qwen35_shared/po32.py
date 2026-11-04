"""Per-option scoring with the float32 two-token readout (noise-floor control)."""
import torch
from model.scripts import eval_gate as eg


@torch.inference_mode()
def _score_option(self, state, instructions, q_type, label, description):
    query = f"{state}\n\nQuestion: {instructions}"
    kind = "Rubric level" if q_type == "score" else "Answer option"
    text = (f"<Instruct>: {eg.RAW_RERANKER_INSTRUCTION}\n<Query>: {query}\n"
            f"<Document>: {kind} {label}: {description}")
    tokens = self.tokenizer.encode(text, add_special_tokens=False)
    budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
    ids = torch.tensor([self.prefix_tokens + tokens[:budget] + self.suffix_tokens], device=self.device)
    hidden = self.model.model(input_ids=ids).last_hidden_state[0, -1]
    head = self.lm_head if self.lm_head is not None else self.model.lm_head
    pair = hidden.float() @ head.weight[[self.false_id, self.true_id]].float().T
    return torch.softmax(pair, -1)[1].item()


eg.RawReranker._score_option = _score_option
eg.main()
