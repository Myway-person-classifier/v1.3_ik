"""
Qwen3 Sequence‑Classification with Contrastive Learning (InfoNCE)
===============================================================
* Drop‑in head for Qwen/Qwen3‑8B (etc.)
* BCE / CE / MSE + NT‑Xent (InfoNCE)
* Safe to **re‑import multiple times** (handles HF registry duplicates)

Requires: `transformers>=4.51`, `torch`.

Usage
-----
```python
from qwen3_seqcls_infonce import Qwen3ForSequenceClassificationCL
from transformers import AutoTokenizer
import torch

model = Qwen3ForSequenceClassificationCL.from_pretrained(
    "Qwen/Qwen3-8B", num_labels=2, device_map="auto")

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
batch  = tok(["A", "B"], padding=True, return_tensors="pt")
labels = torch.tensor([0, 1])

out = model(**batch, labels=labels, contrastive_labels=labels, lambda_cl=0.1)
print(out.loss)
```
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import AutoModelForSequenceClassification

# -----------------------------------------------------------------------------
#  Import Qwen3 backbone -------------------------------------------------------
# -----------------------------------------------------------------------------
try:
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
except ImportError as e:
    raise ImportError("❌  Qwen3 not found – `pip install -U transformers>=4.51`.") from e

# ----------------------------------------------------------------------------
#  Helpers --------------------------------------------------------------------
# ----------------------------------------------------------------------------

def _pool_last(token_logits, input_ids, pad_id):
    if input_ids is None or pad_id is None:
        return token_logits[:, -1]
    ends = (~input_ids.eq(pad_id)).cumsum(-1).argmax(-1)
    return token_logits[torch.arange(token_logits.size(0), device=token_logits.device), ends]


def _compute_standard_loss(cfg, logits, labels, n_labels):
    if labels is None:
        return None
    if cfg.problem_type is None:
        if n_labels == 1:
            cfg.problem_type = "regression"
        elif labels.dtype in (torch.long, torch.int):
            cfg.problem_type = "single_label_classification"
        else:
            cfg.problem_type = "multi_label_classification"
    if cfg.problem_type == "regression":
        return nn.MSELoss()(logits.squeeze(), labels.squeeze())
    if cfg.problem_type == "single_label_classification":
        return nn.CrossEntropyLoss()(logits, labels)
    return nn.BCEWithLogitsLoss()(logits, labels)


def _contrastive_loss(feats, labels, *, temperature: float = 0.07):
    f = nn.functional.normalize(feats.float(), dim=-1)
    sim = (f @ f.T) / temperature
    labels = labels.view(-1)
    pos = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    pos.fill_diagonal_(False)
    sim = sim - torch.eye(sim.size(0), device=sim.device) * 1e9
    logp = sim - sim.logsumexp(dim=1, keepdim=True)
    return -(logp * pos).sum() / pos.sum().clamp_min(1)

# ----------------------------------------------------------------------------
#  Model ----------------------------------------------------------------------
# ----------------------------------------------------------------------------
class Qwen3ForSequenceClassificationCL(Qwen3Model):
    config_class = Qwen3Config
    _no_split_modules = ["QwenBlock", "Qwen3Block"]
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids=None, attention_mask=None, position_ids=None,
        inputs_embeds=None, pixel_values=None, past_key_values=None,
        labels=None, use_cache=None,
        # contrastive
        contrastive_labels=None, lambda_cl: float = 1.0, temperature: float = 0.07,
        # adversarial stub
        adversarial_inputs=None, lambda_adv: float = 1.0,
        # HF
        output_attentions=None, output_hidden_states=None, return_dict=None,
    ):
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        out = super().forward(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            inputs_embeds=inputs_embeds, pixel_values=pixel_values, past_key_values=past_key_values,
            use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=True,
        )
        token_logits = self.score(out.last_hidden_state)
        pooled = _pool_last(token_logits, input_ids, getattr(self.config, "pad_token_id", None))

        loss = _compute_standard_loss(self.config, pooled, labels, self.num_labels)
        if contrastive_labels is not None:
            cls_feats = out.last_hidden_state[:, 0, :]
            cl_loss = _contrastive_loss(cls_feats, contrastive_labels, temperature=temperature)
            loss = (loss or 0) + lambda_cl * cl_loss

        if adversarial_inputs is not None:
            with torch.no_grad():
                adv = self.forward(**adversarial_inputs, labels=None, contrastive_labels=None,
                                    adversarial_inputs=None, output_hidden_states=False, return_dict=True)
            adv_logits = adv.logits
            adv_labels = torch.ones_like(adv_logits) if self.num_labels == 1 else torch.full(
                (adv_logits.size(0),), 1, dtype=torch.long, device=adv_logits.device)
            adv_loss_fn = nn.BCEWithLogitsLoss() if self.num_labels == 1 else nn.CrossEntropyLoss()
            loss = (loss or 0) + lambda_adv * adv_loss_fn(adv_logits, adv_labels)

        if not return_dict:
            tup = (pooled,) + out[1:]
            return ((loss,) + tup) if loss is not None else tup
        return SequenceClassifierOutputWithPast(
            loss=loss, logits=pooled,
            past_key_values=getattr(out, "past_key_values", None),
            hidden_states=out.hidden_states if output_hidden_states else None,
            attentions=out.attentions if output_attentions else None,
        )

# ----------------------------------------------------------------------------
#  HF registry (safe duplicate) -----------------------------------------------
# ----------------------------------------------------------------------------
try:
    AutoModelForSequenceClassification.register(Qwen3Config, Qwen3ForSequenceClassificationCL, exist_ok=True)
except TypeError:  # transformers<4.52 has no exist_ok param → manual check
    mapping = AutoModelForSequenceClassification._model_mapping
    if Qwen3Config not in mapping:
        AutoModelForSequenceClassification.register(Qwen3Config, Qwen3ForSequenceClassificationCL)
