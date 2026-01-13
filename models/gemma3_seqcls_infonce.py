# gemma3_seqcls.py ─────────────────────────────────────────────────────
"""
Universal Gemma-3 sequence-classification head with
• BCE / CE  (기존)
• Contrastive Learning (margin α)
• Adversarial Training  (Generator 텍스트)
"""

import torch, torch.nn as nn
from transformers import AutoModelForSequenceClassification
from transformers.models.gemma3.configuration_gemma3 import (
    Gemma3Config, Gemma3TextConfig
)
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Model, Gemma3TextModel, Gemma3PreTrainedModel
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


# ───────────────────── helpers ─────────────────────────────────────────
def _txt(cfg):  # text 하위 config
    return getattr(cfg, "text_config", cfg)


def _pool_last(token_logits, input_ids, pad_id):
    """문장 끝 토큰 위치의 로짓을 pooling(문자열 길이 가변 대응)."""
    if input_ids is None or pad_id is None:
        return token_logits[:, -1]
    ends = (~input_ids.eq(pad_id)).cumsum(-1).argmax(-1)
    return token_logits[torch.arange(token_logits.size(0)), ends]


def _compute_standard_loss(config, logits, labels, num_labels):
    """기존 BCE/CE/MSE 로직(변경 없음)."""
    if labels is None:
        return None
    if config.problem_type is None:
        if num_labels == 1:
            config.problem_type = "regression"
        elif labels.dtype in (torch.long, torch.int):
            config.problem_type = "single_label_classification"
        else:
            config.problem_type = "multi_label_classification"

    if config.problem_type == "regression":
        return nn.MSELoss()(logits.squeeze(), labels.squeeze())
    if config.problem_type == "single_label_classification":
        return nn.CrossEntropyLoss()(logits, labels)
    return nn.BCEWithLogitsLoss()(logits, labels)


# ───────────────── contrastive loss (코사인 + InfoNCE) ───────────────
def _contrastive_loss(features, labels, temperature: float = 0.07):
    """
    NT-Xent(InfoNCE) 방식:
      • features  : [B, D]  (CLS 임베딩)
      • labels    : [B]     (0/1)
    """
    # ① 임베딩 L2 정규화 → 코사인 유사도
    f = nn.functional.normalize(features.float(), dim=-1)   # BF16 → FP32 안전
    sim = torch.matmul(f, f.T)                              # [-1,1]
    sim = sim / temperature                                 # softmax 스케일 조절

    # ② 양/음 마스크
    labels = labels.view(-1)
    pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)   # same class → True
    pos_mask.fill_diagonal_(False)                          # ←★ 추가
    neg_mask = ~pos_mask

    # ③ self-sim 억제
    sim = sim - torch.eye(sim.size(0), device=sim.device) * 1e9

    # ④ InfoNCE loss
    log_prob = sim - sim.logsumexp(dim=1, keepdim=True)
    loss = -(log_prob * pos_mask).sum() / pos_mask.sum().clamp_min(1)
    return loss



# ───────────────────── Base mixin (공통 forward) ──────────────────────
class _Gemma3SeqClsMixin:
    """텍스트/멀티모달 공통 forward 로직을 mixin 형태로 분리."""

    def _shared_forward(
        self,
        backbone_out, input_ids,
        labels=None,
        # contrastive learning
        contrastive_labels=None,
        lambda_cl: float = 1.0,
        margin: float = 0.5,
        # adversarial training
        adversarial_inputs: dict = None,
        lambda_adv: float = 1.0,
        # misc
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )

        token_logits = self.score(backbone_out.last_hidden_state)
        pad_id = getattr(
            self.config,
            "pad_token_id",
            getattr(_txt(self.config), "pad_token_id", None),
        )
        pooled = _pool_last(token_logits, input_ids, pad_id)  # [B, num_labels]

        # ─── 손실 계산 ───────────────────────────────────────────────
        loss_total = _compute_standard_loss(
            self.config, pooled, labels, self.num_labels
        )
        cl_loss = adv_loss = None

        # Contrastive Loss
        if contrastive_labels is not None:
            # features: CLS hidden state
            features = backbone_out.last_hidden_state[:, 0, :]   # CLS 토큰
            cl_loss = _contrastive_loss(features, contrastive_labels, margin)
            loss_total = (
                loss_total + lambda_cl * cl_loss
                if loss_total is not None
                else lambda_cl * cl_loss
            )

        # Adversarial Loss (Generator 텍스트 logits 필요)
        # if adversarial_inputs is not None:
        #     with torch.no_grad():  # 동일 가중치 공유, gradient는 분류기에만
        #         adv_out = self.forward(**adversarial_inputs, labels=None,
        #                                contrastive_labels=None,
        #                                adversarial_inputs=None,
        #                                output_hidden_states=False,
        #                                return_dict=True)
        #     adv_logits = adv_out.logits
        #     adv_labels = torch.ones_like(adv_logits)  # Generator → AI(1)
        #     adv_loss_fn = nn.BCEWithLogitsLoss()
        #     adv_loss = adv_loss_fn(adv_logits, adv_labels)
        #     loss_total = (
        #         loss_total + lambda_adv * adv_loss
        #         if loss_total is not None
        #         else lambda_adv * adv_loss
        #     )
        # ── Adversarial Loss (Generator 텍스트) ───────────────────────────
        if adversarial_inputs is not None:
            with torch.no_grad():
                adv_out = self.forward(
                    **adversarial_inputs,
                    labels=None,
                    contrastive_labels=None,
                    adversarial_inputs=None,          # 재귀 방지
                    output_hidden_states=False,
                    return_dict=True,
                )
            adv_logits = adv_out.logits                         # [B, num_labels]

            if self.num_labels == 1:  # ▶ BCE 방식
                adv_labels = torch.ones_like(adv_logits)        # [B,1] float
                adv_loss_fn = nn.BCEWithLogitsLoss()
            else:              # ▶ CrossEntropy 방식(num_labels≥2)
                adv_labels = torch.full(                        # [B] long, all 1
                    (adv_logits.size(0),),
                    1, dtype=torch.long, device=adv_logits.device
                )
                adv_loss_fn = nn.CrossEntropyLoss()

            adv_loss = adv_loss_fn(adv_logits, adv_labels)
            loss_total = loss_total + lambda_adv * adv_loss if loss_total is not None else lambda_adv * adv_loss


        # ─── 반환 ───────────────────────────────────────────────────
        if not return_dict:
            out = (pooled,) + backbone_out[1:]
            return ((loss_total,) + out) if loss_total is not None else out

        return SequenceClassifierOutputWithPast(
            loss=loss_total,
            logits=pooled,
            past_key_values=backbone_out.past_key_values,
            hidden_states=backbone_out.hidden_states
            if output_hidden_states
            else None,
            attentions=backbone_out.attentions if output_attentions else None,
            # cl_loss=cl_loss,
            # adv_loss=adv_loss,
        )


# ───────────────────── multimodal variant (4B+) ───────────────────────
class Gemma3ForSequenceClassification(Gemma3Model, _Gemma3SeqClsMixin):
    config_class = Gemma3Config
    _no_split_modules = ["GemmaBlock"]
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, config):
        super().__init__(config)  # Gemma3 backbone
        self.num_labels = config.num_labels
        self.score = nn.Linear(_txt(config).hidden_size,
                               self.num_labels, bias=False)
        self.post_init()

    # ※ NEW: contrastive/adversarial 파라미터 추가
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        pixel_values=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        # ↓↓↓ 추가된 인자 ↓↓↓
        contrastive_labels=None,
        lambda_cl: float = 1.0,
        margin: float = 0.5,
        adversarial_inputs: dict = None,
        lambda_adv: float = 1.0,
        # ↓↓↓ 이하 기존 HF 인자 유지 ↓↓↓
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # contrastive가 요청되면 hidden_states 강제 ON
        # if contrastive_labels is not None:
        #     output_hidden_states = True

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        return self._shared_forward(
            outputs, input_ids,
            labels=labels,
            contrastive_labels=contrastive_labels,
            lambda_cl=lambda_cl,
            margin=margin,
            adversarial_inputs=adversarial_inputs,
            lambda_adv=lambda_adv,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# ───────────────────── text-only variant (1B) ─────────────────────────
class Gemma3TextForSequenceClassification(Gemma3PreTrainedModel,
                                           _Gemma3SeqClsMixin):
    """Wraps Gemma3TextModel in self.model."""
    config_class = Gemma3TextConfig
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma3TextModel(config)     # backbone
        self.score = nn.Linear(config.hidden_size,
                               self.num_labels, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        # ↓ contrastive / adversarial ↓
        contrastive_labels=None,
        lambda_cl: float = 1.0,
        margin: float = 0.5,
        adversarial_inputs: dict = None,
        lambda_adv: float = 1.0,
        # ↓ HF ↓
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # if contrastive_labels is not None:
        #     output_hidden_states = True

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        return self._shared_forward(
            outputs, input_ids,
            labels=labels,
            contrastive_labels=contrastive_labels,
            lambda_cl=lambda_cl,
            margin=margin,
            adversarial_inputs=adversarial_inputs,
            lambda_adv=lambda_adv,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# ───────────────────── HF factory 등록 ────────────────────────────────
AutoModelForSequenceClassification.register(
    Gemma3Config, Gemma3ForSequenceClassification)
AutoModelForSequenceClassification.register(
    Gemma3TextConfig, Gemma3TextForSequenceClassification)
# ───────────────────────────────────────────────────────────────────────
