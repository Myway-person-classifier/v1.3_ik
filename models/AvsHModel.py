import torch
from torch import nn
from transformers import AutoModel, AutoConfig

class AvsHModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load the embedding model
        self.embeddding_model = AutoModel.from_pretrained(args.embedding_model)
        self.config = self.embeddding_model.config
        
        # learnable Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size))

        # paragraph attend Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size,
            nhead=args.num_heads,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            activation="relu",
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)

        print(f"Using TransformerEncoder with {args.num_layers} layers, {args.num_heads} heads, "
        f"feedforward dimension {args.dim_feedforward}, dropout {args.dropout}",
        f"hidden size {self.config.hidden_size}")
        print(f"Parameters in embedding model: {get_check_parameters(self.embeddding_model)}")
        print(f"Parameters in transformer encoder: {get_check_parameters(self.transformer_encoder)}")

        # Classifier layer
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, chunk_size: int = 12, *args, **kwargs):
        """
        chunk_size : 한 번에 embeddding_model 로 보낼 문단 수
                    (메모리 상황에 맞게 조절하세요)
        """
        
        batch_size, num_paragraphs, seq_length = input_ids.size()

        cls_embedding = []

        # --------- 여기만 바꿈 (chunk 단위로 loop) ---------
        for start in range(0, num_paragraphs, chunk_size):
            end = min(start + chunk_size, num_paragraphs)

            # [B, C, L] → [B*C, L]
            input_chunk = input_ids[:, start:end, :].contiguous().view(-1, seq_length)
            attn_chunk  = attention_mask[:, start:end, :].contiguous().view(-1, seq_length)

            outputs = self.embeddding_model(
                input_ids=input_chunk,
                attention_mask=attn_chunk
            )

            # [B*C, D] → [B, C, D]  (C = end-start)
            cls_chunk = outputs.last_hidden_state[:, 0, :].view(batch_size, end - start, -1)
            cls_embedding.append(cls_chunk)
        # -----------------------------------------------

        cls_embedding = torch.cat(cls_embedding, dim=1)  # [B, P, D]

        # Create learnable CLS token per batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]

        # Concatenate: [B, 1 + P, D]
        hidden_states = torch.cat([cls_tokens, cls_embedding], dim=1)

        # Create paragraph-level mask (1 if paragraph has non-zero attention_mask, else 0)
        para_mask = (attention_mask.view(batch_size, num_paragraphs, seq_length)
                                .sum(dim=2) > 0).long()
        cls_mask = torch.ones(batch_size, 1, device=input_ids.device, dtype=para_mask.dtype)
        attention_mask = torch.cat([cls_mask, para_mask], dim=1)  # [B, 1 + P]

        hidden_states = self.transformer_encoder(
            hidden_states,
            src_key_padding_mask=~attention_mask.bool()
        )
        logits = self.classifier(hidden_states[:, 0, :])  # use the first token
        pargraph_logits = self.classifier(hidden_states[:, 1:, :])  # use the rest tokens for paragraph logits
        return logits, pargraph_logits


def get_check_parameters(model):
    """
    Returns the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)