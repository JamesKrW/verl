import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional

from transformers.modeling_outputs import TokenClassifierOutput
from transformers import KimiVLConfig
from transformers.modeling_kimi_vl import (
    KimiVLPreTrainedModel,
    MoonVitPretrainedModel,
    KimiVLMultiModalProjector,
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Model


class KimiVLForTokenClassification(KimiVLPreTrainedModel):
    """
    KimiVL Model with a token classification head on top (a linear layer on top of the hidden-states output)
    for tasks like Named-Entity-Recognition (NER).
    """
    def __init__(self, config: KimiVLConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Vision modules
        self.vision_tower = MoonVitPretrainedModel(config.vision_config)
        self.multi_modal_projector = KimiVLMultiModalProjector(config)

        # Language encoder (decoder-only) without generation head
        self.text_model = DeepseekV3Model(config.text_config)

        # Classification head
        classifier_dropout = (
            config.classifier_dropout
            if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None
            else getattr(config.text_config, "hidden_dropout_prob", 0.1)
        )
        self.dropout = nn.Dropout(classifier_dropout)
        hidden_size = config.text_config.hidden_size
        self.classifier = nn.Linear(hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def _merge_with_image_features(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Replace each image placeholder token embedding with its corresponding vision feature.
        """
        image_token_index = self.config.media_placeholder_token_id
        # flatten batch and seq dims
        batch_size, seq_length, embed_dim = inputs_embeds.shape
        flat_embeds = inputs_embeds.view(-1, embed_dim)
        flat_ids = input_ids.view(-1)

        mask = flat_ids == image_token_index
        if mask.sum().item() != image_features.size(0):
            raise ValueError(
                f"Mismatch image tokens ({mask.sum().item()}) vs features ({image_features.size(0)})"
            )
        flat_embeds[mask] = image_features
        return flat_embeds.view(batch_size, seq_length, embed_dim)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_hws: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> TokenClassifierOutput:
        # Prepare embeddings
        if inputs_embeds is None:
            # token embeddings
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

            # integrate image features if provided
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.vision_tower.dtype)
                # extract raw image tokens from vision tower
                vision_feats = self.vision_tower(pixel_values, grid_hws=image_grid_hws)
                # project to text hidden size
                image_feats = self.multi_modal_projector(vision_feats)
                # merge into inputs_embeds
                inputs_embeds = self._merge_with_image_features(
                    inputs_embeds, input_ids, image_feats
                )

        # Forward through text model
        outputs = self.text_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            output_hidden_states=False,
        )
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        sequence_output = outputs.last_hidden_state

        # Classification head
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                # Only compute loss on active tokens
                active_mask = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_mask,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )