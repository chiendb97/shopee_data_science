import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchcrf import CRF
from transformers import BertPreTrainedModel, RobertaConfig, RobertaModel
from losses import LabelSmoothingCrossEntropy


class RobertaForTokenClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, activation_function, loss_type):
        super(RobertaForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.activation_function = activation_function
        assert self.activation_function in ['softmax', 'crf']
        self.loss_type = loss_type
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(4 * config.hidden_size, config.num_labels)
        if self.activation_function == "crf":
            self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # sequence_output = outputs[0]
        sequence_output = torch.cat((outputs[1][-1], outputs[1][-2], outputs[1][-3],
                                     outputs[1][-4]), -1)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if self.activation_function == 'softmax':
            loss = None
            if labels is not None:
                assert self.loss_type in ['lsr', 'ce']
                if self.loss_type == "ce":
                    loss_fct = CrossEntropyLoss()
                else:
                    loss_fct = LabelSmoothingCrossEntropy()

                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        else:
            loss = None
            if labels is not None:
                loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)

        return (logits, loss) if loss is not None else logits



