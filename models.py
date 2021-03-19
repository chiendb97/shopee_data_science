import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchcrf import CRF
from transformers import BertPreTrainedModel, RobertaConfig, RobertaModel
from losses import LabelSmoothingCrossEntropy


class RobertaForTokenClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, num_ner_labels, num_cf_labels, activation_function, loss_type):
        super(RobertaForTokenClassification, self).__init__(config)
        self.num_ner_labels = num_ner_labels
        self.num_cf_labels = num_cf_labels
        self.activation_function = activation_function
        assert self.activation_function in ['softmax', 'crf']
        self.loss_type = loss_type
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_ner = nn.Linear(4 * config.hidden_size, num_ner_labels)
        self.classifier_cf = nn.Linear(4 * config.hidden_size, num_cf_labels)
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
            labels_ner=None,
            labels_cf=None,
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
        output_ner = torch.cat((outputs[1][-1], outputs[1][-2], outputs[1][-3],
                                outputs[1][-4]), -1)

        output_cf = torch.cat((outputs[1][-1][:, 0, ...], outputs[1][-2][:, 0, ...], outputs[1][-3][:, 0, ...],
                               outputs[1][-4][:, 0, ...]), -1)

        output_ner = self.dropout(output_ner)
        output_cf = self.dropout(output_cf)

        logits_ner = self.classifier_ner(output_ner)
        logits_cf = self.classifier_cf(output_cf)

        if self.activation_function == 'softmax':
            loss_ner = None
            loss_cf = None
            if labels_ner is not None:
                assert self.loss_type in ['lsr', 'ce']
                if self.loss_type == "ce":
                    loss_fct = CrossEntropyLoss()
                else:
                    loss_fct = LabelSmoothingCrossEntropy()

                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits_ner.view(-1, self.num_ner_labels)
                    active_labels = torch.where(
                        active_loss, labels_ner.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels_ner)
                    )
                    loss_ner = loss_fct(active_logits, active_labels)
                else:
                    loss_ner = loss_fct(logits_ner.view(-1, self.num_ner_labels), labels_ner.view(-1))

                loss_cf = loss_fct(logits_cf, labels_cf)

        else:
            loss_ner = None
            loss_cf = None
            if labels_ner is not None:
                loss_fct = CrossEntropyLoss()
                loss_ner = self.crf(emissions=logits_ner, tags=labels_ner, mask=attention_mask, reduction='token_mean')
                loss_ner = -1.0 * loss_ner
                loss_cf = loss_fct(logits_cf, labels_cf)

        return (logits_ner, logits_cf, loss_ner, loss_cf) if loss_ner is not None else (logits_ner, logits_cf)
