import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class AttentionPool(nn.Module):
    """Attention-based pooling layer."""

    def __init__(self, hidden_size):
        super(AttentionPool, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(hidden_size, 1))
        nn.init.xavier_uniform_(
            self.attention_weights
        )  # https://pytorch.org/docs/stable/nn.init.html

    def forward(self, hidden_states):
        attention_scores = torch.matmul(hidden_states, self.attention_weights)
        attention_scores = torch.softmax(attention_scores, dim=1)
        pooled_output = torch.sum(hidden_states * attention_scores, dim=1)
        return pooled_output


class GeneformerMultiTask(nn.Module):
    def __init__(
        self,
        pretrained_path,
        num_labels_list,
        dropout_rate=0.1,
        use_task_weights=False,
        task_weights=None,
        max_layers_to_freeze=0,
        use_attention_pooling=False,
    ):
        super(GeneformerMultiTask, self).__init__()
        self.config = BertConfig.from_pretrained(pretrained_path)
        self.bert = BertModel(self.config)
        self.num_labels_list = num_labels_list
        self.use_task_weights = use_task_weights
        self.dropout = nn.Dropout(dropout_rate)
        self.use_attention_pooling = use_attention_pooling

        if use_task_weights and (
            task_weights is None or len(task_weights) != len(num_labels_list)
        ):
            raise ValueError(
                "Task weights must be defined and match the number of tasks when 'use_task_weights' is True."
            )
        self.task_weights = (
            task_weights if use_task_weights else [1.0] * len(num_labels_list)
        )

        # Freeze the specified initial layers
        for layer in self.bert.encoder.layer[:max_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

        self.attention_pool = (
            AttentionPool(self.config.hidden_size) if use_attention_pooling else None
        )

        self.classification_heads = nn.ModuleList(
            [
                nn.Linear(self.config.hidden_size, num_labels)
                for num_labels in num_labels_list
            ]
        )
        # initialization of the classification heads: https://pytorch.org/docs/stable/nn.init.html
        for head in self.classification_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        try:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        except Exception as e:
            raise RuntimeError(f"Error during BERT forward pass: {e}")

        sequence_output = outputs.last_hidden_state

        try:
            pooled_output = (
                self.attention_pool(sequence_output)
                if self.use_attention_pooling
                else sequence_output[:, 0, :]
            )
            pooled_output = self.dropout(pooled_output)
        except Exception as e:
            raise RuntimeError(f"Error during pooling and dropout: {e}")

        total_loss = 0
        logits = []
        losses = []

        for task_id, (head, num_labels) in enumerate(
            zip(self.classification_heads, self.num_labels_list)
        ):
            try:
                task_logits = head(pooled_output)
            except Exception as e:
                raise RuntimeError(
                    f"Error during forward pass of classification head {task_id}: {e}"
                )

            logits.append(task_logits)

            if labels is not None:
                try:
                    loss_fct = nn.CrossEntropyLoss()
                    task_loss = loss_fct(
                        task_logits.view(-1, num_labels), labels[task_id].view(-1)
                    )
                    if self.use_task_weights:
                        task_loss *= self.task_weights[task_id]
                    total_loss += task_loss
                    losses.append(task_loss.item())
                except Exception as e:
                    raise RuntimeError(
                        f"Error during loss computation for task {task_id}: {e}"
                    )

        return total_loss, logits, losses if labels is not None else logits