import os
import json
import torch
from torch import nn
from transformers import BertTokenizerFast, BertModel

class CRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _log_sum_exp(self, scores):
        max_score, _ = scores.max(dim=1)
        return max_score + torch.log(torch.sum(torch.exp(scores - max_score.unsqueeze(1)), dim=1))

    def forward(self, emissions, tags=None, mask=None, reduction='mean'):
        if tags is not None:
            nll = self._compute_loss(emissions, tags, mask)
            if reduction == 'mean':
                return nll.mean()
            elif reduction == 'sum':
                return nll.sum()
            else:
                return nll
        else:
            return self.decode(emissions, mask)

    def _compute_loss(self, emissions, tags, mask):
        batch_size, seq_length, num_tags = emissions.size()
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0, :].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for t in range(1, seq_length):
            mask_t = mask[:, t]
            emit_t = emissions[:, t, :].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            trans_t = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + emit_t * mask_t + trans_t * mask_t
        seq_end = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, seq_end.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        partition = self._compute_log_partition(emissions, mask)
        return partition - score

    def _compute_log_partition(self, emissions, mask):
        batch_size, seq_length, num_tags = emissions.size()
        alpha = self.start_transitions + emissions[:, 0, :]
        for t in range(1, seq_length):
            emit_t = emissions[:, t, :]
            mask_t = mask[:, t].unsqueeze(1).bool()
            alpha_t = []
            for next_tag in range(num_tags):
                score = self.transitions[:, next_tag].unsqueeze(0) + alpha
                alpha_t.append(self._log_sum_exp(score) + emit_t[:, next_tag])
            alpha_candidate = torch.stack(alpha_t, dim=1)
            alpha = torch.where(mask_t, alpha_candidate, alpha)
        alpha = alpha + self.end_transitions
        return self._log_sum_exp(alpha)

    def decode(self, emissions, mask):
        batch_size, seq_length, num_tags = emissions.size()
        viterbi_score = self.start_transitions + emissions[:, 0, :]
        viterbi_path = torch.zeros(batch_size, seq_length, num_tags, dtype=torch.long)
        for t in range(1, seq_length):
            emit_t = emissions[:, t, :]
            score_t = viterbi_score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_path = score_t.max(dim=1)
            viterbi_score = best_score + emit_t
            viterbi_path[:, t, :] = best_path
        viterbi_score = viterbi_score + self.end_transitions
        best_last_score, best_last_tag = viterbi_score.max(dim=1)
        best_paths = []
        for i in range(batch_size):
            path = [best_last_tag[i].item()]
            for t in range(seq_length - 1, 0, -1):
                path.insert(0, viterbi_path[i, t, path[0]].item())
            best_paths.append(path)
        return best_paths

class BertCRFTagger(nn.Module):
    def __init__(self, num_labels: int, pretrained_model_name: str = "bert-base-chinese"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)
        if labels is not None:
            loss = self.crf(emissions, tags=labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=attention_mask.bool())

class NERPipeline:
    def __init__(self, model_dir: str):
        with open(os.path.join(model_dir, 'label2id.json'), 'r', encoding='utf-8') as f:
            self.label2id = json.load(f)
        with open(os.path.join(model_dir, 'id2label.json'), 'r', encoding='utf-8') as f:
            self.id2label = json.load(f)
        with open(os.path.join(model_dir, 'config.json'), 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        self.tokenizer = BertTokenizerFast.from_pretrained(cfg.get('pretrained', 'bert-base-chinese'))
        self.model = BertCRFTagger(num_labels=len(self.label2id), pretrained_model_name=cfg.get('pretrained', 'bert-base-chinese'))
        state_dict = torch.load(os.path.join(model_dir, 'pytorch_model.bin'), map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, text: str):
        enc = self.tokenizer(text, return_offsets_mapping=True, return_tensors='pt')
        with torch.no_grad():
            pred = self.model(enc['input_ids'], enc['attention_mask'])
        tags = pred[0]
        offsets = enc['offset_mapping'][0].tolist()
        result = []
        current = None
        for i, tag_id in enumerate(tags):
            label = self.id2label[str(tag_id)]
            if label == 'O':
                if current and current['text']:
                    result.append(current)
                    current = None
                continue
            if label.startswith('B-'):
                if current and current['text']:
                    result.append(current)
                current = {'label': label[2:], 'start': offsets[i][0], 'end': offsets[i][1], 'text': text[offsets[i][0]:offsets[i][1]]}
            elif label.startswith('I-'):
                if current and current['label'] == label[2:]:
                    if offsets[i][0] >= current['end']:
                        current['end'] = offsets[i][1]
                        current['text'] = text[current['start']:current['end']]
                else:
                    current = {'label': label[2:], 'start': offsets[i][0], 'end': offsets[i][1], 'text': text[offsets[i][0]:offsets[i][1]]}
        if current and current['text']:
            result.append(current)
        return result

_NER_SINGLETON = None

def get_ner(model_dir: str = None):
    global _NER_SINGLETON
    if _NER_SINGLETON is not None:
        return _NER_SINGLETON
    base = os.path.join('data', 'models', 'ner_bert_crf')
    path = model_dir or base
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'pytorch_model.bin')):
        _NER_SINGLETON = NERPipeline(path)
        return _NER_SINGLETON
    return None

