import torch
from torch import nn
from transformers import AutoModel
from multi_meta_ssd.models.downstream.dual_encoders.utils import mean_pooling, normalized_cls_token

class SBERTForRetrieval(nn.Module):
    """ SBERT dual encoder model for retrieval."""

    def __init__(self, config, trans_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        super().__init__()
        self.trans_model = AutoModel.from_pretrained(trans_model_name)#)
        self.contrastive = False

        self.logit_scale = torch.nn.Parameter(torch.empty(1))
        torch.nn.init.constant_(self.logit_scale, 100.0)
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(
            self,
            q_input_ids=None,
            q_attention_mask=None,
            q_token_type_ids=None,
            a_input_ids=None,
            a_attention_mask=None,
            a_token_type_ids=None,
            n_input_ids=None,
            n_attention_mask=None,
            n_token_type_ids=None,
            inference=False):

        outputs_q = self.trans_model(input_ids=q_input_ids,
                                     attention_mask=q_attention_mask,
                                     token_type_ids=q_token_type_ids)

        sentence_q = mean_pooling(outputs_q, q_attention_mask)
        q_encodings = normalized_cls_token(sentence_q)

        if inference:
            # In inference mode, only use the first tower to get the encodings.
            # Check how the precision of the model is computed from here in
            return q_encodings.cpu().detach().numpy()

        outputs_a = self.trans_model(input_ids=a_input_ids,
                                     attention_mask=a_attention_mask,
                                     token_type_ids=a_token_type_ids)

        sentence_a = mean_pooling(outputs_a, a_attention_mask)
        a_encodings = normalized_cls_token(sentence_a)


        outputs_n = self.trans_model(input_ids=n_input_ids,
                                     attention_mask=n_attention_mask,
                                     token_type_ids=n_token_type_ids)
        sentence_n = mean_pooling(outputs_n, n_attention_mask)
        n_encodings = normalized_cls_token(sentence_n)

        ## anchors, positive and negative triplets need to be aligned and same shape
        # print("q_encodings.shape:", q_encodings.shape, 
        #       " a_encodings.shape:", a_encodings.shape, 
        #       " n_encodings.shape:", n_encodings.shape)

        if self.contrastive:
            similarity = torch.matmul(
                q_encodings, torch.transpose(a_encodings, 0, 1))
            logits = similarity * self.logit_scale
            batch_size = list(q_encodings.size())[0]
            labels = torch.arange(0, batch_size, device=logits.device) # contrastive loss
            loss = torch.nn.CrossEntropyLoss()(logits, labels)

        else:
            loss = self.triplet_loss(q_encodings, a_encodings, n_encodings)

        q_encodings = q_encodings.cpu().detach().numpy()
        a_encodings = a_encodings.cpu().detach().numpy()
        n_encodings = n_encodings.cpu().detach().numpy()

        del outputs_a
        del outputs_n
        del outputs_q

        return loss, q_encodings, a_encodings, n_encodings