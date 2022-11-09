import torch
from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
#     sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#     return sum_embeddings / sum_mask


class SBERTForRetrieval(BertPreTrainedModel):
    """ SBERT dual encoder model for retrieval."""

    def __init__(self, config, trans_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        super().__init__(config)
        self.trans_model = AutoModel.from_pretrained(trans_model_name)#)
        self.contrastive = False

        def normalized_cls_token(cls_token):
            return torch.nn.functional.normalize(cls_token, p=2, dim=1)

        self.normalized_cls_token = normalized_cls_token
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
            scores_gs=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            inference=False):

        outputs_q = self.trans_model(input_ids=q_input_ids,
                                     attention_mask=q_attention_mask,
                                     token_type_ids=q_token_type_ids,
                                     position_ids=position_ids, # TODO DOUBLE CHECK THESE
                                     head_mask=head_mask, # TODO DOUBLE CHECK THESE
                                     inputs_embeds=inputs_embeds)

        sentence_q = mean_pooling(outputs_q, q_attention_mask)

        if inference:
            # In inference mode, only use the first tower to get the encodings.
            # Check how the precision of the model is computed from here in
            return self.normalized_cls_token(sentence_q).cpu().detach().numpy()
            # return sentence_q.cpu().detach().numpy()

        outputs_a = self.trans_model(input_ids=a_input_ids,
                                     attention_mask=a_attention_mask,
                                     token_type_ids=a_token_type_ids,
                                     position_ids=position_ids, # TODO DOUBLE CHECK THESE
                                     head_mask=head_mask, # TODO DOUBLE CHECK THESE
                                     inputs_embeds=inputs_embeds)

        sentence_a = mean_pooling(outputs_a, a_attention_mask)

        q_encodings = self.normalized_cls_token(sentence_q)
        a_encodings = self.normalized_cls_token(sentence_a)
        ## anchors, positive and negative triplets need to be aligned and same shape

        # print("q_encodings.shape:", q_encodings.shape)
        # print("a_encodings.shape:", a_encodings.shape)

        scores = torch.squeeze(torch.bmm(q_encodings.view(q_encodings.shape[0], 1, q_encodings.shape[1]), a_encodings.view(q_encodings.shape[0], q_encodings.shape[1], 1)), axis=1) 
        # scores = q_encodings.dot(a_encodings.T)[0][0]
        scores = (scores - 0.0) / (5.0 - 0.0)
        scores_gs = scores_gs.view(scores_gs.shape[0], 1)

        # print("scores.shape:", scores.shape, scores)
        # print("scores_gs.shape:", scores_gs.shape, scores_gs)

        vx = scores - torch.mean(scores)
        vy = scores_gs - torch.mean(scores_gs)

        # print("vx:", vx, " vy:", vy)

        loss = - vx * vy * torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2))

        q_encodings = q_encodings.cpu().detach().numpy()
        a_encodings = a_encodings.cpu().detach().numpy()

        del outputs_a
        del outputs_q

        return loss.mean(), q_encodings, a_encodings