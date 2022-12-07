import torch
from torch import nn
from transformers import AutoModel
from multi_meta_ssd.models.downstream.dual_encoders.utils import mean_pooling, normalized_cls_token

class SBERTForRetrieval(nn.Module):
    """ SBERT dual encoder model for symmetric semantic search retrieval."""

    def __init__(self, config, trans_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        super().__init__()
        self.trans_model = AutoModel.from_pretrained(trans_model_name)

        self.cos_score_transformation = nn.Identity()
        self.loss_fct = nn.MSELoss()


    def forward(
            self,
            sent1_input_ids=None,
            sent1_attention_mask=None,
            sent1_token_type_ids=None,
            sent2_input_ids=None,
            sent2_attention_mask=None,
            sent2_token_type_ids=None,
            scores_gs=None,
            inference=False):

        outputs_sent1 = self.trans_model(input_ids=sent1_input_ids,
                                         attention_mask=sent1_attention_mask,
                                         token_type_ids=sent1_token_type_ids)

        sentence_sent1 = mean_pooling(outputs_sent1, sent1_attention_mask)
        sent1_encodings = normalized_cls_token(sentence_sent1)

        if inference:
            # In inference mode, only use the first tower to get the encodings.
            # Check how the precision of the model is computed from here in
            return sent1_encodings.cpu().detach().numpy()

        outputs_sent2 = self.trans_model(input_ids=sent2_input_ids,
                                         attention_mask=sent2_attention_mask,
                                         token_type_ids=sent2_token_type_ids)

        sentence_sent2 = mean_pooling(outputs_sent2, sent2_attention_mask)
        sent2_encodings = normalized_cls_token(sentence_sent2)

        ##############################################################################
        #
        # Old Code for pearson correlation loss
        #
        ##############################################################################
        # scores = torch.squeeze(torch.bmm(sent1_encodings.view(sent1_encodings.shape[0], 1, sent1_encodings.shape[1]), sent2_encodings.view(q_encodings.shape[0], sent2_encodings.shape[1], 1)), axis=1) 
        # scores = (scores - 0.0) / (5.0 - 0.0)
        # scores_gs = scores_gs.view(scores_gs.shape[0], 1)

        # vx = scores - torch.mean(scores)
        # vy = scores_gs - torch.mean(scores_gs)

        # loss = - vx * vy * torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2))

        output = self.cos_score_transformation(torch.cosine_similarity(sent1_encodings, sent2_encodings))
        loss = self.loss_fct(output, scores_gs.view(-1))

        sent1_encodings = sent1_encodings.cpu().detach().numpy()
        sent2_encodings = sent2_encodings.cpu().detach().numpy()

        del outputs_sent1
        del outputs_sent2

        return loss.mean(), sent1_encodings, sent2_encodings