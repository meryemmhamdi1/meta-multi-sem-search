"""BERT dual encoder model for retrieval (from XTREME https://github.com/google-research/xtreme) """

import torch
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
# ---> old from transformers.modeling_bert import BertModel, BertPreTrainedModel


class BertForRetrieval(BertPreTrainedModel):
    """ BERT dual encoder model for retrieval."""

    def __init__(self, config, model_attr_name='bert', model_cls=BertModel):
        super().__init__(config)

        # print("BertForRetrieval model_attr_name:", model_attr_name)
        # self.trans_model = BertModel.from_pretrained('bert-base-multilingual-cased')

        self.model_attr_name = model_attr_name
        self.model_cls = model_cls
        self.contrastive = False

        # Set model attribute, e.g. self.bert = BertModel(config)
        setattr(self, model_attr_name, model_cls(config))

        def normalized_cls_token(cls_token):
            return torch.nn.functional.normalize(cls_token, p=2, dim=1)

        self.normalized_cls_token = normalized_cls_token
        self.logit_scale = torch.nn.Parameter(torch.empty(1))
        torch.nn.init.constant_(self.logit_scale, 100.0)
        self.init_weights()
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    def model(self):
        return getattr(self, self.model_attr_name)

    def forward_original(
        self,
        q_input_ids=None,
        q_attention_mask=None,
        q_token_type_ids=None,
        a_input_ids=None,
        a_attention_mask=None,
        a_token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        inference=False):

        outputs_a = self.model()(
            q_input_ids,
            attention_mask=q_attention_mask,
            token_type_ids=q_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds)
        if inference:
            # In inference mode, only use the first tower to get the encodings.
            return self.normalized_cls_token(outputs_a[1])

        outputs_b = self.model()(
            a_input_ids,
            attention_mask=a_attention_mask,
            token_type_ids=a_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds)

        a_encodings = self.normalized_cls_token(outputs_a[1])
        b_encodings = self.normalized_cls_token(outputs_b[1])
        similarity = torch.matmul(
            a_encodings, torch.transpose(b_encodings, 0, 1))
        logits = similarity * self.logit_scale
        batch_size = list(a_encodings.size())[0]
        labels = torch.arange(0, batch_size, device=logits.device) # contrastive loss
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return loss, a_encodings, b_encodings

    def forward_contrastive(
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
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            inference=False):

        print("CONTRASTIVE inference ", inference)

        outputs_q = self.model()(
            q_input_ids,
            attention_mask=q_attention_mask,
            token_type_ids=q_token_type_ids,
            position_ids=position_ids, # TODO DOUBLE CHECK THESE
            head_mask=head_mask, # TODO DOUBLE CHECK THESE
            inputs_embeds=inputs_embeds)

        if inference:
            # In inference mode, only use the first tower to get the encodings.
            # Check how the precision of the model is computed from here in
            return self.normalized_cls_token(outputs_q[1]).cpu().detach().numpy()

        outputs_a = self.model()(
            a_input_ids,
            attention_mask=a_attention_mask,
            token_type_ids=a_token_type_ids,
            position_ids=position_ids, # TODO DOUBLE CHECK THESE
            head_mask=head_mask, # TODO DOUBLE CHECK THESE
            inputs_embeds=inputs_embeds)

        outputs_n = self.model()(
            n_input_ids,
            attention_mask=n_attention_mask,
            token_type_ids=n_token_type_ids,
            position_ids=position_ids, # TODO DOUBLE CHECK THESE
            head_mask=head_mask, # TODO DOUBLE CHECK THESE
            inputs_embeds=inputs_embeds)

        q_encodings = self.normalized_cls_token(outputs_q[1])
        a_encodings = self.normalized_cls_token(outputs_a[1])
        n_encodings = self.normalized_cls_token(outputs_n[1])

        similarity_a = torch.matmul(q_encodings, torch.transpose(a_encodings, 0, 1))
        similarity_n = torch.matmul(q_encodings, torch.transpose(n_encodings, 0, 1))

        q_encodings = q_encodings.cpu().detach().numpy()
        a_encodings = a_encodings.cpu().detach().numpy()
        n_encodings = n_encodings.cpu().detach().numpy()

        loss = - torch.sum(similarity_a - torch.logsumexp(similarity_n, 1))
        del outputs_a
        del outputs_n
        del outputs_q

        del similarity_a
        del similarity_n

        # loss = - (1/self.logit_scale) * torch.sum(similarity_a - torch.log(torch.sum(torch.exp(similarity_n))))
        return loss, q_encodings, a_encodings, n_encodings

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
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            inference=False):

        # print("FORWARD")

        # print("outputs_q computation => ", " q_input_ids.shape:", q_input_ids.shape, 
        #       " q_attention_mask.shape: ", q_attention_mask.shape, " q_token_type_ids:", q_token_type_ids.shape,
        #       " position_ids:", position_ids, " head_mask:", head_mask, 
        #       " inputs_embeds:", inputs_embeds)

        # print("self.model:", self.model)

        outputs_q = self.model()(q_input_ids, # self.model()
                                 attention_mask=q_attention_mask,
                                 token_type_ids=q_token_type_ids,
                                 position_ids=position_ids, # TODO DOUBLE CHECK THESE
                                 head_mask=head_mask, # TODO DOUBLE CHECK THESE
                                 inputs_embeds=inputs_embeds)

        if inference:
            # In inference mode, only use the first tower to get the encodings.
            # Check how the precision of the model is computed from here in
            return self.normalized_cls_token(outputs_q[1]).cpu().detach().numpy()
        
        # print("outputs_a computation => ", " a_input_ids.shape:", a_input_ids.shape, 
        #       " a_attention_mask.shape: ", a_attention_mask.shape, " a_token_type_ids:", a_token_type_ids.shape,
        #       " position_ids.shape:", position_ids, " head_mask.shape:", head_mask, 
        #       " inputs_embeds.shape:", inputs_embeds)

        outputs_a = self.model()(a_input_ids,
                                 attention_mask=a_attention_mask,
                                 token_type_ids=a_token_type_ids,
                                 position_ids=position_ids, # TODO DOUBLE CHECK THESE
                                 head_mask=head_mask, # TODO DOUBLE CHECK THESE
                                 inputs_embeds=inputs_embeds)

        # print("outputs_n computation => ")
        # print("n_input_ids.shape:", n_input_ids.shape)

        outputs_n = self.model()(
            n_input_ids,
            attention_mask=n_attention_mask,
            token_type_ids=n_token_type_ids,
            position_ids=position_ids, # TODO DOUBLE CHECK THESE
            head_mask=head_mask, # TODO DOUBLE CHECK THESE
            inputs_embeds=inputs_embeds)

        # print("Normalization => ")

        q_encodings = self.normalized_cls_token(outputs_q[1])
        a_encodings = self.normalized_cls_token(outputs_a[1])
        n_encodings = self.normalized_cls_token(outputs_n[1])
        ## anchors, positive and negative triplets need to be aligned and same shape

        if self.contrastive:
            similarity = torch.matmul(
                q_encodings, torch.transpose(a_encodings, 0, 1))
            logits = similarity * self.logit_scale
            batch_size = list(q_encodings.size())[0]
            labels = torch.arange(0, batch_size, device=logits.device) # contrastive loss
            loss = torch.nn.CrossEntropyLoss()(logits, labels)

        else:
            # print("Triplet Loss => ")
            loss = self.triplet_loss(q_encodings, a_encodings, n_encodings)


        q_encodings = q_encodings.cpu().detach().numpy()
        a_encodings = a_encodings.cpu().detach().numpy()
        n_encodings = n_encodings.cpu().detach().numpy()

        # print("q_encodings.shape:", q_encodings.shape, 
        #       " a_encodings.shape:", a_encodings.shape, 
        #       " n_encodings.shape:", n_encodings.shape)

        del outputs_a
        del outputs_n
        del outputs_q

        return loss, q_encodings, a_encodings, n_encodings
