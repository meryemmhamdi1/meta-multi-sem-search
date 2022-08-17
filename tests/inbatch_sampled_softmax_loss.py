import torch
from transformers import BertConfig, BertTokenizer, XLMRobertaTokenizer

def normalized_cls_token(cls_token):
    return torch.nn.functional.normalize(cls_token, p=2, dim=1)

def in_batch_sampled_softmax_loss(outputs_q,
                                  outputs_a,
                                  outputs_n):

        q_encodings = normalized_cls_token(outputs_q)
        a_encodings = normalized_cls_token(outputs_a)
        n_encodings = normalized_cls_token(outputs_n)

        # q_encodings = outputs_q
        # a_encodings = outputs_a
        # n_encodings = outputs_n

        print("q_encodings:", q_encodings)
        print("a_encodings:", a_encodings)
        print("n_encodings:", n_encodings)

        similarity_a = torch.matmul(q_encodings,
                                    torch.transpose(a_encodings, 0, 1))

        similarity_n = torch.matmul(q_encodings,
                                    torch.transpose(n_encodings, 0, 1))

        batch_size = list(q_encodings.size())[0]
        # labels = torch.arange(0, batch_size, device=logits.device)
        other_loss = - (1/batch_size) * torch.sum(similarity_a - torch.log(torch.sum(torch.exp(similarity_n))))
        loss = - torch.sum(similarity_a - torch.logsumexp(similarity_n, 1))
        return other_loss, loss, q_encodings, a_encodings, n_encodings, similarity_a


outputs_q = torch.tensor([[0, 1, 1, 1]], dtype=float)
outputs_a = torch.tensor([[0, 100, 100, 100]], dtype=float)
outputs_n = torch.tensor([[0, 1, 1, 2], [0, 1, 1, 3]], dtype=float)
other_loss, loss, q_encodings, q_encodings, n_encodings, similarity_a = in_batch_sampled_softmax_loss(outputs_q,
                                                                            outputs_a,
                                                                            outputs_n)

print("other_loss:", other_loss, " loss:", loss, " similarity_a:", similarity_a)



import numpy as np
print(np.dot([0, 1, 1, 1],[0, 100, 100, 100]) - np.log(np.exp(np.dot([0, 1, 1, 1], [0, 1, 1, 2]))+np.exp(np.dot([0, 1, 1, 1], [0, 1, 1, 3]))))
print(loss)