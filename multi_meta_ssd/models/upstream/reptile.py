from multi_meta_ssd.models.upstream.meta_generic import MetaGeneric
from multi_meta_ssd.processors.downstream.utils_lareqa import mean_avg_prec_at_k_meta
from multi_meta_ssd.log import *
import torch
import copy
import numpy as np
from tqdm import tqdm


class MetaLearner(MetaGeneric):
    def __init__(self,
                 tokenizer,
                 base_model,
                 device,
                 meta_learn_config):

        super(MetaLearner, self).__init__(tokenizer,
                                          base_model,
                                          device,
                                          meta_learn_config)

        self.adapt_opt = torch.optim.Adam(self.base_model.parameters(),
                                          lr=self.splits_params["train"]["alpha_lr"],
                                          betas=(0, 0.999))

        self.opt = torch.optim.Adam(self.base_model.parameters(),
                                    lr=self.splits_params["train"]["beta_lr"])

        self.adapt_opt_state = self.adapt_opt.state_dict()

    def forward(self,  split_name, meta_tasks_batch, ep, batch_step, writer):
        n_tasks_total = self.splits_params[split_name]["n_tasks_total"]
        n_tasks_batch = self.splits_params[split_name]["n_tasks_batch"]
        n_up_steps = self.splits_params[split_name]["n_up_steps"]
        alpha_lr = self.splits_params[split_name]["alpha_lr"]

        for pg in self.adapt_opt.param_groups:
            pg['lr'] = alpha_lr

        loss_qry_all = []
        precision_qry_all = []
        loss_qry_avg_batch = 0.0

        # One pass over an instance of the meta-dataset
        for _ in range(n_tasks_total // n_tasks_batch):
            self.opt.zero_grad()

            # zero-grad the parameters
            for p in self.base_model.parameters():
                p.grad = torch.zeros_like(p.data)

            pbar = tqdm(range(n_tasks_batch))
            pbar.set_description("-----Batch_step %d "%(int(batch_step)//int(n_tasks_batch)))
            for j in pbar:
                spt_set = {k:v.to(self.device) for k, v in meta_tasks_batch[j].spt_features.items()}
                learner = copy.deepcopy(self.base_model)

                self.adapt_opt = torch.optim.Adam(learner.parameters(),
                                                  lr=alpha_lr,
                                                  betas=(0, 0.999))

                self.adapt_opt.load_state_dict(self.adapt_opt_state)

                for _ in range(n_up_steps):
                    self.adapt_opt.zero_grad()
                    loss_spt, q_encodings_spt, a_encodings_spt, n_encodings_spt = learner(**spt_set)

                    map_at_20_spt = mean_avg_prec_at_k_meta([meta_tasks_batch[j].spt.question_cluster], # question_list
                                                        q_encodings_spt,
                                                        [meta_tasks_batch[j].spt.all_candidates],
                                                        np.concatenate((a_encodings_spt, n_encodings_spt), axis=0),
                                                        k=20)

                    writer.add_scalar(split_name+"_loss_spt", loss_spt, ep * n_tasks_total + (batch_step+j))
                    writer.add_scalar(split_name+"_map_at_20_spt", map_at_20_spt, ep * n_tasks_total + (batch_step+j))

                    logger.info("loss_spt: {}, map_at_20_spt: {}".format(loss_spt, map_at_20_spt))

                    loss_spt.backward()
                    self.adapt_opt.step()

                qry_set = {k:v.to(self.device) for k, v in meta_tasks_batch[j].qry_features[0].items()}

                loss_qry, q_encodings_qry, a_encodings_qry, n_encodings_qry = learner(**qry_set)

                map_at_20_qry = mean_avg_prec_at_k_meta([meta_tasks_batch[j].qry[q_n].question_cluster for q_n in range(len(meta_tasks_batch[j].qry))], # question_list
                                                         q_encodings_qry,
                                                        [meta_tasks_batch[j].qry[q_n].all_candidates for q_n in range(len(meta_tasks_batch[j].qry))],
                                                        np.concatenate((a_encodings_qry, n_encodings_qry), axis=0),
                                                        k=20)
                logger.info("loss_qry: {}, map_at_20_qry: {}".format(loss_qry, map_at_20_qry))

                writer.add_scalar(split_name+"_loss_qry", loss_qry, ep * n_tasks_total + (batch_step+j))
                writer.add_scalar(split_name+"_map_at_20_qry", map_at_20_qry, ep * n_tasks_total + (batch_step+j))

                precision_qry_all.append(map_at_20_qry)

                if split_name == "train":
                    # On the query data => computing the gradients of the loss on the query
                    # and saving that for the outer loop

                    self.adapt_opt_state = self.adapt_opt.state_dict()
                    for p, l in zip(self.base_model.parameters(), learner.parameters()):
                        p.grad.data.add_(-1.0, l.data)

                    loss_qry.backward()
                    loss_qry_avg_batch += loss_qry

                loss_qry_all.append(loss_qry)

            if split_name != "test":
                for p in self.base_model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(1.0 / n_tasks_batch)
                self.opt.step()

        return loss_qry_avg_batch, loss_qry_all, precision_qry_all
