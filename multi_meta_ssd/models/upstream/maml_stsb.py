from stringprep import map_table_b2
import torch
import learn2learn as l2l
# import logging as logger
from multi_meta_ssd.models.upstream.meta_generic import MetaGeneric
from multi_meta_ssd.processors.downstream.utils_lareqa import mean_avg_prec_at_k_meta, convert_features, compute_sem_sim
from multi_meta_ssd.log import *
import numpy as np
from tqdm import tqdm
import gc

class MetaLearner(MetaGeneric):
    def __init__(self,
                 tokenizer,
                 base_model,
                 device,
                 meta_learn_config,
                 opt=None,
                 task_name="lareqa"):

        super(MetaLearner, self).__init__(tokenizer,
                                          base_model,
                                          device,
                                          meta_learn_config)

        # Initializing with the training learning rates
        self.maml = l2l.algorithms.MAML(self.base_model,
                                        lr=5e-5,
                                        first_order=True)

        self.opt = opt
        self.task_name=task_name
        if self.task_name == "lareqa":
            self.scores_name = "map_at_1"
        else:
            self.scores_name = "pearson_corr"

    def forward(self, split_name, meta_tasks_batches, ep, batch_step, writer): #
        n_tasks_total = self.meta_learn_config[split_name]["n_tasks_total"]
        n_tasks_batch = self.meta_learn_config[split_name]["n_tasks_batch"]
        n_up_steps = self.meta_learn_config[split_name]["n_up_steps"]
        alpha_lr = self.meta_learn_config[split_name]["alpha_lr"]
        self.maml.lr = alpha_lr

        # One pass over an instance of the meta-dataset
        loss_qry_all = []
        scores_qry_all = []
        loss_qry_batch = 0.0
        pbar = tqdm(range(n_tasks_batch))
        pbar.set_description("-----Batch_step %d "%(int(batch_step)//int(n_tasks_batch)))
        self.maml = self.maml.to(self.device)
        for j in pbar:
            self.opt.zero_grad()
            learner = self.maml.clone()

            loss_spt_avg = 0.0
            scores_spt_avg = 0.0

            
            spt_set = {k:v.to(self.device) for k, v in meta_tasks_batches[j].spt.inputs.items()}
        
            for _ in range(n_up_steps):
                spt_outputs = learner(**spt_set)

                loss_spt, q_encodings_spt, a_encodings_spt = spt_outputs
                scores = compute_sem_sim(q_encodings_spt, a_encodings_spt, meta_tasks_batches[j].spt.scores)
                    

                logger.info("loss_spt: {}, {}: {}".format(self.scores_name+"_spt", loss_spt, scores))
            
                loss_spt_avg += loss_spt
                scores_spt_avg += scores

                learner.adapt(loss_spt, allow_nograd=True, allow_unused=True)

                del spt_outputs
                del q_encodings_spt
                del a_encodings_spt
                
                torch.cuda.empty_cache()
                gc.collect()

            writer.add_scalar(split_name+"_loss_spt", loss_spt_avg/n_up_steps, ep * n_tasks_total + (batch_step+j))
            writer.add_scalar(split_name+"_"+self.scores_name+"_spt", scores_spt_avg/n_up_steps, ep * n_tasks_total + (batch_step+j))


            del spt_set
            torch.cuda.empty_cache()
            gc.collect()

            qry_set = {k:v.to(self.device) for k, v in meta_tasks_batches[j].qry.inputs.items()}
            qry_outputs = learner(**qry_set)

            loss_qry, q_encodings_qry, a_encodings_qry = qry_outputs
            scores = compute_sem_sim(q_encodings_qry, a_encodings_qry, meta_tasks_batches[j].qry.scores)

            logger.info("loss_qry: {}, {}: {}".format(self.scores_name+"_qry", loss_qry, scores))

            loss_qry_all.append(loss_qry.cpu().detach().numpy().item())
            scores_qry_all.append(scores)

            if split_name != "test":
                # On the query data => computing the gradients of the loss on the query
                # and saving that for the outer loop

                loss_qry_batch += loss_qry

            del qry_set
            del qry_outputs
            del q_encodings_qry
            del a_encodings_qry
            torch.cuda.empty_cache()
            gc.collect()

            writer.add_scalar(split_name+"_loss_qry", loss_qry, ep * n_tasks_total + (batch_step+j))
            writer.add_scalar(split_name+"_"+self.scores_name+"_qry", scores, ep * n_tasks_total + (batch_step+j))

        if split_name != "test":
            loss_qry_avg_batch = loss_qry_batch / n_tasks_batch
            for p in self.maml.parameters():
                if p.grad is not None:
                    p.grad.mul_(1.0 / n_tasks_batch)
            loss_qry_avg_batch.backward()
            self.opt.step()

        del learner
        self.base_model = self.base_model.cpu()
        torch.cuda.empty_cache()
        gc.collect()
        self.maml = self.maml.cpu()
        if split_name != "test":
            loss_qry_avg_batch = loss_qry_avg_batch.cpu().detach().numpy()
            print("loss_qry_avg_batch:", loss_qry_avg_batch)
        else:
            loss_qry_avg_batch = 0
        print("loss_qry_all:", loss_qry_all)
        print("scores_qry_all:", scores_qry_all)

        return loss_qry_avg_batch, loss_qry_all, scores_qry_all
