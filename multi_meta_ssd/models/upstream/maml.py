from stringprep import map_table_b2
import torch
import learn2learn as l2l
from multi_meta_ssd.models.upstream.meta_generic import MetaGeneric
from multi_meta_ssd.processors.downstream.utils_lareqa import mean_avg_prec_at_k_meta, convert_features
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
                 opt=None):

        super(MetaLearner, self).__init__(tokenizer,
                                          base_model,
                                          device,
                                          meta_learn_config)

        # Initializing with the training learning rates
        self.maml = l2l.algorithms.MAML(self.base_model,
                                        lr=5e-5,
                                        first_order=True)

        self.opt = opt

    def forward(self, split_name, meta_tasks_batches, ep, batch_step, writer): #
        n_tasks_total = self.meta_learn_config[split_name]["n_tasks_total"]
        n_tasks_batch = self.meta_learn_config[split_name]["n_tasks_batch"]
        n_up_steps = self.meta_learn_config[split_name]["n_up_steps"]
        alpha_lr = self.meta_learn_config[split_name]["alpha_lr"]
        self.maml.lr = alpha_lr

        # One pass over an instance of the meta-dataset
        loss_qry_all = []
        precision_qry_all = []
        loss_qry_batch = 0.0
        pbar = tqdm(range(n_tasks_batch))
        pbar.set_description("-----Batch_step %d "%(int(batch_step)//int(n_tasks_batch)))
        self.maml = self.maml.to(self.device)
        for j in pbar:
            self.opt.zero_grad()
            learner = self.maml.clone()

            loss_spt_avg = 0.0
            map_at_1_spt_avg = 0.0

            for s_n in range(len(meta_tasks_batches[j]["spt_features"])):
                spt_set = {k:v.to(self.device) for k, v in meta_tasks_batches[j]["spt_features"][s_n].items()}
                print("spt_shapes:", [v.shape for k,v in spt_set.items()])

                for _ in range(n_up_steps):
                    spt_outputs = learner(**spt_set)

                    loss_spt, q_encodings_spt, a_encodings_spt, n_encodings_spt = spt_outputs

                    loss_spt_avg += loss_spt

                    map_at_1_spt = mean_avg_prec_at_k_meta(meta_tasks_batches[j]["spt_questions"], # question_list
                                                           q_encodings_spt,
                                                           meta_tasks_batches[j]["spt_candidates"],
                                                           np.concatenate((a_encodings_spt, n_encodings_spt), axis=0),
                                                           k=1)

                    map_at_1_spt_avg += map_at_1_spt


                    logger.info("s_n: {}, loss_spt: {}, map_at_1_spt: {}".format(s_n, loss_spt, map_at_1_spt))

                    learner.adapt(loss_spt, allow_nograd=True, allow_unused=True)

                    del spt_outputs
                    del q_encodings_spt
                    del a_encodings_spt
                    del n_encodings_spt
                    torch.cuda.empty_cache()
                    gc.collect()

            writer.add_scalar(split_name+"_loss_spt", loss_spt/len(meta_tasks_batches[j]["spt_features"]), ep * n_tasks_total + (batch_step+j))
            writer.add_scalar(split_name+"_map_at_1_spt", map_at_1_spt_avg/len(meta_tasks_batches[j]["spt_features"]), ep * n_tasks_total + (batch_step+j))


            del spt_set
            torch.cuda.empty_cache()
            gc.collect()

            loss_qry_avg = 0.0
            map_at_1_qry_avg = 0.0

            print("MAX_QRY_FEATURES:", range(min(len(meta_tasks_batches[j]["qry_features"]), 4)))
            for q_n in tqdm(range(min(len(meta_tasks_batches[j]["qry_features"]), 4))):
                # print(meta_tasks_batches[j]["qry_features"][q_n])
                qry_set = {k:v.to(self.device) for k, v in meta_tasks_batches[j]["qry_features"][q_n].items()}

                # print("qry_shapes:", [v.shape for k,v in qry_set.items()])
                print("keys:", qry_set.keys())
                print("qry_shapes:", [v.shape for k,v in qry_set.items()])

                qry_outputs = learner(**qry_set)

                loss_qry, q_encodings_qry, a_encodings_qry, n_encodings_qry = qry_outputs
                loss_qry_avg += loss_qry

                map_at_1_qry = mean_avg_prec_at_k_meta(meta_tasks_batches[j]["qry_questions"][q_n], # question_list
                                                       q_encodings_qry,
                                                       meta_tasks_batches[j]["qry_candidates"][q_n],
                                                       np.concatenate((a_encodings_qry, n_encodings_qry), axis=0),
                                                       k=1)

                map_at_1_qry_avg += map_at_1_qry

                logger.info("q_n: {}, loss_qry: {}, map_at_1_qry: {}".format(q_n, loss_qry, map_at_1_qry))

                loss_qry_all.append(loss_qry.cpu().detach().numpy().item())
                precision_qry_all.append(map_at_1_qry)

                if split_name != "test":
                    # On the query data => computing the gradients of the loss on the query
                    # and saving that for the outer loop

                    loss_qry_batch += loss_qry

                del qry_set
                del qry_outputs
                del q_encodings_qry
                del a_encodings_qry
                del n_encodings_qry
                torch.cuda.empty_cache()
                gc.collect()

            writer.add_scalar(split_name+"_loss_qry", loss_qry_avg/len(meta_tasks_batches[j]["qry_features"]), ep * n_tasks_total + (batch_step+j))
            writer.add_scalar(split_name+"_map_at_1_qry", map_at_1_qry_avg/len(meta_tasks_batches[j]["qry_features"]), ep * n_tasks_total + (batch_step+j))

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
        print("precision_qry_all:", precision_qry_all)

        return loss_qry_avg_batch, loss_qry_all, precision_qry_all
