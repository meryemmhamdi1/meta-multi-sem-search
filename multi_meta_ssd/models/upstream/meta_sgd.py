import learn2learn as l2l

from multi_meta_ssd.models.upstream.maml import MetaLearner as MAML
from multi_meta_ssd.log import *


class MetaLearner(MAML):
    """
    Very similar to MAML with per parameter adaptive learning rate.
    forward() does not to be overridden.
    """

    def __init__(self,
                 base_model,
                 device,
                 meta_learn_config):

        super(MetaLearner, self).__init__(base_model,
                                          device,
                                          meta_learn_config)

        # we'll start with the same initial lr for all parameters.
        # in the future, if we want to use different initial parameters,
        # we need to pass "lrs" and create the corresponding torch Parameters.
        self.maml = l2l.algorithms.MetaSGD(self.base_model,
                                           lr=self.splits_params["train"]["alpha_lr"],
                                           first_order=True)
