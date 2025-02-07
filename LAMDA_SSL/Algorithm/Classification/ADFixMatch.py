import copy
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from sklearn.base import ClassifierMixin
import LAMDA_SSL.Config.ADFixMatch as config
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy
from LAMDA_SSL.Loss.Semi_Supervised_Loss import Semi_Supervised_Loss
from LAMDA_SSL.Base.BaseOptimizer import BaseOptimizer
from LAMDA_SSL.Base.BaseScheduler import BaseScheduler
import torch
import math
import torch.nn as nn
from easydl import *
import torch.optim as optim
class ADFixMatch(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
                 threshold=config.threshold,
                 lambda_u=config.lambda_u,
                 adv_warmup=config.adv_warmup,
                 T=config.T,
                 mu=config.mu,
                 weight_decay=config.weight_decay,
                 ema_decay=config.ema_decay,
                 epoch=config.epoch,
                 num_it_epoch=config.num_it_epoch,
                 num_it_total=config.num_it_total,
                 eval_epoch=config.eval_epoch,
                 eval_it=config.eval_it,
                 optimizer=config.optimizer,
                 scheduler=config.scheduler,
                 discriminator=config.discriminator,
                 discriminator_optimizer=config.discriminator_optimizer,
                 discriminator_scheduler=config.discriminator_scheduler,
                 device=config.device,
                 train_dataset=config.train_dataset,
                 labeled_dataset=config.labeled_dataset,
                 unlabeled_dataset=config.unlabeled_dataset,
                 valid_dataset=config.valid_dataset,
                 test_dataset=config.test_dataset,
                 train_dataloader=config.train_dataloader,
                 valid_dataloader=config.valid_dataloader,
                 test_dataloader=config.test_dataloader,
                 augmentation=config.augmentation,
                 network=config.network,
                 train_sampler=config.train_sampler,
                 train_batch_sampler=config.train_batch_sampler,
                 valid_sampler=config.valid_sampler,
                 valid_batch_sampler=config.valid_batch_sampler,
                 test_sampler=config.test_sampler,
                 test_batch_sampler=config.test_batch_sampler,
                 labeled_dataloader=config.labeled_dataloader,
                 unlabeled_dataloader=config.unlabeled_dataloader,
                 labeled_sampler=config.labeled_sampler,
                 unlabeled_sampler=config.unlabeled_sampler,
                 labeled_batch_sampler=config.labeled_batch_sampler,
                 unlabeled_batch_sampler=config.unlabeled_batch_sampler,
                 parallel=config.parallel,
                 evaluation=config.evaluation,
                 file=config.file,
                 verbose=config.verbose
                 ):

        # >> Parameter:
        # >> - threshold: The confidence threshold for choosing samples.
        # >> - lambda_u: The weight of unsupervised loss.
        # >> - T: Sharpening temperature.
        # >> - num_classes: The number of classes for the classification task.
        # >> - thresh_warmup: Whether to use threshold warm-up mechanism.
        # >> - use_hard_labels: Whether to use hard labels in the consistency regularization.
        # >> - use_DA: Whether to perform distribution alignment for soft labels.
        # >> - p_target: p(y) based on the labeled examples seen during training.

        DeepModelMixin.__init__(self,train_dataset=train_dataset,
                                    valid_dataset=valid_dataset,
                                    labeled_dataset=labeled_dataset,
                                    unlabeled_dataset=unlabeled_dataset,
                                    test_dataset=test_dataset,
                                    train_dataloader=train_dataloader,
                                    valid_dataloader=valid_dataloader,
                                    test_dataloader=test_dataloader,
                                    augmentation=augmentation,
                                    network=network,
                                    train_sampler=train_sampler,
                                    train_batch_sampler=train_batch_sampler,
                                    valid_sampler=valid_sampler,
                                    valid_batch_sampler=valid_batch_sampler,
                                    test_sampler=test_sampler,
                                    test_batch_sampler=test_batch_sampler,
                                    labeled_dataloader=labeled_dataloader,
                                    unlabeled_dataloader=unlabeled_dataloader,
                                    labeled_sampler=labeled_sampler,
                                    unlabeled_sampler=unlabeled_sampler,
                                    labeled_batch_sampler=labeled_batch_sampler,
                                    unlabeled_batch_sampler=unlabeled_batch_sampler,
                                    epoch=epoch,
                                    num_it_epoch=num_it_epoch,
                                    num_it_total=num_it_total,
                                    eval_epoch=eval_epoch,
                                    eval_it=eval_it,
                                    mu=mu,
                                    weight_decay=weight_decay,
                                    ema_decay=ema_decay,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    device=device,
                                    evaluation=evaluation,
                                    parallel=parallel,
                                    file=file,
                                    verbose=verbose
                                    )
        self.lambda_u=lambda_u
        self.threshold=threshold
        self.discriminator = discriminator
        self.discriminator_optimizer=discriminator_optimizer
        self.discriminator_scheduler=discriminator_scheduler
        self.adv_warmup = adv_warmup
        self.T=T
        self.weight_decay=weight_decay
        self._estimator_type=ClassifierMixin._estimator_type

    def init_transform(self):
        self._train_dataset.add_unlabeled_transform(copy.copy(self.train_dataset.unlabeled_transform),dim=0,x=1)
        self._train_dataset.add_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.strong_augmentation,dim=1,x=1,y=0)

    def init_model(self):
        self._network = copy.deepcopy(self.network)
        self._parallel = copy.deepcopy(self.parallel)
        self._discriminator=copy.deepcopy(self.discriminator)
        if self.device is None:
            self.device='cpu'
        if self.device is not 'cpu':
            torch.cuda.set_device(self.device)
        self._discriminator=self._discriminator.to(self.device)
        self._network=self._network.to(self.device)
        if self._parallel is not None:
            self._network=self._parallel.init_parallel(self._network)
            self._discriminator=self._parallel.init_parallel(self._discriminator)

    def init_optimizer(self):
        self._optimizer=copy.deepcopy(self.optimizer)
        self._discriminator_optimizer = copy.deepcopy(self.discriminator_optimizer)

        if isinstance(self._optimizer,BaseOptimizer):
            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in self._network.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self._network.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0}
            ]
            self._optimizer=self._optimizer.init_optimizer(params=grouped_parameters)
        # self._optimizer = OptimWithSheduler(self._optimizer, self.scheduler)
        if isinstance(self._discriminator_optimizer,BaseOptimizer):
            no_decay = ['bias', 'bn']
            grouped_parameters = [
                {'params': [p for n, p in self._discriminator.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self._discriminator.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0}
            ]
            self._discriminator_optimizer=self._discriminator_optimizer.init_optimizer(params=grouped_parameters)

    def start_fit(self, *args, **kwargs):
        self.init_epoch()
        self._network.zero_grad()
        self._network.train()
        self._discriminator.zero_grad()
        self._discriminator.train()

    def init_scheduler(self):
        # pass
        self._scheduler=copy.deepcopy(self.scheduler)
        if isinstance(self._scheduler,BaseScheduler):
            self._scheduler=self._scheduler.init_scheduler(optimizer=self._optimizer)
        #
        self._discriminator_scheduler=copy.deepcopy(self.discriminator_scheduler)
        if isinstance(self._discriminator_scheduler,BaseScheduler):
            self._discriminator_scheduler=self._discriminator_scheduler.init_scheduler(optimizer=self._discriminator_optimizer)

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X=lb_X[0] if isinstance(lb_X,(tuple,list)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(tuple,list)) else lb_y
        w_ulb_X,s_ulb_X=ulb_X[0],ulb_X[1]
        inputs=torch.cat((lb_X, w_ulb_X, s_ulb_X))
        batch_size=lb_X.shape[0]
        features,logits = self._network(inputs)
        lb_logits = logits[:batch_size]
        w_ulb_logits, s_ulb_logits = logits[batch_size:].chunk(2)
        l_feature=features[:batch_size]
        u_feature,_=features[batch_size:].chunk(2)
        l_domain_prob = self._discriminator.forward(l_feature)
        u_domain_prob = self._discriminator.forward(u_feature)

        # inputs=torch.cat((lb_X, w_ulb_X, s_ulb_X))
        # logits = self._network(inputs)
        # lb_logits = logits[:batch_size]
        # w_ulb_logits, s_ulb_logits = logits[batch_size:].chunk(2)
        train_result=(lb_logits,lb_y,w_ulb_logits, s_ulb_logits,l_domain_prob,u_domain_prob)
        return train_result

    def get_loss(self,train_result,*args,**kwargs):
        lb_logits, lb_y, w_ulb_logits, s_ulb_logits,l_domain_prob,u_domain_prob = train_result
        sup_loss=Cross_Entropy(reduction='mean')(logits=lb_logits,targets=lb_y)
        pseudo_label = torch.softmax(w_ulb_logits.detach() / self.T, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()
        unsup_loss = (Cross_Entropy(reduction='none')(s_ulb_logits, targets_u) * mask).mean()
        adv_coef = 1. * math.exp(-5 * (1 - min(self.it_total /self.adv_warmup , 1)) ** 2)
        adv_loss = torch.zeros(1).to(self.device)
        tmp = nn.BCELoss(reduction="none")(l_domain_prob, torch.zeros_like(l_domain_prob))
        adv_loss += torch.mean(tmp, dim=0)
        tmp = nn.BCELoss(reduction="none")(u_domain_prob, torch.ones_like(u_domain_prob))
        adv_loss += torch.mean(tmp, dim=0)
        loss = Semi_Supervised_Loss(lambda_u=self.lambda_u)(sup_loss, unsup_loss)+adv_coef*adv_loss
        return loss

    @torch.no_grad()
    def estimate(self, X, idx=None, *args, **kwargs):
        _,outputs = self._network(X)
        return outputs

    def optimize(self,loss,*args,**kwargs):

        # self._optimizer.zero_grad()
        # self._discriminator_optimizer.zero_grad()
        self._network.zero_grad()
        self._discriminator.zero_grad()
        # with OptimizerManager([self._optimizer]):
        #     loss.backward()
        loss.backward()
        self._optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()
        self._discriminator_optimizer.step()
        if self._discriminator_scheduler is not None:
            self._discriminator_scheduler.step()
        if self.ema is not None:
            self.ema.update()

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)