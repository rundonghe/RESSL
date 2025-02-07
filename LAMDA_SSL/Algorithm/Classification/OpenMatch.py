from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from sklearn.base import ClassifierMixin
import LAMDA_SSL.Config.FixMatch as config
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy
from LAMDA_SSL.Loss.Semi_Supervised_Loss import Semi_Supervised_Loss
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
import torch
import copy
from LAMDA_SSL.utils import class_status

def ova_loss(logits_open, label):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.range(0, logits_open.size(0) - 1).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :] + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :] + 1e-8) * label_sp_neg, 1)[0])
    Lo = open_loss_neg + open_loss
    return Lo

   
def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open *
                                   torch.log(logits_open + 1e-8), 1), 1))
    return Le

class OpenMatch(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
                 T=1,
                 start_fix=10/512,
                 lambda_oem=0.1,
                 lambda_socr=0.5,
                 num_classes=None,
                 threshold=0.95,
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
        self.T=T
        self.open_augmentation=None
        self.weight_decay=weight_decay
        self.num_classes=num_classes
        self.lambda_oem=lambda_oem
        self.lambda_socr=lambda_socr
        self.threshold=threshold
        self.start_fix=start_fix
        self._estimator_type=ClassifierMixin._estimator_type

    def start_fit(self):
        self.num_classes = self.num_classes if self.num_classes is not None else \
            class_status(self._train_dataset.labeled_dataset.y).num_classes
        self._network.zero_grad()
        self._network.train()

    def init_augmentation(self):
        self._augmentation = copy.deepcopy(self.augmentation)
        if self._augmentation is not None:
            if isinstance(self._augmentation, dict):
                self.weak_augmentation = self._augmentation['augmentation'] \
                    if 'augmentation' in self._augmentation.keys() \
                    else self._augmentation['weak_augmentation']
                if 'strong_augmentation' in self._augmentation.keys():
                    self.strong_augmentation = self._augmentation['strong_augmentation']
            elif isinstance(self._augmentation, (list, tuple)):
                self.weak_augmentation = self._augmentation[0]
                if len(self._augmentation) > 1:
                    self.strong_augmentation = self._augmentation[1]
            else:
                self.weak_augmentation = copy.copy(self._augmentation)
            if self.strong_augmentation is None:
                self.strong_augmentation = copy.copy(self.weak_augmentation)

            if 'open_augmentation' in self._augmentation.keys():
                self.open_augmentation = self._augmentation['open_augmentation']
            else:
                self.open_augmentation = copy.copy(self.weak_augmentation)

    def init_transform(self):
        self._train_dataset.add_transform(copy.copy(self.train_dataset.transform),dim=0,x=1)
        self._train_dataset.add_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_transform(self.open_augmentation,dim=1,x=1,y=0)
        self._train_dataset.add_unlabeled_transform(copy.copy(self.train_dataset.unlabeled_transform),dim=0,x=1)
        self._train_dataset.add_unlabeled_transform(copy.copy(self.train_dataset.unlabeled_transform),dim=0,x=2)
        self._train_dataset.add_unlabeled_transform(copy.copy(self.train_dataset.unlabeled_transform),dim=0,x=3)
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation,dim=1,x=1,y=0)
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation,dim=1,x=2,y=0)
        self._train_dataset.add_unlabeled_transform(self.strong_augmentation,dim=1,x=3,y=0)

    def train(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X_w,lb_X_op=lb_X[0],lb_X[1]
        lb_y=lb_y[0] if isinstance(lb_y,(tuple,list)) else lb_y
        inputs_u,inputs_u1,inputs_u2,inputs_u3=ulb_X[0],ulb_X[1],ulb_X[2],ulb_X[3]
        batch_size = lb_X_w.shape[0]
        inputs_all = torch.cat([inputs_u, inputs_u1], 0)
        inputs = torch.cat([lb_X_op, lb_X_w,
                            inputs_all], 0).to(self.device)
        targets_x = torch.zeros(batch_size, self.num_classes).to(self.device).scatter_(1, lb_y.view(-1,1), 1)
        ## Feed data
        logits, logits_open = self._network(inputs)
        logits_open_u1, logits_open_u2 = logits_open[2*batch_size:].chunk(2)

        ## Loss for labeled samples
        # print(targets_x.shape)
        # print(logits[:2*batch_size].shape)
        Lx = F.cross_entropy(logits[:2*batch_size], torch.cat([targets_x,targets_x], dim=0), reduction='mean')
        Lo = ova_loss(logits_open[:2*batch_size], torch.cat([lb_y,lb_y], dim=0))

        ## Open-set entropy minimization
        L_oem = ova_ent(logits_open_u1) / 2.
        L_oem += ova_ent(logits_open_u2) / 2.

        ## Soft consistenty regularization
        logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
        logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
        logits_open_u1 = F.softmax(logits_open_u1, 1)
        logits_open_u2 = F.softmax(logits_open_u2, 1)
        L_socr = torch.mean(torch.sum(torch.sum(torch.abs(logits_open_u1 - logits_open_u2)**2, 1), 1))
        print(self.it_total)
        print(self.start_fix*self.num_it_total)
        if self.it_total >= self.start_fix*self.num_it_total:
            inputs_ws = torch.cat([inputs_u2, inputs_u3], 0).to(self.device)
            logits, logits_open_fix = self._network(inputs_ws)
            logits_u_w, logits_u_s = logits.chunk(2)
            pseudo_label = torch.softmax(logits_u_w.detach()/self.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.threshold).float()
            L_fix = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
        else:
            L_fix = torch.zeros(1).to(self.device).mean()
        loss = Lx + Lo + self.lambda_oem * L_oem  \
               + self.lambda_socr * L_socr + L_fix
        return loss



    def get_loss(self,train_result,*args,**kwargs):
        loss= train_result

        return loss

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)