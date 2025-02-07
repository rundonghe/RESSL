from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy
from LAMDA_SSL.Loss.Consistency import Consistency
from LAMDA_SSL.Loss.Semi_Supervised_Loss import Semi_Supervised_Loss
from LAMDA_SSL.utils import to_device
from LAMDA_SSL.utils import Bn_Controller
# from LAMDA_SSL.Loss.Consistency import Consistency
from LAMDA_SSL.utils import to_numpy
from LAMDA_SSL.Network.TransferNet import KMM

import copy
import numpy as np
import LAMDA_SSL.Config.DomainAdaption as config
from torch.utils.data.dataset import Dataset
import torch.nn as nn
from LAMDA_SSL.Network.TransferNet import TransferLoss
from LAMDA_SSL.Dataset.TrainDataset import TrainDataset
import torch
from LAMDA_SSL.Network.TransferNet import TransferNet
from LAMDA_SSL.utils import Bn_Controller
class Supervised_FixMatch(DeepModelMixin,InductiveEstimator,ClassifierMixin):
    def __init__(self,
                 lamda_so=1.0,
                 lamda_ta=1.0,
                 transfer_loss='weight',
                 T=1.0,
                 weight=False,
                 threshold=0.95,
                 T_teacher=1.0,
                 threshold_teacher=0.95,
                 lamda_u=config.lambda_u,
                 lambda_t=config.lambda_t,
                 warmup=config.warmup,
                 mu=config.mu,
                 ema_decay=config.ema_decay,
                 weight_decay=config.weight_decay,
                 epoch=config.epoch,
                 num_it_epoch=config.num_it_epoch,
                 num_it_total=config.num_it_total,
                 eval_epoch=config.eval_epoch,
                 eval_it=config.eval_it,
                 device=config.device,
                 train_dataset=config.train_dataset,
                 labeled_dataset=config.labeled_dataset,
                 unlabeled_dataset=config.unlabeled_dataset,
                 valid_dataset=config.valid_dataset,
                 test_dataset=config.test_dataset,
                 train_dataloader=config.train_dataloader,
                 labeled_dataloader=config.labeled_dataloader,
                 unlabeled_dataloader=config.unlabeled_dataloader,
                 valid_dataloader=config.valid_dataloader,
                 test_dataloader=config.test_dataloader,
                 train_sampler=config.train_sampler,
                 train_batch_sampler=config.train_batch_sampler,
                 valid_sampler=config.valid_sampler,
                 valid_batch_sampler=config.valid_batch_sampler,
                 test_sampler=config.test_sampler,
                 test_batch_sampler=config.test_batch_sampler,
                 labeled_sampler=config.labeled_sampler,
                 unlabeled_sampler=config.unlabeled_sampler,
                 labeled_batch_sampler=config.labeled_batch_sampler,
                 unlabeled_batch_sampler=config.unlabeled_batch_sampler,
                 augmentation=config.augmentation,
                 network=config.network,
                 optimizer=config.optimizer,
                 scheduler=config.scheduler,
                 evaluation=config.evaluation,
                 parallel=config.parallel,
                 file=config.file,
                 verbose=config.verbose
                 ):
        # >> Parameter:
        # >> - lambda_u: The weight of unsupervised loss.
        # >> - warmup: The end position of warmup. For example, num_it_total is 100 and warmup is 0.4,
        #              then warmup is performed in the first 40 iterations.
        DeepModelMixin.__init__(self,train_dataset=train_dataset,
                                    valid_dataset=valid_dataset,
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
                                    labeled_dataset=labeled_dataset,
                                    unlabeled_dataset=unlabeled_dataset,
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
        self.ema_decay=ema_decay
        self.lamda_u=lamda_u
        self.lambda_t=lambda_t
        self.lamda_so=lamda_so
        self.lamda_ta=lamda_ta
        self.T=T
        self.threshold=threshold
        self.weight=weight
        self.transfer_loss=transfer_loss
        self.weight_decay=weight_decay
        self.warmup=warmup
        self.T_teacher=T_teacher
        self.threshold_teacher=threshold_teacher
        self.bn_controller=Bn_Controller()
        self._estimator_type = ClassifierMixin._estimator_type

    def init_transform(self):
        self._train_dataset.add_transform(copy.copy(self.train_dataset.transform),dim=0,x=1)
        self._train_dataset.add_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_transform(self.strong_augmentation,dim=1,x=1,y=0)

    def fit(self,X=None,y=None,unlabeled_X=None,valid_X=None,valid_y=None,unlabeled_y=None,logits=None):
        self.labeled_logits=torch.Tensor(logits).to(self.device)
        self.unlabeled_X=unlabeled_X
        self.unlabeled_y=unlabeled_y
        self.init_train_dataset(X,y,unlabeled_X,unlabeled_y)
        self.init_train_dataloader()
        if self.network is not None:
            self.init_model()
            self.init_ema()
            self.init_optimizer()
            self.init_scheduler()
        self.init_augmentation()
        self.init_transform()
        self.start_fit()
        self.fit_epoch_loop(valid_X,valid_y)
        self.end_fit()
        return self

    def init_train_dataset(self,X=None,y=None,unlabeled_X=None,unlabeled_y=None, *args, **kwargs):
        self._train_dataset=copy.deepcopy(self.train_dataset)
        if isinstance(X,TrainDataset):
            self._train_dataset=X
        elif isinstance(X,Dataset) and y is None:
            self._train_dataset.init_dataset(labeled_dataset=X, unlabeled_dataset=unlabeled_X,unlabeled_y=unlabeled_y)
        else:
            self._train_dataset.init_dataset(labeled_X=X, labeled_y=y,unlabeled_X=unlabeled_X,unlabeled_y=unlabeled_y)

    def fit_batch_loop(self,valid_X=None,valid_y=None):
        for (lb_idx, lb_X, lb_y), (ulb_idx, ulb_X, ulb_y) in zip(self._labeled_dataloader, self._unlabeled_dataloader):
            if self.it_epoch >= self.num_it_epoch or self.it_total >= self.num_it_total:
                break
            self.start_fit_batch()
            lb_idx = to_device(lb_idx,self.device)
            lb_X = to_device(lb_X,self.device)
            lb_y = to_device(lb_y,self.device)
            ulb_idx = to_device(ulb_idx,self.device)
            ulb_X  = to_device(ulb_X,self.device)
            ulb_y=to_device(ulb_y,self.device)
            train_result = self.train(lb_X=lb_X, lb_y=lb_y, ulb_X=ulb_X, ulb_y=ulb_y, lb_idx=lb_idx, ulb_idx=ulb_idx)
            self.end_fit_batch(train_result)
            self.it_total += 1
            self.it_epoch += 1
            if self.verbose:
                print(self.it_total,file=self.file)
                print(self.it_total)
            if valid_X is not None and self.eval_it is not None and self.it_total % self.eval_it == 0:
                self.evaluate(X=self.unlabeled_X,y=self.unlabeled_y,valid=True)
                self.evaluate(X=valid_X, y=valid_y,valid=True)
                self.valid_performance.update({"epoch_" + str(self._epoch) + "_it_" + str(self.it_epoch): self.performance})

    def train(self,lb_X,lb_y,ulb_X,ulb_y=None,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X ,s_lb_X= lb_X[0] ,lb_X[1]
        lb_y = lb_y[0] if isinstance(lb_y, (tuple, list)) else lb_y
        ulb_X = ulb_X[0] if isinstance(ulb_X, (tuple, list)) else ulb_X
        ulb_y = ulb_y[0] if isinstance(ulb_y, (tuple, list)) else ulb_y
        batch= ulb_X.shape[0]
        target_features,target_logits=self._network(ulb_X)
        source_features ,source_logits= self._network(lb_X,source=False)
        self.bn_controller.freeze_bn(self._network)
        source_aug_features ,source_aug_logits= self._network(s_lb_X,source=False)
        self.bn_controller.unfreeze_bn(self._network)
        # features,target_logits=self._network(ulb_X)

        #
        # pseudo_label = torch.softmax(source_logits.detach() / self.T, dim=-1)
        # max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        # mask = max_probs.ge(self.threshold).float()
        #
        source_logits_teacher=self.labeled_logits[lb_idx]
        # #
        pseudo_label_teacher = torch.softmax(source_logits_teacher.detach() / self.T, dim=-1)
        max_probs_teacher, targets_u_teacher = torch.max(pseudo_label_teacher, dim=-1)
        mask_teacher = max_probs_teacher.ge(self.threshold).float()
        # #
        # print(mask_teacher.sum())
        # #
        # # # print(mask.sum())
        target_clf_loss = Cross_Entropy(reduction='mean')(target_logits, ulb_y)
        if self.weight:
            weight = torch.Tensor(KMM(kernel_type='linear').fit(to_numpy(source_features), to_numpy(target_features))).to(self.device).detach()
            weight = weight / (weight.sum()) * weight.shape[0]
            # source_clf_loss = (Cross_Entropy(reduction='none')(source_logits, lb_y.long())* mask_teacher*weight).mean()
            source_clf_loss=(Consistency(reduction='none')(source_logits_teacher.detach(),source_logits)*weight).mean()
            # unsup_loss=0
            # source_clf_loss = 0
            unsup_loss = (Cross_Entropy(reduction='none')(source_aug_logits, targets_u_teacher) * mask_teacher * weight).mean()
        else:
            # source_clf_loss = (Cross_Entropy(reduction='none')(source_logits, lb_y.long())* mask_teacher).mean()
            source_clf_loss = (Consistency(reduction='none')(source_logits_teacher.detach(), source_logits) ).mean()
            # unsup_loss = 0
            # source_clf_loss=0
            unsup_loss = (Cross_Entropy(reduction='none')(source_aug_logits, targets_u_teacher) * mask_teacher).mean()
        # transfer_loss_args = {
        #     "loss_type": self.transfer_loss,
        #     "max_iter": self.it_epoch,
        #     "num_classes": 31,
        #     "device": self.device
        # }
        # self.adapt_loss = TransferLoss(**transfer_loss_args)
        # kwargs = {}
        # if self.transfer_loss == "lmmd":
        #     kwargs['source_label'] = lb_X
        #     kwargs['target_logits'] = torch.nn.functional.softmax(target_logits, dim=1)
        # elif self.transfer_loss == "daan":
        #     kwargs['source_logits'] = torch.nn.functional.softmax(source_logits, dim=1)
        #     kwargs['target_logits'] = torch.nn.functional.softmax(target_logits, dim=1)
        # elif self.transfer_loss == 'bnm':
        #     target_features = nn.Softmax(dim=1)(target_logits)
        # transfer_loss = self.adapt_loss(source_features, target_features, **kwargs)
        # return target_clf_loss
        return target_clf_loss,source_clf_loss,unsup_loss

    def get_loss(self,train_result,*args,**kwargs):
        target_clf_loss,source_clf_loss, unsup_loss=train_result
        _warmup = float(np.clip((self.it_total) / (self.warmup * self.num_it_total), 0., 1.))
        self.lamda_u=self.lamda_u*_warmup
        self.lamda_so=self.lamda_so*_warmup
        loss = self.lamda_ta*target_clf_loss+self.lamda_u*unsup_loss+self.lamda_so*source_clf_loss
        # target_clf_loss = train_result
        # target_clf_loss=train_result
        # loss=target_clf_loss
        return loss

    @torch.no_grad()
    def estimate(self, X, idx=None, *args, **kwargs):
        _,outputs = self._network(X)
        return outputs

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)



