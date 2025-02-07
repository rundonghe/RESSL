import copy
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from LAMDA_SSL.Base.DeepModelMixin import DeepModelMixin
from sklearn.base import ClassifierMixin
import LAMDA_SSL.Config.FixMatch as config
from LAMDA_SSL.Loss.Cross_Entropy import Cross_Entropy
from LAMDA_SSL.Loss.Semi_Supervised_Loss import Semi_Supervised_Loss
from LAMDA_SSL.utils import EMA
from LAMDA_SSL.Base.BaseOptimizer import BaseOptimizer
from LAMDA_SSL.Base.BaseScheduler import BaseScheduler
import torch
from LAMDA_SSL.utils import to_device
from LAMDA_SSL.utils import to_numpy
import cvxpy as cp
from LAMDA_SSL.Network.TransferNet import KMM
class Two_Time_Adaption(InductiveEstimator,DeepModelMixin,ClassifierMixin):
    def __init__(self,
                 threshold=config.threshold,
                 lambda_u=config.lambda_u,
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
        self.T=T
        self.weight_decay=weight_decay
        self._estimator_type=ClassifierMixin._estimator_type

    def init_model(self):
        self._network = copy.deepcopy(self.network)
        self._parallel = copy.deepcopy(self.parallel)
        if isinstance(self._network,(tuple,list)):
            self.labeled_network=self._network[0]
            self.unlabeled_network=self._network[1]
        elif isinstance(self._network,dict):
            self.labeled_network=self._network['labeled']
            self.unlabeled_network=self._network['unlabeled']
        else:
            self.labeled_network=self._network
            self.unlabeled_network=copy.deepcopy(self.labeled_network)
        if self.device is None:
            self.device='cpu'
        if self.device is not 'cpu':
            torch.cuda.set_device(self.device)
        self.labeled_network = self.labeled_network.to(self.device)
        self.unlabeled_network = self.unlabeled_network.to(self.device)
        if self._parallel is not None:
            self.labeled_network=self._parallel.init_parallel(self.labeled_network)
            self.unlabeled_network = self._parallel.init_parallel(self.unlabeled_network)

    def init_ema(self):
        if self.ema_decay is not None:
            self.labeled_ema=EMA(model=self.labeled_network,decay=self.ema_decay)
            self.labeled_ema.register()
            self.unlabeled_ema = EMA(model=self.unlabeled_network, decay=self.ema_decay)
            self.unlabeled_ema.register()
        else:
            self.labeled_ema=None
            self.unlabeled_ema=None

    def init_optimizer(self):
        self._optimizer = copy.deepcopy(self.optimizer)
        if isinstance(self._optimizer,(tuple,list)):
            self.labeled_optimizer=self._optimizer[0]
            self.unlabeled_optimizer=self._optimizer[1]
        elif isinstance(self._optimizer,dict):
            self.labeled_optimizer=self._optimizer['labeled']
            self.unlabeled_optimizer=self._optimizer['unlabeled']
        else:
            self.labeled_optimizer=self._optimizer
            self.unlabeled_optimizer=copy.deepcopy(self.labeled_optimizer)
        if isinstance(self._optimizer,BaseOptimizer):
            no_decay = ['bias', 'bn']
            labeled_grouped_parameters = [
                {'params': [p for n, p in self.labeled_network.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self.labeled_network.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self.labeled_optimizer=self.labeled_optimizer.init_optimizer(params=labeled_grouped_parameters)
            unlabeled_grouped_parameters = [
                {'params': [p for n, p in self.unlabeled_network.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self.unlabeled_network.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self.unlabeled_optimizer=self.unlabeled_optimizer.init_optimizer(params=unlabeled_grouped_parameters)

    def init_scheduler(self):
        self._scheduler = copy.deepcopy(self.scheduler)
        if isinstance(self._scheduler,(tuple,list)):
            self.labeled_scheduler=self._scheduler[0]
            self.unlabeled_scheduler=self._scheduler[1]
        elif isinstance(self._scheduler,dict):
            self.labeled_scheduler=self._scheduler['labeled']
            self.unlabeled_scheduler=self._scheduler['unlabeled']
        else:
            self.labeled_scheduler=self._scheduler
            self.unlabeled_scheduler=copy.deepcopy(self.labeled_scheduler)

        self.labeled_scheduler=copy.deepcopy(self.scheduler)
        if isinstance(self.labeled_scheduler,BaseScheduler):
            self.labeled_scheduler=self.labeled_scheduler.init_scheduler(optimizer=self.labeled_optimizer)

        self.unlabeled_scheduler=copy.deepcopy(self.scheduler)
        if isinstance(self.unlabeled_scheduler,BaseScheduler):
            self.unlabeled_scheduler=self.unlabeled_scheduler.init_scheduler(optimizer=self.unlabeled_optimizer)

    def init_transform(self):
        self._train_dataset.add_unlabeled_transform(copy.copy(self.train_dataset.unlabeled_transform),dim=0,x=1)
        self._train_dataset.add_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.weak_augmentation,dim=1,x=0,y=0)
        self._train_dataset.add_unlabeled_transform(self.strong_augmentation,dim=1,x=1,y=0)

    def start_fit(self, *args, **kwargs):
        self.init_epoch()
        self.unlabeled_network.zero_grad()
        self.unlabeled_network.train()
        self.labeled_network.zero_grad()
        self.labeled_network.train()
        self.labeled_mean=0
        self.labeled_num=0
        self.unlabeled_mean=0
        self.unlabeled_num=0

    def optimize_unlabeled(self,loss,*args,**kwargs):
        self.unlabeled_network.zero_grad()
        loss.backward()
        self.unlabeled_optimizer.step()
        if self.unlabeled_scheduler is not None:
            self.unlabeled_scheduler.step()
        if self.unlabeled_ema is not None:
            self.unlabeled_ema.update()

    def optimize_labeled(self,loss,*args,**kwargs):
        self.labeled_network.zero_grad()
        loss.backward()
        self.labeled_optimizer.step()
        if self.labeled_scheduler is not None:
            self.labeled_scheduler.step()
        if self.labeled_ema is not None:
            self.labeled_ema.update()

    def end_fit_batch_unlabeled(self, train_result,*args, **kwargs):
        self.loss = self.get_loss_unlabeled(train_result)
        self.optimize_unlabeled(self.loss)

    def end_fit_batch_labeled(self, train_result,*args, **kwargs):
        self.loss = self.get_loss_labeled(train_result)
        self.optimize_labeled(self.loss)

    def fit_batch_loop(self,valid_X=None,valid_y=None):
        for (lb_idx, lb_X, lb_y), (ulb_idx, ulb_X, _) in zip(self._labeled_dataloader, self._unlabeled_dataloader):
            if self.it_epoch >= self.num_it_epoch or self.it_total >= self.num_it_total:
                break
            self.start_fit_batch()
            lb_idx = to_device(lb_idx,self.device)
            lb_X = to_device(lb_X,self.device)
            lb_y = to_device(lb_y,self.device)
            ulb_idx = to_device(ulb_idx,self.device)
            ulb_X  = to_device(ulb_X,self.device)
            train_result_unlabeled = self.train_unlabeled(lb_X=lb_X, lb_y=lb_y, ulb_X=ulb_X, lb_idx=lb_idx, ulb_idx=ulb_idx)
            self.end_fit_batch_unlabeled(train_result_unlabeled)
            train_result_labeled = self.train_labeled(lb_X=lb_X, lb_y=lb_y, ulb_X=ulb_X, lb_idx=lb_idx, ulb_idx=ulb_idx)
            self.end_fit_batch_labeled(train_result_labeled)
            self.it_total += 1
            self.it_epoch += 1
            if self.verbose:
                print(self.it_total,file=self.file)
                print(self.it_total)
            if valid_X is not None and self.eval_it is not None and self.it_total % self.eval_it == 0:
                self.evaluate(X=self.unlabeled_X, y=self.unlabeled_y, valid=True)
                self.evaluate(X=valid_X, y=valid_y,valid=True)
                self.valid_performance.update({"epoch_" + str(self._epoch) + "_it_" + str(self.it_epoch): self.performance})

    def train_unlabeled(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X=lb_X[0] if isinstance(lb_X,(tuple,list)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(tuple,list)) else lb_y
        w_ulb_X,s_ulb_X=ulb_X[0],ulb_X[1]
        labeled_batch_size= lb_X.shape[0]
        unlabeled_batch_size=ulb_X[0].shape[0]
        inputs=torch.cat((lb_X, w_ulb_X, s_ulb_X))
        # l_features,l_logits = self.labeled_network(inputs)
        u_features,u_logits = self.unlabeled_network(inputs)

        # l_lb_logits = l_logits[:labeled_batch_size]
        # l_w_ulb_logits, l_s_ulb_logits = l_logits[labeled_batch_size:].chunk(2)
        # l_lb_features = l_features[:labeled_batch_size]
        # l_w_ulb_features, l_s_ulb_features = l_features[labeled_batch_size:].chunk(2)

        u_lb_logits = u_logits[:labeled_batch_size]
        u_w_ulb_logits, u_s_ulb_logits = u_logits[labeled_batch_size:].chunk(2)
        u_lb_features = u_features[:labeled_batch_size]
        u_w_ulb_features, u_s_ulb_features = u_features[labeled_batch_size:].chunk(2)

        # self.labeled_mean=(self.labeled_mean*self.labeled_num+torch.sum(l_lb_features,dim=0))/(self.labeled_num+labeled_batch_size)
        # self.labeled_num+=labeled_batch_size

        # self.unlabeled_mean=(self.unlabeled_mean*self.unlabeled_num+torch.sum(u_w_ulb_features,dim=0))/(self.unlabeled_num+unlabeled_batch_size)
        # self.unlabeled_num+=unlabeled_batch_size
        # w = cp.Variable(labeled_batch_size)
        # A = to_numpy(u_lb_features.T)
        # b = to_numpy(self.unlabeled_mean)
        # objective = cp.Minimize(cp.sum_squares(A @ w - b))
        # constraints = [0 <= w,w<=1]
        # prob = cp.Problem(objective, constraints)
        # result = prob.solve()
        # weight_labeled = torch.Tensor(w.value).to(self.device)
        # weight_labeled=weight_labeled*labeled_batch_size/torch.sum(weight_labeled)
        weight_labeled = torch.Tensor(
            KMM(kernel_type='linear').fit(to_numpy(u_lb_features), to_numpy(u_w_ulb_features))).to('cuda:0').detach()
        return u_lb_logits,lb_y,u_w_ulb_logits,u_s_ulb_logits,weight_labeled,u_lb_features

        # if unlabeled_batch_size<=u_w_ulb_features.shape[1]:
        #     w = cp.Variable(labeled_batch_size)
        #     A=to_numpy(u_lb_features.T)
        #     b=to_numpy(self.unlabeled_mean)
        #     objective = cp.Minimize(cp.sum_squares(A * w - b))
        #     constraints = [0 <= w]
        #     prob = cp.Problem(objective, constraints)
        #     result = prob.solve()
        #     weight_labeled=torch.Tensor(w.value).to(self.device)
        #     # weight_labeled=torch.matmul(torch.mm(torch.linalg.inv(torch.mm(u_lb_features,u_lb_features.T)),u_lb_features),self.unlabeled_mean)
        #     # self.weight_unlabeled=torch.matmul(torch.mm(torch.linalg.inv(torch.mm(l_w_ulb_features,l_w_ulb_features.T)),l_w_ulb_features),self.labeled_mean)
        # else:
        #     weight_labeled=torch.matmul(torch.mm(u_lb_features,torch.linalg.inv(torch.mm(u_lb_features.T,u_lb_features))),self.unlabeled_mean)
            # self.weight_unlabeled=torch.matmul(torch.mm(l_w_ulb_features,torch.mm(torch.linalg.inv(l_w_ulb_features.T),l_w_ulb_features)),self.labeled_mean)

    def train_labeled(self,lb_X,lb_y,ulb_X,lb_idx=None,ulb_idx=None,*args,**kwargs):
        lb_X=lb_X[0] if isinstance(lb_X,(tuple,list)) else lb_X
        lb_y=lb_y[0] if isinstance(lb_y,(tuple,list)) else lb_y
        w_ulb_X,s_ulb_X=ulb_X[0],ulb_X[1]
        labeled_batch_size= lb_X.shape[0]
        unlabeled_batch_size=ulb_X[0].shape[0]
        inputs=torch.cat((lb_X, w_ulb_X, s_ulb_X))
        l_features, l_logits = self.labeled_network(inputs)
        l_lb_logits = l_logits[:labeled_batch_size]
        l_w_ulb_logits, l_s_ulb_logits = l_logits[labeled_batch_size:].chunk(2)
        l_lb_features = l_features[:labeled_batch_size]
        l_w_ulb_features, l_s_ulb_features = l_features[labeled_batch_size:].chunk(2)
        # self.labeled_mean=(self.labeled_mean*self.labeled_num+torch.sum(l_lb_features,dim=0))/(self.labeled_num+labeled_batch_size)
        # self.labeled_num+=labeled_batch_size
        # if labeled_batch_size <= l_lb_features.shape[1]:
        #     weight_unlabeled = torch.matmul(
        #         torch.mm(torch.linalg.inv(torch.mm(l_w_ulb_features, l_w_ulb_features.T)), l_w_ulb_features),
        #         self.labeled_mean)
        # else:
        #     weight_unlabeled = torch.matmul(
        #         torch.mm(l_w_ulb_features, torch.mm(torch.linalg.inv(l_w_ulb_features.T), l_w_ulb_features)),
        #         self.labeled_mean)
        # w = cp.Variable(unlabeled_batch_size)
        # A = to_numpy(l_w_ulb_features.T)
        # b = to_numpy(self.labeled_mean)
        # objective = cp.Minimize(cp.sum_squares(A @ w - b))
        # constraints = [0 <= w,w<=1]
        # prob = cp.Problem(objective, constraints)
        # result = prob.solve()
        # weight_unlabeled = torch.Tensor(w.value).to(self.device)
        # weight_unlabeled = weight_unlabeled * unlabeled_batch_size / torch.sum(weight_unlabeled)
        return l_lb_logits,lb_y,l_s_ulb_logits,l_lb_features

    def get_loss_unlabeled(self,train_result,*args,**kwargs):
        u_lb_logits,lb_y,u_w_ulb_logits,u_s_ulb_logits,weight_labeled,u_lb_features = train_result
        sup_loss=(Cross_Entropy(reduction='none')(logits=u_lb_logits,targets=lb_y)*weight_labeled.detach()).mean()
        # print(sup_loss)
        self.it_pseudo_label = torch.softmax(u_w_ulb_logits.detach() / self.T, dim=-1)
        max_probs, targets_u = torch.max(self.it_pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()
        unsup_loss = (Cross_Entropy(reduction='none')(u_s_ulb_logits, targets_u) * mask).mean()
        # print(unsup_loss)
        loss=Semi_Supervised_Loss(lambda_u =self.lambda_u)(sup_loss,unsup_loss)
        return loss

    def get_loss_labeled(self,train_result,*args,**kwargs):
        l_lb_logits,lb_y,l_s_ulb_logits,l_lb_features = train_result
        sup_loss=Cross_Entropy(reduction='mean')(logits=l_lb_logits,targets=lb_y)
        max_probs, targets_u = torch.max(self.it_pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()
        unsup_loss = (Cross_Entropy(reduction='none')(l_s_ulb_logits, targets_u) * mask).mean()
        # print(unsup_loss)
        loss=Semi_Supervised_Loss(lambda_u =self.lambda_u)(sup_loss,unsup_loss)
        return loss

    def start_predict(self, *args, **kwargs):
        self.labeled_network.eval()
        if self.labeled_ema is not None:
            self.labeled_ema.apply_shadow()
        self.y_est = torch.Tensor().to(self.device)

    @torch.no_grad()
    def estimate(self, X, idx=None, *args, **kwargs):
        _, outputs = self.labeled_network(X)
        return outputs

    def predict(self,X=None,valid=None):
        return DeepModelMixin.predict(self,X=X,valid=valid)
