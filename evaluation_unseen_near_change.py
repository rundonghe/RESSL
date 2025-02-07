from LAMDA_SSL.Dataset.Vision.CIFAR10 import CIFAR10
from LAMDA_SSL.Dataset.Vision.Mnist import Mnist
from LAMDA_SSL.Algorithm.Classification.VAT import VAT
from LAMDA_SSL.Sampler.RandomSampler import RandomSampler
from LAMDA_SSL.Sampler.SequentialSampler import SequentialSampler
from LAMDA_SSL.Dataloader.UnlabeledDataloader import UnlabeledDataLoader
from LAMDA_SSL.Dataloader.LabeledDataloader import LabeledDataLoader
from LAMDA_SSL.Augmentation.Vision.RandomHorizontalFlip import RandomHorizontalFlip
from LAMDA_SSL.Augmentation.Vision.RandomCrop import RandomCrop
from LAMDA_SSL.Augmentation.Vision.RandAugment import RandAugment
from LAMDA_SSL.Augmentation.Vision.Cutout import Cutout
from LAMDA_SSL.Network.ResNet50Fc import ResNet50Fc
from LAMDA_SSL.Opitimizer.SGD import SGD
from LAMDA_SSL.Scheduler.CosineWarmup import CosineWarmup
from LAMDA_SSL.Split.DataSplit import DataSplit
from LAMDA_SSL.Algorithm.Classification.OpenMatch import OpenMatch
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Transform.ToImage import ToImage
from LAMDA_SSL.Transform.ToTensor import ToTensor
from LAMDA_SSL.Transform.Vision.Normalization import Normalization
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Algorithm.Classification.UASD import UASD
from LAMDA_SSL.Algorithm.Classification.Supervised import Supervised
from LAMDA_SSL.Algorithm.Classification.PiModel import PiModel
from LAMDA_SSL.Algorithm.Classification.UDA import UDA
from LAMDA_SSL.Algorithm.Classification.PseudoLabel import PseudoLabel
from LAMDA_SSL.Algorithm.Classification.ICT import ICT
from LAMDA_SSL.Algorithm.Classification.MixMatch import MixMatch
from LAMDA_SSL.Algorithm.Classification.FlexMatch import FlexMatch
from LAMDA_SSL.Algorithm.Classification.FreeMatch import FreeMatch
from LAMDA_SSL.Algorithm.Classification.FixMatch import FixMatch
from LAMDA_SSL.Algorithm.Classification.ReMixMatch import ReMixMatch
from LAMDA_SSL.Algorithm.Classification.MixMatch import MixMatch
from LAMDA_SSL.Algorithm.Classification.Fix_A_Step import Fix_A_Step
from LAMDA_SSL.Transform.ToTensor import ToTensor
from LAMDA_SSL.Transform.Vision.Normalization import Normalization
from LAMDA_SSL.Algorithm.Classification.SoftMatch import SoftMatch
from LAMDA_SSL.Algorithm.Classification.VAT import VAT
from LAMDA_SSL.Algorithm.Classification.OpenMatch import OpenMatch
from LAMDA_SSL.Algorithm.Classification.MTCF import MTCF
from LAMDA_SSL.Algorithm.Classification.CAFA_Pi import CAFA
from PIL import Image
import os
import argparse
import copy
from sklearn.pipeline import Pipeline
import numpy as np
import torch
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_mena_std(imgs):
    tmp_imgs=[]
    for _ in range(imgs.shape[0]):
        tmp_imgs.append(imgs[_].transpose(2,0,1).reshape(3,-1))
    tmp_imgs=np.hstack(tmp_imgs)
    mean = np.mean(tmp_imgs / 255, axis=1)
    std = np.std(tmp_imgs / 255, axis=1)
    return mean,std


def worker_init():
    worker_seed = seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def convert_to_rgb(image):
    image_rgb = np.stack((image,) * 3, axis=-1)
    return image_rgb

# 函数：调整图像大小为32x32（可选）
def resize_image(image):
    image_pil = Image.fromarray(image)
    return np.array(image_pil.resize((32, 32)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/root/hrd_pg/safeSSL-evaluation/data')
    parser.add_argument('--dataset', type=str, default='cifar-10')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--iteration', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda:6')
    parser.add_argument('--labels', type=int, help="the number of L", default=100)
    parser.add_argument('--hint', type=str, default='')
    parser.add_argument('--algo', type=str, default='OpenMatch')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--unseen_ratio', type=float, default=0.5)
    parser.add_argument('--seen_ratio', type=float, default=0.5)
    parser.add_argument('--unseen_class_num', type=int, default=5)


    args = parser.parse_args()
    num_classes = args.num_classes
    batch_size = args.batch_size
    iteration = args.iteration
    device = args.device
    labels = args.labels
    seen_ratio = args.seen_ratio
    unseen_ratio = args.unseen_ratio

    # Load data
    if args.dataset == 'cifar-10':
        dataset = CIFAR10(root=args.root,
                          labeled_size=None, stratified=False, shuffle=False, download=False, default_transforms=True)
        far_dataset = Mnist(root=args.root,
                          labeled_size=None, stratified=False, shuffle=False, download=True, default_transforms=True)

    performance_list = []
    performance_list_r = []
    for seed in range(3):
        all_labeled_X = dataset.labeled_X
        all_labeled_y = np.array(dataset.labeled_y)
        test_X = dataset.test_X
        test_y = np.array(dataset.test_y)
        set_seed(seed)
        far_all_labeled_X = far_dataset.labeled_X
        far_all_labeled_y = np.array(far_dataset.labeled_y)

        S_X, S_y = all_labeled_X[all_labeled_y < int(num_classes / 2)], all_labeled_y[
            all_labeled_y < int(num_classes / 2)]
        T_X, T_y = far_all_labeled_X[
            far_all_labeled_y >= int(num_classes / 2 + num_classes / 2 - args.unseen_class_num)], far_all_labeled_y[
            far_all_labeled_y >= int(num_classes / 2 + num_classes / 2 - args.unseen_class_num)]
        test_X, test_y = test_X[test_y < int(num_classes / 2)], test_y[test_y < int(num_classes / 2)]
        labeled_X, labeled_y, _S_X, _S_y = DataSplit(stratified=True, shuffle=True, random_state=seed,
                                                     X=S_X, y=S_y, size_split=args.labels)
        unlabels_num = S_X.shape[0]
        print("unlabels_num:", unlabels_num)  # 25000

        # Load unlabeled data
        target_X_in, target_y_in, target_X_in_r, target_y_in_r = DataSplit(stratified=True,
                                                                           shuffle=True, random_state=seed,
                                                                           X=_S_X, y=_S_y,
                                                                           size_split=int(
                                                                               unlabels_num * seen_ratio))
        target_X_out, target_y_out, target_X_out_r, target_y_out_r = DataSplit(stratified=True,
                                                                               shuffle=True,
                                                                               random_state=seed,
                                                                               X=T_X,
                                                                               y=T_y,
                                                                               size_split=int(
                                                                                   unlabels_num * unseen_ratio))

        target_X_out = np.array([resize_image(convert_to_rgb(img)) for img in target_X_out])
        if unseen_ratio == 0:
            unlabeled_X = target_X_in
            unlabeled_y = target_y_in
        else:
            unlabeled_X = np.concatenate((target_X_in, target_X_out), axis=0)
            unlabeled_y = np.concatenate((target_y_in, target_y_out), axis=0)

        labeled_sampler = RandomSampler(replacement=True, num_samples=batch_size * iteration)
        unlabeled_sampler = RandomSampler(replacement=True)
        valid_sampler = SequentialSampler()
        test_sampler = SequentialSampler()

        labeled_dataloader = LabeledDataLoader(batch_size=batch_size, num_workers=0, drop_last=True,
                                               worker_init_fn=worker_init)
        unlabeled_dataloader = UnlabeledDataLoader(num_workers=0, drop_last=True, worker_init_fn=worker_init)
        valid_dataloader = UnlabeledDataLoader(batch_size=batch_size, num_workers=0, drop_last=False,
                                               worker_init_fn=worker_init)
        test_dataloader = UnlabeledDataLoader(batch_size=batch_size, num_workers=0, drop_last=False,
                                              worker_init_fn=worker_init)

        # Load augmentation
        train_pre_transform = Pipeline([('ToImage', ToImage()), ])
        valid_pre_transform = Pipeline([('ToImage', ToImage()), ])
        test_pre_transform = Pipeline([('ToImage', ToImage()), ])
        weak_augmentation = Pipeline([('RandomHorizontalFlip', RandomHorizontalFlip()),
                                      ('RandomCrop', RandomCrop(padding=0.125, padding_mode='reflect')),
                                      ])
        open_augmentation = Pipeline([('RandomHorizontalFlip', RandomHorizontalFlip()),
                                      # ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
                                      ])
        strong_augmentation = Pipeline([('RandomHorizontalFlip', RandomHorizontalFlip()),
                                        ('RandomCrop', RandomCrop(padding=0.125, padding_mode='reflect')),
                                        ('RandAugment', RandAugment(n=2, m=10, num_bins=10)),
                                        ('Cutout', Cutout(v=0.5, fill=(127, 127, 127)))
                                        ])
        augmentation = {
            'open_augmentation': open_augmentation,
            'weak_augmentation': weak_augmentation,
            'strong_augmentation': strong_augmentation
        }

        optimizer = SGD(lr=5e-4, momentum=0.9)
        optimizer_wnet = SGD(lr=5e-4, momentum=0.9)
        scheduler = CosineWarmup(num_cycles=7. / 16, num_training_steps=iteration)
        scheduler_wnet = CosineWarmup(num_cycles=7. / 16, num_training_steps=iteration)

        # Load algo
        if args.algo == 'OpenMatch':
            network = ResNet50Fc(num_classes=[num_classes // 2, num_classes // 2 * 2], bias=[True, False],
                                 output_feature=False)
            model = OpenMatch(
                # lambda_u=1.0,
                mu=1, weight_decay=5e-4,
                eval_it=None,
                epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                device=device,
                labeled_sampler=copy.deepcopy(labeled_sampler),
                unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                valid_sampler=copy.deepcopy(valid_sampler),
                test_sampler=copy.deepcopy(test_sampler),
                labeled_dataloader=copy.deepcopy(labeled_dataloader),
                unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                valid_dataloader=copy.deepcopy(valid_dataloader),
                test_dataloader=copy.deepcopy(test_dataloader),
                augmentation=copy.deepcopy(augmentation),
                network=copy.deepcopy(network),
                optimizer=copy.deepcopy(optimizer),
                scheduler=copy.deepcopy(scheduler),
                verbose=False
            )
        if args.algo == 'UASD':
            network = ResNet50Fc(num_classes=num_classes // 2, output_feature=False)
            model = UASD(
                lambda_u=1.0,
                mu=1, weight_decay=5e-4,
                eval_it=None,
                epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                device=device,
                labeled_sampler=copy.deepcopy(labeled_sampler),
                unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                valid_sampler=copy.deepcopy(valid_sampler),
                test_sampler=copy.deepcopy(test_sampler),
                labeled_dataloader=copy.deepcopy(labeled_dataloader),
                unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                valid_dataloader=copy.deepcopy(valid_dataloader),
                test_dataloader=copy.deepcopy(test_dataloader),
                augmentation=copy.deepcopy(augmentation),
                network=copy.deepcopy(network),
                optimizer=copy.deepcopy(optimizer),
                scheduler=copy.deepcopy(scheduler),
                verbose=False
            )
        if args.algo == 'MTCF':
            network = ResNet50Fc(num_classes=num_classes // 2, output_feature=False)
            model = MTCF(
                # lambda_u=1.0,
                mu=1, weight_decay=5e-4,
                eval_it=None,
                epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                device=device,
                labeled_sampler=copy.deepcopy(labeled_sampler),
                unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                valid_sampler=copy.deepcopy(valid_sampler),
                test_sampler=copy.deepcopy(test_sampler),
                labeled_dataloader=copy.deepcopy(labeled_dataloader),
                unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                valid_dataloader=copy.deepcopy(valid_dataloader),
                test_dataloader=copy.deepcopy(test_dataloader),
                augmentation=copy.deepcopy(augmentation),
                network=copy.deepcopy(network),
                optimizer=copy.deepcopy(optimizer),
                scheduler=copy.deepcopy(scheduler),
                verbose=False
            )
        if args.algo == 'FlexMatch':
            network = ResNet50Fc(num_classes=num_classes // 2, output_feature=False)
            model = FlexMatch(
                lambda_u=1.0,
                mu=1, weight_decay=5e-4,
                eval_it=None,
                epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                device=device,
                labeled_sampler=copy.deepcopy(labeled_sampler),
                unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                valid_sampler=copy.deepcopy(valid_sampler),
                test_sampler=copy.deepcopy(test_sampler),
                labeled_dataloader=copy.deepcopy(labeled_dataloader),
                unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                valid_dataloader=copy.deepcopy(valid_dataloader),
                test_dataloader=copy.deepcopy(test_dataloader),
                augmentation=copy.deepcopy(augmentation),
                network=copy.deepcopy(network),
                optimizer=copy.deepcopy(optimizer),
                scheduler=copy.deepcopy(scheduler), verbose=False
            )
        if args.algo == 'FixMatch':
            network = ResNet50Fc(num_classes=num_classes // 2, output_feature=False)
            model = FixMatch(
                lambda_u=1.0,
                mu=1, weight_decay=5e-4,
                eval_it=None,
                epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                device=device,
                labeled_sampler=copy.deepcopy(labeled_sampler),
                unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                valid_sampler=copy.deepcopy(valid_sampler),
                test_sampler=copy.deepcopy(test_sampler),
                labeled_dataloader=copy.deepcopy(labeled_dataloader),
                unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                valid_dataloader=copy.deepcopy(valid_dataloader),
                test_dataloader=copy.deepcopy(test_dataloader),
                augmentation=copy.deepcopy(augmentation),
                network=copy.deepcopy(network),
                optimizer=copy.deepcopy(optimizer),
                scheduler=copy.deepcopy(scheduler),
                verbose=False
            )
        if args.algo == 'Supervised':
            network = ResNet50Fc(num_classes=num_classes // 2, output_feature=False)
            model = Supervised(lambda_u=1.0,
                               mu=1, weight_decay=5e-4,
                               eval_it=None,
                               epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                               device=device,
                               labeled_sampler=copy.deepcopy(labeled_sampler),
                               unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                               valid_sampler=copy.deepcopy(valid_sampler),
                               test_sampler=copy.deepcopy(test_sampler),
                               labeled_dataloader=copy.deepcopy(labeled_dataloader),
                               unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                               valid_dataloader=copy.deepcopy(valid_dataloader),
                               test_dataloader=copy.deepcopy(test_dataloader),
                               augmentation=copy.deepcopy(augmentation),
                               network=copy.deepcopy(network),
                               optimizer=copy.deepcopy(optimizer),
                               scheduler=copy.deepcopy(scheduler),
                               verbose=False
                               )
        if args.algo == 'UDA':
            network = ResNet50Fc(num_classes=num_classes // 2, output_feature=False)
            model = UDA(lambda_u=1.0,
                        mu=1, weight_decay=5e-4,
                        eval_it=None,
                        epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                        device=device,
                        labeled_sampler=copy.deepcopy(labeled_sampler),
                        unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                        valid_sampler=copy.deepcopy(valid_sampler),
                        test_sampler=copy.deepcopy(test_sampler),
                        labeled_dataloader=copy.deepcopy(labeled_dataloader),
                        unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                        valid_dataloader=copy.deepcopy(valid_dataloader),
                        test_dataloader=copy.deepcopy(test_dataloader),
                        augmentation=copy.deepcopy(augmentation),
                        network=copy.deepcopy(network),
                        optimizer=copy.deepcopy(optimizer),
                        scheduler=copy.deepcopy(scheduler),
                        verbose=False
                        )
        if args.algo == 'SoftMatch':
            network = ResNet50Fc(num_classes=num_classes // 2, output_feature=False)
            model = SoftMatch(
                use_DA=True,
                lambda_u=1.0,
                mu=1, weight_decay=5e-4,
                eval_it=None,
                epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                device=device,
                labeled_sampler=copy.deepcopy(labeled_sampler),
                unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                valid_sampler=copy.deepcopy(valid_sampler),
                test_sampler=copy.deepcopy(test_sampler),
                labeled_dataloader=copy.deepcopy(labeled_dataloader),
                unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                valid_dataloader=copy.deepcopy(valid_dataloader),
                test_dataloader=copy.deepcopy(test_dataloader),
                augmentation=copy.deepcopy(augmentation),
                network=copy.deepcopy(network),
                optimizer=copy.deepcopy(optimizer),
                scheduler=copy.deepcopy(scheduler), verbose=False
            )
        if args.algo == 'VAT':
            network = ResNet50Fc(num_classes=num_classes // 2, output_feature=False)
            model = VAT(
                mu=1, weight_decay=5e-4,
                eval_it=None,
                epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                device=device,
                labeled_sampler=copy.deepcopy(labeled_sampler),
                unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                valid_sampler=copy.deepcopy(valid_sampler),
                test_sampler=copy.deepcopy(test_sampler),
                labeled_dataloader=copy.deepcopy(labeled_dataloader),
                unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                valid_dataloader=copy.deepcopy(valid_dataloader),
                test_dataloader=copy.deepcopy(test_dataloader),
                augmentation=copy.deepcopy(augmentation),
                network=copy.deepcopy(network),
                optimizer=copy.deepcopy(optimizer),
                scheduler=copy.deepcopy(scheduler),
                verbose=False
            )
        if args.algo == 'CAFA':
            network = ResNet50Fc(num_classes=num_classes // 2, output_feature=True)
            model = CAFA(
                lambda_u=1.0,
                mu=1, weight_decay=5e-4,
                eval_it=None,
                epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                device=device,
                labeled_sampler=copy.deepcopy(labeled_sampler),
                unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                valid_sampler=copy.deepcopy(valid_sampler),
                test_sampler=copy.deepcopy(test_sampler),
                labeled_dataloader=copy.deepcopy(labeled_dataloader),
                unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                valid_dataloader=copy.deepcopy(valid_dataloader),
                test_dataloader=copy.deepcopy(test_dataloader),
                augmentation=copy.deepcopy(augmentation),
                network=copy.deepcopy(network),
                optimizer=copy.deepcopy(optimizer),
                scheduler=copy.deepcopy(scheduler),
                verbose=False
            )
        if args.algo == 'Fix_A_Step':
            network = ResNet50Fc(num_classes=num_classes // 2, output_feature=False)
            model = Fix_A_Step(
                mu=1, weight_decay=5e-4,
                eval_it=None,
                epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                device=device,
                labeled_sampler=copy.deepcopy(labeled_sampler),
                unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                valid_sampler=copy.deepcopy(valid_sampler),
                test_sampler=copy.deepcopy(test_sampler),
                labeled_dataloader=copy.deepcopy(labeled_dataloader),
                unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                valid_dataloader=copy.deepcopy(valid_dataloader),
                test_dataloader=copy.deepcopy(test_dataloader),
                augmentation=copy.deepcopy(augmentation),
                network=copy.deepcopy(network),
                optimizer=copy.deepcopy(optimizer),
                scheduler=copy.deepcopy(scheduler),
                verbose=False
            )
        if args.algo == 'PseudoLabel':
            network = ResNet50Fc(num_classes=num_classes // 2, output_feature=False)
            model = PseudoLabel(lambda_u=1.0,
                                mu=1, weight_decay=5e-4,
                                eval_it=None,
                                epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                                device=device,
                                labeled_sampler=copy.deepcopy(labeled_sampler),
                                unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                                valid_sampler=copy.deepcopy(valid_sampler),
                                test_sampler=copy.deepcopy(test_sampler),
                                labeled_dataloader=copy.deepcopy(labeled_dataloader),
                                unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                                valid_dataloader=copy.deepcopy(valid_dataloader),
                                test_dataloader=copy.deepcopy(test_dataloader),
                                augmentation=copy.deepcopy(augmentation),
                                network=copy.deepcopy(network),
                                optimizer=copy.deepcopy(optimizer),
                                scheduler=copy.deepcopy(scheduler),
                                verbose=False
                                )
        if args.algo == 'FreeMatch':
            network = ResNet50Fc(num_classes=num_classes // 2, output_feature=False)
            model = FreeMatch(
                lambda_u=1.0,
                mu=1, weight_decay=5e-4,
                eval_it=None,
                epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                device=device,
                labeled_sampler=copy.deepcopy(labeled_sampler),
                unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                valid_sampler=copy.deepcopy(valid_sampler),
                test_sampler=copy.deepcopy(test_sampler),
                labeled_dataloader=copy.deepcopy(labeled_dataloader),
                unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                valid_dataloader=copy.deepcopy(valid_dataloader),
                test_dataloader=copy.deepcopy(test_dataloader),
                augmentation=copy.deepcopy(augmentation),
                network=copy.deepcopy(network),
                optimizer=copy.deepcopy(optimizer),
                scheduler=copy.deepcopy(scheduler),
                verbose=False
            )
        if args.algo == 'PiModel':
            network = ResNet50Fc(num_classes=num_classes // 2, output_feature=False)
            model = PiModel(lambda_u=1.0,
                            mu=1, weight_decay=5e-4,
                            eval_it=None,
                            epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                            device=device,
                            labeled_sampler=copy.deepcopy(labeled_sampler),
                            unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                            valid_sampler=copy.deepcopy(valid_sampler),
                            test_sampler=copy.deepcopy(test_sampler),
                            labeled_dataloader=copy.deepcopy(labeled_dataloader),
                            unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                            valid_dataloader=copy.deepcopy(valid_dataloader),
                            test_dataloader=copy.deepcopy(test_dataloader),
                            augmentation=copy.deepcopy(augmentation),
                            network=copy.deepcopy(network),
                            optimizer=copy.deepcopy(optimizer),
                            scheduler=copy.deepcopy(scheduler),
                            verbose=False
                            )
        if args.algo == 'ICT':
            network = ResNet50Fc(num_classes=num_classes // 2, output_feature=False)
            model = ICT(lambda_u=1.0,
                        mu=1, weight_decay=5e-4,
                        eval_it=None,
                        epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                        device=device,
                        labeled_sampler=copy.deepcopy(labeled_sampler),
                        unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                        valid_sampler=copy.deepcopy(valid_sampler),
                        test_sampler=copy.deepcopy(test_sampler),
                        labeled_dataloader=copy.deepcopy(labeled_dataloader),
                        unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                        valid_dataloader=copy.deepcopy(valid_dataloader),
                        test_dataloader=copy.deepcopy(test_dataloader),
                        augmentation=copy.deepcopy(augmentation),
                        network=copy.deepcopy(network),
                        optimizer=copy.deepcopy(optimizer),
                        scheduler=copy.deepcopy(scheduler),
                        verbose=False
                        )
        print('labeled_X')
        print(labeled_X.shape)
        print('unlabeled_X')
        print(unlabeled_X.shape)
        mean, std = get_mena_std(labeled_X)
        transform = Pipeline(
            [('ToTensor', ToTensor(dtype='float', image=True)),
             ('Normalization', Normalization(mean=mean, std=std))])

        model_1 = copy.deepcopy(model)
        labeled_dataset = LabeledDataset(pre_transform=train_pre_transform, transform=transform)
        model_1.labeled_dataset = copy.deepcopy(labeled_dataset)
        unlabeled_dataset = UnlabeledDataset(pre_transform=train_pre_transform, transform=transform)
        model_1.unlabeled_dataset = copy.deepcopy(unlabeled_dataset)
        valid_dataset = UnlabeledDataset(pre_transform=valid_pre_transform, transform=transform)
        model_1.valid_dataset = copy.deepcopy(valid_dataset)
        test_dataset = UnlabeledDataset(pre_transform=test_pre_transform, transform=transform)
        model_1.test_dataset = copy.deepcopy(test_dataset)

        model_1.fit(labeled_X, labeled_y, unlabeled_X)
        pred_y = model_1.predict(test_X)
        performance = Accuracy().scoring(test_y, pred_y)
        performance_list.append(performance)
        print("performance:", performance)
    performance_list = np.array(performance_list)
    mean = performance_list.mean()
    std = performance_list.std()
    print("mean:", mean, "std:", std)






