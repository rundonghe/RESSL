import numpy as np
from LAMDA_SSL.Dataset.SemiDataset import SemiDataset
from LAMDA_SSL.Base.VisionMixin import VisionMixin
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import os
import pickle
from LAMDA_SSL.Split.DataSplit import DataSplit
from LAMDA_SSL.Dataset.TrainDataset import TrainDataset
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset

class ImageNet32(SemiDataset,VisionMixin):
    base_folder = "ImageNet32"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
    mean=[0.5071, 0.4865, 0.4409]#[0.4914, 0.4822, 0.4465]
    std=[0.2673, 0.2564, 0.2762]#[0.2471, 0.2435, 0.2616]

    def __init__(
        self,
        root: str,
        default_transforms=False,
        pre_transform=None,
        transforms=None,
        transform = None,
        target_transform = None,
        unlabeled_transform=None,
        valid_transform=None,
        test_transform=None,
        valid_size=None,
        labeled_size=0.1,
        stratified=False,
        shuffle=True,
        random_state=None,
        download: bool = False,

    ) -> None:
        self.default_transforms=default_transforms
        self.labeled_X=None
        self.labeled_y=None
        self.unlabeled_X=None
        self.unlabeled_y=None
        self.valid_X=None
        self.valid_y=None
        self.test_X=None
        self.test_y=None

        self.labeled_dataset=None
        self.unlabeled_dataset=None
        self.train_dataset=None
        self.valid_dataset = None
        self.test_dataset=None

        self.data_initialized=False

        self.len_test=None
        self.len_valid = None
        self.len_labeled=None
        self.len_unlabeled=None

        self.labeled_X_indexing_method=None
        self.labeled_y_indexing_method =None
        self.unlabeled_X_indexing_method =None
        self.unlabeled_y_indexing_method =None
        self.valid_X_indexing_method=None
        self.valid_indexing_method=None
        self.test_X_indexing_method=None
        self.test_y_indexing_method=None


        SemiDataset.__init__(self,pre_transform=pre_transform,transforms=transforms,transform=transform, target_transform=target_transform,
                             unlabeled_transform=unlabeled_transform,test_transform=test_transform,
                             valid_transform=valid_transform,labeled_size=labeled_size,valid_size=valid_size,
                             stratified=stratified,shuffle=shuffle,random_state=random_state)
        VisionMixin.__init__(self,mean=self.mean,std=self.std)


        if isinstance(root, (str, bytes)):
            root = os.path.expanduser(root)
        self.root = root

        if download:
            self.download()

        # if not self._check_integrity():
        #     raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # self._load_meta()
        if self.default_transforms:
            self.init_default_transforms()
        self.init_dataset()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict

    def load_databatch(self, data_folder, idx, img_size=32):
        data_file = os.path.join(data_folder, 'train_data_batch_')
        d = self.unpickle(data_file + str(idx))

        x = d['data']
        y = d['labels']
        mean_image = d['mean']
        # x = x / np.float32(255)
        # mean_image = mean_image / np.float32(255)

        # Labels are indexed from 1, shift it so that indexes start at 0
        y = [i - 1 for i in y]
        data_size = x.shape[0]

        # x -= mean_image

        img_size2 = img_size * img_size

        x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3))

        # create mirrored images
        X_train = x[0:data_size, :, :, :]
        Y_train = y[0:data_size]

        # Y_train = np.random.choice(list(range(3)), size=data_size, replace=True)
        # X_train_flip = X_train[:, :, :, ::-1]
        # Y_train_flip = Y_train
        # X_train = np.concatenate((X_train, X_train_flip), axis=0)
        # Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

        return dict(
            images=np.array(X_train),
            labels=np.array(Y_train)
        )

    def _init_dataset(self):
        data_folder = '/data/imagenet'
        test_file = os.path.join(data_folder, 'val_data')
        img_size = 32
        d = self.unpickle(test_file)
        x = d['data']
        y = d['labels']
        # x = x / np.float32(255)
        y = np.array([i - 1 for i in y])
        img_size2 = img_size * img_size

        x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
        test_X = x.reshape((x.shape[0], img_size, img_size, 3))
        test_y = y

        index = test_y < 100
        test_y = test_y[index]
        test_X = test_X[index]


        train_X = []
        train_y = []

        # now load the picked numpy arrays
        for i in range(10):
            d = self.load_databatch(data_folder, i + 1, img_size=32)
            index = d["labels"] < 100

            train_X.append(d["images"][index])
            train_y.extend(d["labels"][index])
        train_X = np.vstack(train_X)



        if self.valid_size is not None:
            valid_X, valid_y, train_X, train_y = DataSplit(X=train_X, y=train_y,
                                                                   size_split=self.valid_size,
                                                                   stratified=self.stratified,
                                                                   shuffle=self.shuffle,
                                                                   random_state=self.random_state
                                                                   )
        else:
            valid_X=None
            valid_y=None

        if self.labeled_size is not None:
            labeled_X, labeled_y, unlabeled_X, unlabeled_y = DataSplit(X=train_X, y=train_y,
                                                                   size_split=self.labeled_size,
                                                                   stratified=self.stratified,
                                                                   shuffle=self.shuffle,
                                                                   random_state=self.random_state
                                                                   )
        else:
            labeled_X, labeled_y=train_X,train_y
            unlabeled_X, unlabeled_y=None,None

        self.test_dataset=LabeledDataset(pre_transform=self.pre_transform,transform=self.test_transform)
        self.test_dataset.init_dataset(test_X,test_y)
        self.valid_dataset=LabeledDataset(pre_transform=self.pre_transform,transform=self.valid_transform)
        self.valid_dataset.init_dataset(valid_X,valid_y)
        self.train_dataset = TrainDataset(pre_transform=self.pre_transform,transforms=self.transforms,transform=self.transform,
                                          target_transform=self.target_transform,unlabeled_transform=self.unlabeled_transform)
        labeled_dataset=LabeledDataset(pre_transform=self.pre_transform,transforms=self.transforms,transform=self.transform,
                                          target_transform=self.target_transform)
        labeled_dataset.init_dataset(labeled_X, labeled_y)
        unlabeled_dataset=UnlabeledDataset(pre_transform=self.pre_transform,transform=self.unlabeled_transform)
        unlabeled_dataset.init_dataset(unlabeled_X, unlabeled_y)
        self.train_dataset.init_dataset(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset)


