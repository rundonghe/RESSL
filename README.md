# Re-evaluating the Impact of Unseen-Class Unlabeled Data on SSL Model
This repository contains PyTorch implementation for our paper: [Re-evaluating the Impact of Unseen-Class Unlabeled Data on SSL Model]()


## Quick Start
### 1. varying the number of unseen-class examples
seen = unlabels_num * seen_ratio, unlabels_num, seen_ratio=0.2, change unseen_ratio
```train
sh scripts/unseenc.sh 
```

### 2. varying the number of unseen-class categories
seen = unlabels_num * seen_ratio, unlabels_num, seen_ratio=0.2, unseen_ratio=0.2, change unseen_class_num
```train
sh scripts/unseen_class_c.sh 
```

### 3. varying the index of unseen classes
seen = unlabels_num * seen_ratio, unlabels_num, seen_ratio=0.2, unseen_ratio=0.2, unseen_class_num=1, change unseen_class_index
```train
sh scripts/unseen_index_c.sh 
```

### 4. varying the degrees of nearness in unseen classes
#### 4.1 with the number of unseen-class categories
seen = unlabels_num * seen_ratio, unlabels_num=25000, seen_ratio=0.2, unseen_ratio=0.2, with MNIST, change unseen_class_num
```train
sh scripts/unseen_near_c.sh 
```
#### 4.1 with the number of unseen-class index
seen = unlabels_num * seen_ratio, unlabels_num=25000, seen_ratio=0.2, unseen_ratio=0.2, unseen_class_num=1, change unseen_class_index
```train
sh scripts/unseen_near_index_c.sh 
```

### 5. varying the label distribution in unseen classes
seen = unlabels_num * seen_ratio, unlabels_num=25000, seen_ratio=0.2, unseen_ratio=0.4, unseen_class_num=5, change imb_factor
```train
sh scripts/unseen_imbalance.sh 
```
