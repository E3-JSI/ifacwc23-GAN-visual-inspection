import os
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm.auto import tqdm

RANDOM_SEED = 744
from sklearn.neural_network import MLPClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'

### DATASET
class Shavers(datasets.ImageFolder):
    """Philips shaver prints dataset. """
    def __init__(self, root='data/shavers/', dims=None, return_tensors=True,
                 normalize_transform=None, **kwargs):
        self.root = root
                
        t_list = []
        if dims:
            # First dimension is 'n_channels'
            if len(dims) == 3 and dims[0] == 1:
                t_list.append(transforms.Grayscale())
            t_list.append(transforms.Resize(dims[1:]))
        
        if return_tensors:
            t_list.append(transforms.ToTensor()) 
        
        if normalize_transform:
            t_list.append(normalize_transform)
                
        self.transforms = transforms.Compose(t_list)
        super().__init__(root, transform=self.transforms)

       
### IMAGE NORMALIZATION
dims_resnet18 = (3, 224, 224)
normalize_resnet18 = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])

### CLASSIFICATION MODEL
def get_resnet18_embeddings(dataset):
    model = resnet18(pretrained=True).to(device)
    model.eval()

    extraction_layer = model._modules.get('avgpool')
    reprs = []
    
    def copy_data(m, i, o):
        reprs.append(o.data.flatten())

    h = extraction_layer.register_forward_hook(copy_data)
    with torch.no_grad():
        for image, _ in tqdm(dataset):    
            h_x = model(image[None, :].to(device))
    h.remove()

    return torch.vstack(reprs).cpu().numpy()

def mlp_to_torch(model):
    mlp = nn.Sequential(
        nn.Linear(*model.coefs_[0].shape),
        nn.ReLU(),
        nn.Linear(*model.coefs_[1].shape)
        # nn.Softmax(dim=-1)
    )

    mlp[0].weight = nn.Parameter(torch.tensor(model.coefs_[0].T))
    mlp[0].bias = nn.Parameter(torch.tensor(model.intercepts_[0]))

    mlp[2].weight = nn.Parameter(torch.tensor(model.coefs_[1].T))
    mlp[2].bias = nn.Parameter(torch.tensor(model.intercepts_[1]))
    return mlp

def get_trained_model(dataset, labels=None):
    if labels is None:
        labels = [label for _, label in dataset]
    
    reprs = get_resnet18_embeddings(dataset)
    
    mlp = MLPClassifier(random_state=744, max_iter=10000)
    mlp.fit(reprs, labels)
    
    model = resnet18(pretrained=True)
    model.fc = mlp_to_torch(mlp)
    
    return model

def copy_images(files, dir, resize_dim=(256, 256)):
    os.makedirs(dir)
    out_names = []
    
    for i, fn in enumerate(files):
        out_names.append('%d.png' % i)
        
        img_path = dir + out_names[-1]
        shutil.copy(fn, img_path)
    
        if resize_dim:
            img = Image.open(img_path)
            img = img.resize(resize_dim, Image.ANTIALIAS)
            img.save(img_path)
            
    return out_names


from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)


def make_custom_scorer(class_to_idx):
    """ good_ind: index of the good class. """
    def custom_scorer(clf, X, y):
        y_proba = clf.predict_proba(X)
        y_pred = clf.predict(X)
        
        y_good = (y == class_to_idx['good']).astype(int)
        return {'binary': roc_auc_score(y_good, y_proba[:, class_to_idx['good']]),
                'binary_recall': metrics.recall_score(
                    y_good, (y_pred == class_to_idx['good']).astype(int), pos_label=0),
               'multiclass': roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')}
    return custom_scorer

class ReductionStratifiedKFold:
    def __init__(self, n_splits=3, keep=1.0, good_class=1, shuffle=True, random_state=0):
        self.n_splits = n_splits
        self.random_state = random_state
        self.good_class = good_class
        self.keep = keep

    def split(self, X, y, groups=None):
        rng = np.random.RandomState(self.random_state)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for train, test in skf.split(X, y):
            train_y = y[train]
            train_new = [train[train_y == self.good_class]]
            
            for c in np.unique(y):
                if c != self.good_class:
                    c_inds = train[train_y == c]
                    train_new.append(
                        rng.choice(c_inds, round(len(c_inds) * self.keep), replace=False))
            yield np.sort(np.concatenate(train_new)), test
        
    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
    
def evaluate(model, X, y, class_to_idx, keep=1.0, n_splits=10):
    cv = ReductionStratifiedKFold(n_splits=n_splits, keep=keep, 
                                 good_class=class_to_idx['good'],
                                 random_state=RANDOM_SEED)
    results = cross_validate(model, X, y, cv=cv, n_jobs=-1,
                        scoring=make_custom_scorer(class_to_idx))
    return results


def evaluate_keep(model, X, y, model_name, class_to_idx, n_splits=10, keep_ratios=[1.0, 0.75, 0.5, 0.25]):
    rs = []
    for keep in keep_ratios:
        r = evaluate(model, X, y, class_to_idx, keep, n_splits=n_splits)
        r['model'] = model_name
        r['keep'] = keep
        r['fold'] = np.arange(len(r['fit_time']))
        rs.append(pd.DataFrame(r))
    return pd.concat(rs)
