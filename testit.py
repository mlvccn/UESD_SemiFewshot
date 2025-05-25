import os
import json
import pickle
import pprint
import torch
import argparse
import random
import numpy as np

from src import datasets, models
from src.models import backbones
from torch.utils.data import DataLoader

from haven import haven_utils as hu


def trainval(exp_dict, savedir_base, datadir, ckpt, title = None, num_workers=0):
    # bookkeeping
    # ---------------

    # get experiment directory
    # exp_id = hu.hash_dict(exp_dict)
    exp_id = title
    savedir = os.path.join(savedir_base, exp_id)

    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    pprint.pprint(exp_dict)
    print('Experiment saved in %s' % savedir)

    # load datasets
    # ==========================
    test_set = datasets.get_dataset(dataset_name=exp_dict["dataset_test"],
                data_root=os.path.join(datadir, exp_dict["dataset_test_root"]),
                split="test", 
                transform=exp_dict["transform_val"], 
                classes=exp_dict["classes_test"],
                support_size=exp_dict["support_size_test"],
                query_size=exp_dict["query_size_test"], 
                n_iters=exp_dict["test_iters"],
                unlabeled_size=exp_dict["unlabeled_size_test"])

    # get dataloaders
    # ==========================
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        drop_last=False)

    # create model and trainer
    # ==========================

    # Create model, opt, wrapper
    backbone = backbones.get_backbone(backbone_name=exp_dict['model']["backbone"], exp_dict=exp_dict)
    model = models.get_model(model_name=exp_dict["model"]['name'], backbone=backbone,
                             n_classes=exp_dict["n_classes"],
                             exp_dict=exp_dict,
                             pretrained_weights_dir='just some stupid path',
                             savedir_base=None,
                             load_pretrained=False)

    if ckpt is not None:
        print('=> Model from `{}` loaded'.format(ckpt))
        model.model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'], strict=True)
    
    # Pretrain or Fine-tune or run SSL
    assert exp_dict["model"]['name'] == 'ssl'
    # runs the SSL experiments
    score_list_path = os.path.join(savedir, 'score_list.pkl')
    if not os.path.exists(score_list_path):
        test_dict = model.test_on_loader(test_loader, max_iter=None)
        hu.save_pkl(score_list_path, [test_dict])
    else:
        print('=> Cached result loaded')

    with open(score_list_path, 'rb') as f:
        test_dict = pickle.load(f)
    print('=>', test_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('cfg', type=str, help='json config path')
    parser.add_argument('ckpt', type=str, help='checkpoint path')

    parser.add_argument('-sb', '--savedir_base', required=True,
                        help='Testing result wil be saved under {savedir_base}/[hash_id]')
    parser.add_argument('-d', '--datadir', default='/data16t')
    parser.add_argument('-nw', '--num_workers', default=2, type=int)
    parser.add_argument('-s', '--selection', default='ssl',
                        help='Pseudo-label generation method, choose from {ssl, kmeans}')
    parser.add_argument('--seed', default=1996, type=str)
    parser.add_argument('-t', '--title', default=None, type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.cfg) as f:
        exp_dict = json.load(f)

    exp_dict['selection'] = args.selection
    exp_dict['seed'] = args.seed
    correct, wrong, num = 0., 0., 0.

    trainval(exp_dict=exp_dict,
             savedir_base=args.savedir_base,
             datadir=args.datadir,
             ckpt=args.ckpt,
             title=args.title,
             num_workers=args.num_workers)
