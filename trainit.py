import torch
import argparse
import pandas as pd
import os
import json
import pprint
from src import utils as ut

from src import datasets, models
from src.models import backbones
from torch.utils.data import DataLoader

from haven import haven_utils as hu
from haven import haven_chk as hc


def trainval(exp_dict, savedir_base, datadir, reset=False,
             num_workers=0, title=None,
             ckpt=None):
    # bookkeeping
    # ---------------

    # get experiment directory
    # exp_id = hu.hash_dict(exp_dict) + '-' + exp_dict['name'].rpartition('.')[0]
    exp_id = title
    savedir = os.path.join(savedir_base, exp_id)
    os.makedirs(savedir, exist_ok=True)
    ut.setup_logger(os.path.join(savedir, 'train_log.txt'))

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)

    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    pprint.pprint(exp_dict)
    print('Experiment saved in %s' % savedir)

    # load datasets
    # ==========================
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset_train"],
                                     data_root=os.path.join(datadir, exp_dict["dataset_train_root"]),
                                     split="train",
                                     transform=exp_dict["transform_train"],
                                     classes=exp_dict["classes_train"],
                                     support_size=exp_dict["support_size_train"],
                                     query_size=exp_dict["query_size_train"],
                                     n_iters=exp_dict["train_iters"],
                                     unlabeled_size=exp_dict["unlabeled_size_train"])

    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset_val"],
                                   data_root=os.path.join(datadir, exp_dict["dataset_val_root"]),
                                   split="val",
                                   transform=exp_dict["transform_val"],
                                   classes=exp_dict["classes_val"],
                                   support_size=exp_dict["support_size_val"],
                                   query_size=exp_dict["query_size_val"],
                                   n_iters=exp_dict.get("val_iters", None),
                                   unlabeled_size=exp_dict["unlabeled_size_val"])

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
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=exp_dict["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ut.get_collate(exp_dict["collate_fn"]),
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        drop_last=True)

    # create model and trainer
    # ==========================

    # Create model, opt, wrapper
    backbone = backbones.get_backbone(backbone_name=exp_dict['model']["backbone"], exp_dict=exp_dict)
    model = models.get_model(model_name=exp_dict["model"]['name'], backbone=backbone,
                             n_classes=exp_dict["n_classes"],
                             exp_dict=exp_dict,
                             pretrained_weights_dir=None,
                             savedir_base=savedir_base)

    if ckpt is not None:
        print('=> Model from `{}` loaded'.format(ckpt))
        a, b = model.model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'], strict=False)
        if a:
            print('Missing keys:', a)
        if b:
            print('Unexpected keys:', b)

    # Checkpoint
    # -----------
    checkpoint_path = os.path.join(savedir, 'checkpoint.pth')
    score_list_path = os.path.join(savedir, 'score_list.pkl')

    if os.path.exists(score_list_path):
        # resume experiment
        model.load_state_dict(hu.torch_load(checkpoint_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]['epoch'] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Run training and validation
    for epoch in range(s_epoch, exp_dict["max_epoch"]):
        score_dict = {"epoch": epoch}
        score_dict.update(model.get_lr())

        # train
        score_dict.update(model.train_on_loader(train_loader))

        # validate
        score_dict.update(model.val_on_loader(val_loader))
        # score_dict.update(model.test_on_loader(test_loader))

        # Add score_dict to score_list
        score_list += [score_dict]

        # Report
        score_df = pd.DataFrame(score_list)
        print(score_df.tail())

        # Save checkpoint
        hu.save_pkl(score_list_path, score_list)
        hu.torch_save(checkpoint_path, model.get_state_dict())
        print("Saved: %s" % savedir)

        if "accuracy" in exp_dict["target_loss"]:
            is_best = score_dict[exp_dict["target_loss"]] >= score_df[exp_dict["target_loss"]][:-1].max()
        else:
            is_best = score_dict[exp_dict["target_loss"]] <= score_df[exp_dict["target_loss"]][:-1].min()

            # Save best checkpoint
        if is_best:
            hu.save_pkl(os.path.join(savedir, "score_list_best.pkl"), score_list)
            hu.torch_save(os.path.join(savedir, "checkpoint_best.pth"), model.get_state_dict())
            print("Saved Best: %s" % savedir)

            # Check for end of training conditions
        if model.is_end_of_training():
            return

import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('cfg', type=str, help='json config path')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='model checkpoint you wanna resume from')

    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', default='data/')
    parser.add_argument('-nw', '--num_workers', default=2, type=int)
    parser.add_argument('-t', '--title', default=None, type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    with open(args.cfg) as f:
        cfg = json.load(f)

    trainval(exp_dict=cfg,
             savedir_base=args.savedir_base,
             reset=False,
             datadir=args.datadir,
             num_workers=args.num_workers,
             title = args.title,
             ckpt=args.ckpt)


