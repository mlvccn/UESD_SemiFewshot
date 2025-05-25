python trainit.py cfgs/finetune_miniimagenet_wrn_1.json \
        --ckpt save/pretrain_miniimagenet_wrn/checkpoint_best.pth \
        -sb logs/finetune \
        -t miniimagenet_wrn_1 \
        -d /data16t