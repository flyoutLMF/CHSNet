# train VGG16Trans model on FSC
python train.py --tag cl1 --no-wandb --device 1 --scheduler step --step 400 --dcsize 8 --batch-size 4 --lr 4e-5 --val-start 100 --val-epoch 10
#python train.py --tag cl_nonlocal_attention_point --device 1 --batch-size 8 --resume ./50_ckpt.tar