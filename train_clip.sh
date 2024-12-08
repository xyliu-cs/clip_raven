# enter the src folder of the open_clip repository
cd /apdcephfs_qy3/share_301812049/xiaoyuanliu/Github/open_clip/src

# set the training args
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --batch-size 64 \
    --precision amp_bf16 \
    --workers 4 \
    --report-to wandb \
    --wandb-project-name "clip_raven" \
    --save-frequency 1 \
    --logs="./logs" \
    --dataset-type csv \
    --csv-separator="," \
    --train-data /apdcephfs_qy3/share_301812049/xiaoyuanliu/data/RAVEN/train_with_held_out_formatted.csv \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --warmup 1000 \
    --lr=5e-6 \
    --wd=0.1 \
    --epochs=32 \
    --model ViT-L-14 \
    --pretrained /apdcephfs_qy3/share_301812049/xiaoyuanliu/Github/clip_raven/ViT-L-14.pt

# args explanation
	#  --nproc_per_node 6    # On each server, 6 GPUs are used, corresponding to the number specified earlier.
	 
	#  --report-to tensorboard    # (Optional) Send training details to the corresponding TensorBoard file, but make sure to install the required packages beforehand.
	 
	#  --save-frequency 1    # save a checkpoint after each epoch
	 
	#  --logs="/models/clip/openclip_finetuning/logs"    # local path to store the training log and the checkpoints
	 
	#  --dataset-type csv    # （important！） specify the index file type
	 
	#  --csv-separator=","     # （important！） specify the csv separator of your csv file, OpenClip official uses the "Tab" key as a delimiter, but generally CSV files default to using "," as a delimiter. Remember to modify this delimiter, otherwise an error will occur!
	#  --train-data /path/to/your/local/training_dict.csv # your local path to training data CSV index file，validation data CSV index file is the same principle, and here I have omitted it.
	 
	#  --csv-img-key filepath 
    #  --csv-caption-key caption # （important！）make sure to modify these two values according to the headers in your custom CSV file. You can refer to the CSV demo I provided above for reference.
     
    #  --lr=5e-6 # （important！）the final learning rate should not be set too high, otherwise the training loss will oscillate severely. Experimental evidence has shown that e-6 is a good unit to use, but remember to adjust it according to your specific situation.
	 
	#  --pretrained #（important！）a pre-trained model type or /path/to/your/local/model
