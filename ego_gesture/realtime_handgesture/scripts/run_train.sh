#"$1" root_path 
#"$2" pretrain path
#"$3" model
#"$4" width_mult

python main.py --root_path "$1" \
	--video_path "$1"/images \
	--annotation_path "$1"/annotation_EgoGesture/egogestureall_but_None.json \
	--result_path "$1"/cent_exps \
	--pretrain_path "$2" \
	--dataset egogesture \
	--n_classes 27 \
	--n_finetune_classes 83 \
	--model "$3" \
	--width_mult "$4" \
	--model_depth 101 \
	--resnet_shortcut B \
	--resnext_cardinality 16 \
	--train_crop random \
	--learning_rate 0.01 \
	--sample_duration 32 \
	--modality Depth \
	--pretrain_modality RGB \
	--downsample 1 \
	--batch_size 24 \
	--n_threads 16 \
	--checkpoint 1 \
	--n_val_samples 1 \
    --n_epochs 60 \
    --ft_portion complete \