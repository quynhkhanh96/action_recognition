#!/bin/bash
dataset="afosr" 
hostname=$(hostname)
echo "[bash] Running on $hostname"

case "$hostname" in
	"X9DRL-3F-iF")
		data="/home/nhquan/datasets/afosr2022/data"
		trlist="/home/nhquan/datasets/afosr2022/train.txt"
		telist="/home/nhquan/datasets/afosr2022/val.txt"    
    	cd /mnt/works/projectComvis/AFOSR-2020/video_classification
		source ../multi_stream_videonet/.env/bin/activate
    	;;
    "Server2")
    	data=/mnt/disk3/datasets/afors2022/data
    	trlist=/mnt/disk3/datasets/afors2022/train.txt
    	telist=/mnt/disk3/datasets/afors2022/val.txt
    	export CUDA_VISIBLE_DEVICES="0"
    	cd /mnt/disk3/users/nhquan/projectComvis/AFOSR-2020/video_classification
		source ~/pytorch_env/bin/activate
    	;;
  	"hungvuong")
    	data=/ext_data2/comvis/datasets/afors2022/data
    	trlist=/ext_data2/comvis/datasets/afors2022/train.txt
    	telist=/ext_data2/comvis/datasets/afors2022/val.txt
    	export CUDA_VISIBLE_DEVICES="0"
    	;;
	*)
		echo other
 		;;
esac
 
model="movinet_a0" 	# resnet183d resnet503d c3d_bn movinet_a0 movinet_a2 efficientnet3d mobilenet3d_v2 mobilenet3d_v2_0.45x
widthMult=0.45 # using with mobilenet
saveDir="./log/$dataset/$model"		
echo $saveDir

if [ "$model" = "c3d_bn" ]; then
	python3 main.py --job train -a $model --dataset-dir $data --train-annotation-file $trlist \
	--test-annotation-file $telist --height 112 --width 112 --dataset $dataset --train-batch 4 \
	--max-epoch 30 --eval-step 10 --print-freq 5 --save-dir $saveDir --seq-len 16 \
	--pretrained-model=./pre-trained/c3d-bn_kinetics.pth  	
elif [ "$model" = "c3d" ]; then
	python3 main.py --job train -a $model --dataset-dir $data --train-annotation-file $trlist \
	--test-annotation-file $telist --height 112 --width 112 --dataset $dataset --train-batch 4 \
	--max-epoch 30 --eval-step 10 --print-freq 5 --save-dir $saveDir --seq-len 16 \
	--pretrained-model=./pre-trained/c3d_sports1m.pth		
elif [ "$model" = "resnet503d" ]; then
	python3 main.py --job train -a $model --dataset-dir $data --train-annotation-file $trlist \
	--test-annotation-file $telist --height 224 --width 224 --dataset $dataset --train-batch 4 \
	--max-epoch 30 --eval-step 10 --print-freq 5 --save-dir $saveDir --seq-len 16 \
	--pretrained-model=./pre-trained/resnet-50-kinetics.pth 
elif [ "$model" = "resnet183d" ]; then
	python3 main.py --job train -a $model --dataset-dir $data --train-annotation-file $trlist \
	--test-annotation-file $telist --height 224 --width 224 --dataset $dataset --train-batch 4 \
	--max-epoch 30 --eval-step 10 --print-freq 5 --save-dir $saveDir --seq-len 16 \
	--pretrained-model=./pre-trained/resnet-18-kinetics.pth 
elif [ "${model:0:7}" = "movinet" ]; then
	python3 main.py --job train -a $model --dataset-dir $data --train-annotation-file $trlist \
	--test-annotation-file $telist --height 172 --width 172 --dataset $dataset --train-batch 4 \
	--max-epoch 30 --eval-step 10 --print-freq 5 --save-dir $saveDir --seq-len 16
elif [ "$model" = "efficientnet3d" ]; then
	python3 main.py --job train -a $model --dataset-dir $data --train-annotation-file $trlist \
	--test-annotation-file $telist --height 224 --width 224 --dataset $dataset --train-batch 4 \
	--max-epoch 30 --eval-step 10 --print-freq 5 --save-dir $saveDir --seq-len 16
elif [ "${model:0:14}" = "mobilenet3d_v2" ]; then
	python3 main.py --job train -a $model --dataset-dir $data --train-annotation-file $trlist \
	--test-annotation-file $telist --height 112 --width 112 --dataset $dataset --train-batch 16 \
	--max-epoch 30 --eval-step 10 --print-freq 5 --save-dir $saveDir --seq-len 16 \
	--width-mult $widthMult --pretrained-model=./pre-trained/kinetics_mobilenetv2_0.45x_RGB_16_best.pth
else	
	echo "Wrong parameter --> code cound not  be runed"
fi