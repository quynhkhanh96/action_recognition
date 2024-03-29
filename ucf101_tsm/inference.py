import torch 
import argparse
from tqdm import tqdm 
import argparse

from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmcv import Config
from mmaction.models import build_recognizer
from mmcv.runner import load_checkpoint
from mmaction.core.evaluation import top_k_accuracy

def evaluate_video_recognizer(model, test_dataset, device):
    # TODO: validation loss 
    labels = []
    scores = []
    num_samples = len(test_dataset)
    with torch.no_grad():
        for i in tqdm(range(num_samples), total=num_samples):
            data = {'imgs': test_dataset[i]['imgs'][None].to(device)}
            score = model(return_loss=False, **data)[0]
            scores.append(score)
            labels.append(test_dataset[i]['label'])

    top1_acc, top5_acc = top_k_accuracy(scores, labels, topk=(1, 5))

    return {'top1': top1_acc, 'top5': top5_acc}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference and evaluate recognizer')
    # parser.add_argument(
    #     "--cfg_path",
    #     type=str,
    #     help="mmaction2 config path",
    # )
    parser.add_argument(
        "--epoch",
        type=int,
        help="epoch to run inference",
    )
    args = parser.parse_args()

    # cfg = Config.fromfile(args.cfg_path)
    cfg = Config.fromfile('configs/ucf101_rgb_tsm_k400_pretrained_r50_1x1x8.py')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model 
    cfg = Config.fromfile(args.cfg_path)
    ckpt_path = cfg.work_dir + f'/epoch_{args.epoch}.pth'
    cfg.model.backbone.pretrained = None
    model = build_recognizer(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt_path, map_location='cpu')
    model.cfg = cfg
    model.to(device)
    model.eval()

    # validation dataset
    val_dataset = build_dataset(cfg.data.val)

    # run inference and evaluate 
    metrics = evaluate_video_recognizer(model, val_dataset, device)
    print(metrics)

