from model.PMN import IFA_MatchingNet
from util.utils import count_params, set_seed, mIOU
import imageio
import numpy as np
import argparse
import os
import torch
from torch.nn import DataParallel
from tqdm import tqdm
import glob
from dataset import FSSDataset

def parse_args():
    parser = argparse.ArgumentParser(description='IFA for CD-FSS')
    # basic arguments
    parser.add_argument('--data-root', type=str, default='../dataset', help='root path of training dataset')
    parser.add_argument('--dataset', type=str, default='deepglobe', choices=['fss', 'deepglobe', 'isic', 'lung'], help='training dataset')
    parser.add_argument('--backbone',  type=str, default='resnet50', choices=['resnet50', 'resnet101'], help='backbone of semantic segmentation model')
    parser.add_argument('--shot', type=int, default=1, help='number of support pairs')
    parser.add_argument('--seed', type=int, default=0, help='random seed to generate tesing samples')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size of training')

    args = parser.parse_args()
    return args


def evaluate(model, dataloader, args):
    tbar = tqdm(dataloader)

    if args.dataset == 'fss':
        num_classes = 1000
    elif args.dataset == 'deepglobe':
        num_classes = 6
    elif args.dataset == 'isic':
        num_classes = 3
    elif args.dataset == 'lung':
        num_classes = 1

    metric = mIOU(num_classes)

    save_path = 'prediction/models/%s' %(args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, (img_s_list, mask_s_list, img_q, mask_q, cls, _, id_q) in enumerate(tbar):

        img_s_list = img_s_list.permute(1,0,2,3,4)
        mask_s_list = mask_s_list.permute(1,0,2,3)
            
        img_s_list = img_s_list.numpy().tolist()
        mask_s_list = mask_s_list.numpy().tolist()

        for k in range(len(img_s_list)):
            img_s_list[k], mask_s_list[k] = torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k])
            # img_s_list[supp_local_proto], mask_s_list[supp_local_proto] = img_s_list[supp_local_proto].cuda(), mask_s_list[supp_local_proto].cuda()

        cls = cls[0].item()
        cls = cls + 1

        with torch.no_grad():
            final_out = model(img_s_list, mask_s_list, img_q, mask_q)
            pred = torch.argmax(final_out, dim=1)

            for i, output in enumerate(pred):
                output_image = output.cpu().numpy()
                imageio.imwrite(os.path.join(save_path, f'image_{i}.png'), (output_image*255).astype(np.uint8))

        pred[pred == 1] = cls
        mask_q[mask_q == 1] = cls

        metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())
        tbar.set_description("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))

    return metric.evaluate() * 100.0

def main():
    args = parse_args()
    print('\n' + str(args))

    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    testloader = FSSDataset.build_dataloader(args.dataset, args.batch_size, 4, '0', 'val', args.shot)

    model = IFA_MatchingNet(args.backbone, args.refine)

    ### Please modify the following paths with your model path if needed.
    if args.dataset == 'deepglobe':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/PMN/deepglobe/resnet50_1shot_avg_35.32.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/PMN/deepglobe/resnet50_5shot_avg_35.33.pth'
    if args.dataset == 'isic':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/PMN/isic/resnet50_1shot_avg_46.80.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/PMN/isic/resnet50_5shot_avg_56.17.pth'
    if args.dataset == 'lung':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/PMN/lung/resnet50_1shot_avg_71.67.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/PMN/lung/resnet50_5shot_avg_79.33.pth'
    if args.dataset == 'fss':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/PMN/fss/resnet50_1shot_avg_74.90.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/PMN/fss/resnet50_5shot_avg_77.12.pth'

    print('Evaluating model:', checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    print('\nParams: %.1fM' % count_params(model))

    best_model = DataParallel(model)

    print('\nEvaluating on 5 seeds.....')
    total_miou = 0.0
    model.eval()
    for seed in range(5):
        print('\nRun %i:' % (seed + 1))
        set_seed(args.seed + seed)

        miou = evaluate(best_model, testloader, args)
        total_miou += miou

    print('\n' + '*' * 32)
    print('Averaged mIOU on 5 seeds: %.2f' % (total_miou / 5))
    print('*' * 32 + '\n')

if __name__ == '__main__':
    main()