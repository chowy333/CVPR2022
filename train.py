import argparse
import time
import csv
import datetime
from path import Path
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import sys
import models

import custom_transforms
from utils import tensor2array, save_checkpoint
from datasets.sequence_folders import SequenceFolder
from datasets.pair_folders import PairFolder
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--folder-type', type=str, choices=['sequence', 'pair'], default='sequence', help='the dataset dype to train')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')   # Epoch : 200
parser.add_argument('--epoch-size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')   # Learning rate : 1e-4
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs at validation step')
parser.add_argument('--resnet-layers',  type=int, default=18, choices=[18, 50], help='number of ResNet layers for depth estimation.')   # ResNet18
parser.add_argument('--num-scales', '--number-of-scales', type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)   # Photometric loss weight : 1
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)   # Smooth loss weight : 0.1
parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss', metavar='W', default=0.5)   # Geometry consistency loss weight : 0.5
parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
parser.add_argument('--with-mask', type=int, default=1, help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int,  default=0, help='with the the mask for stationary points')
parser.add_argument('--with-pretrain', type=int,  default=1, help='with or without imagenet pretrain for resnet')
parser.add_argument('--dataset', type=str, choices=['kitti', 'nyu'], default='kitti', help='the dataset to train')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='path to pre-trained Posenet model')
parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')

def main():

    ################## parser ########################
    args = parser.parse_args()

    ################## for reproducibility ########################
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    ################## Loading the network ########################
    # create model
    print("=> creating model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)

    #print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1:   # DispNet / PoseNet 모델 받아오기
        disp_net = torch.nn.DataParallel(models.DispResNet(args.resnet_layers, args.with_pretrain)).to(device)   # DispResNet(18, 1)
        pose_net = torch.nn.DataParallel(models.PoseResNet(18, args.with_pretrain)).to(device)   # PoseResNet(18, 1)
    else:
        disp_net = models.DispResNet(args.resnet_layers, args.with_pretrain).to(device)   # DispResNet(18, 1)
        pose_net = models.PoseResNet(18, args.with_pretrain).to(device)   # PoseResNet(18, 1)


    ################## Loading checkpoints ########################
    if args.pretrained_disp:   # DispNet pre-trained weights 받아오기
        print("=> using pre-trained weights for DispResNet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'], strict=False)

    if args.pretrained_pose:   # PoseNet pre-trained weights 받아오기
        print("=> using pre-trained weights for PoseResNet")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()


    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],   # 데이터 전처리 (Nomalization)
                                            std=[0.225, 0.225, 0.225])

    train_transform = custom_transforms.Compose([   # Training 데이터 전처리
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])   # Test 데이터 전처리

    print("=> fetching scenes in '{}'".format(args.data))   # Training set 폴더 접근하기
    if args.folder_type == 'sequence':
        train_set = SequenceFolder(
            args.data,
            transform=train_transform,
            seed=args.seed,
            train=True,
            sequence_length=args.sequence_length,
            dataset=args.dataset
        )
    else:
        train_set = PairFolder(
            args.data,
            seed=args.seed,
            train=True,
            transform=train_transform
        )


    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    if args.with_gt:   # Test set 폴더 접근하기
        from datasets.validation_folders import ValidationSet
        val_set = ValidationSet(
            args.data,
            transform=valid_transform,
            dataset=args.dataset
        )
    else:
        val_set = SequenceFolder(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            sequence_length=args.sequence_length,
            dataset=args.dataset
        )
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(   # Training set 불러오기
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(   # Test set 불러오기
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:   # Epoch size 조정 (별도로 설정 안할 시, 전체 training set = 1 epoch)
        args.epoch_size = len(train_loader)


    print('=> setting adam solver')   # Optimizer 설정하기 (Adam)
    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    with open(args.save_path/args.log_summary, 'w') as csvfile:   # Log 파일 생성
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])
    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'smooth_loss', 'geometry_consistency_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))   # Log 기록 시작
    logger.epoch_bar.start()

    ############################ train start!!! ############################
    for epoch in range(args.epochs):   # Training
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()

        n_iter = 0   # Training 지표 클래스 선언
        best_error = -1
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(precision=4)

        end = time.time()
        logger.train_bar.update(0)

        for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):   # Training data 읽어오기
            log_losses = i > 0 and n_iter % args.print_freq == 0   # Log 출력 여부 결정

            disp_net.train()   # Training 모드 전환
            pose_net.train()

            # measure data loading time
            data_time.update(time.time() - end)   # Training data 자료형 변환
            tgt_img = tgt_img.to(device)   # Shape : [4, 3, 256, 832] / [Batch, RGB, H, W]
            ref_imgs = [img.to(device) for img in ref_imgs]    # Shape : [2, 4, 3, 256, 832] / [Pair, Batch, RGB, H, W]
                                                               # ref_imgs의 두 frames는 tgt_img의 이웃한 두 frames임
            intrinsics = intrinsics.to(device)   # Shape : [4, 3, 3] / [Batch, 3x3 intrinsic-parameters matrix]
                                                 # ex)
                                                 # [[527.7861,   0.0000, 398.8493]
                                                 #  [  0.0000, 527.1625, 123.5155]
                                                 #  [  0.0000,   0.0000,   1.0000]]
                                                 # intrinsics_inv도 intrinsics와 shape가 동일함 (np.linarg.inv(intrinsics))

            # # compute output
            # print("disp_net, tgt_img, ref_imgs", tgt_img.shape, len(ref_imgs))
            # tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
            # k = disp_net(tgt_img).to(device) # 여기서 계속
            # print("tgt_img =", tgt_img)
            # print("output", disp_net(tgt_img))

            tgt_depth = [1/disp for disp in disp_net(tgt_img)]   # DispNet으로 target 이미지 depth map 추출
                                                                 # 각 pixel 값 분포 : 약 0.18 ~ 0.22

            ref_depths = []   # DispNet으로 reference (이웃) 이미지 depth map 추출
            for ref_img in ref_imgs:
                ref_depth = [1/disp for disp in disp_net(ref_img)]
                ref_depths.append(ref_depth)

            # loss
            poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)   # Pose 계산
                                                                                    # Shape : [4, 6] / [Batch, 6-DoF poses]
                                                                                    # poses_inv도 poses와 shape가 동일함

            loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,   # Photometric loss 및 geometry consistency loss 계산 / Mask 적용
                                                             poses, poses_inv, args.num_scales, args.with_ssim,
                                                             args.with_mask, args.with_auto_mask, args.padding_mode)

            loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)   # Smooth loss 계산
            w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight
            loss = w1*loss_1 + w2*loss_2 + w3*loss_3   # Loss 합치기 (w1 : 1, w2 : 0.1, w3 : 0.5)



            if log_losses:   # Log 기록
                training_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
                training_writer.add_scalar('disparity_smoothness_loss', loss_2.item(), n_iter)
                training_writer.add_scalar('geometry_consistency_loss', loss_3.item(), n_iter)
                training_writer.add_scalar('total_loss', loss.item(), n_iter)

            # record loss and EPE
            losses.update(loss.item(), args.batch_size)   # Loss 업데이트

            # compute gradient and do Adam step
            optimizer.zero_grad()   # Gradient 계산 후, parameters 값 업데이트
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            with open(args.save_path/args.log_full, 'a') as csvfile:   # Log 파일에 기록
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
            logger.train_bar.update(i+1)
            if i % args.print_freq == 0:
                logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
            if i >= args.epoch_size - 1:
                break

            n_iter += 1

        train_loss = losses.avg[0]   # Loss 평균 계산


        #train_loss = train(args, train_loader, disp_net, pose_net, optimizer, args.epoch_size, logger, training_writer, device)

        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))   # Log에 평균 loss 기록
        ####################### 1 Epoch training 종료 #######################

        # evaluate on validation set
        logger.reset_valid_bar()

        if args.with_gt:   # Test 전 ground-truth 존재 여부에 따른 평가 방식 결정
            errors, error_names = validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers, device)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers, device)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        sys.exit(0)
        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        # Up to you to choose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[1]   # 평가 지표 선택
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error   # 해당 epoch에서 최고 성능인지 확인
        best_error = min(best_error, decisive_error)   # 평가 지표 값 저장
        save_checkpoint(   # Weights 저장
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': disp_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': pose_net.module.state_dict()
            },
            is_best)

        with open(args.save_path/args.log_summary, 'a') as csvfile:   # Log 파일에 기록
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


####################### 통합되어 사용 안함 #######################
# def train(args, train_loader, disp_net, pose_net, optimizer, epoch_size, logger, train_writer, device):
#
#     n_iter = 0
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter(precision=4)
#     w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight
#
#     # switch to train mode
#     disp_net.train()
#     pose_net.train()
#
#     end = time.time()
#     logger.train_bar.update(0)
#
#     for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
#         log_losses = i > 0 and n_iter % args.print_freq == 0
#
#         # measure data loading time
#         data_time.update(time.time() - end)
#         tgt_img = tgt_img.to(device)
#         ref_imgs = [img.to(device) for img in ref_imgs]
#         intrinsics = intrinsics.to(device)
#
#         disp_net = disp_net.to(device)
#         # compute output
#         #print("disp_net, tgt_img, ref_imgs", tgt_img.shape, len(ref_imgs))
#         #tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
#         #k = disp_net(tgt_img).to(device) # 여기서 계속
#         print("tgt_img =", tgt_img)
#         print("output", disp_net(tgt_img))
#         tgt_depth = [1/disp for disp in disp_net(tgt_img)]
#
#         ref_depths = []
#         for ref_img in ref_imgs:
#             ref_depth = [1/disp for disp in disp_net(ref_img)]
#             ref_depths.append(ref_depth)
#
#
#         poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
#
#         loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
#                                                          poses, poses_inv, args.num_scales, args.with_ssim,
#                                                          args.with_mask, args.with_auto_mask, args.padding_mode)
#
#         loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)
#
#         loss = w1*loss_1 + w2*loss_2 + w3*loss_3
#
#         if log_losses:
#             train_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
#             train_writer.add_scalar('disparity_smoothness_loss', loss_2.item(), n_iter)
#             train_writer.add_scalar('geometry_consistency_loss', loss_3.item(), n_iter)
#             train_writer.add_scalar('total_loss', loss.item(), n_iter)
#
#         # record loss and EPE
#         losses.update(loss.item(), args.batch_size)
#
#         # compute gradient and do Adam step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         with open(args.save_path/args.log_full, 'a') as csvfile:
#             writer = csv.writer(csvfile, delimiter='\t')
#             writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
#         logger.train_bar.update(i+1)
#         if i % args.print_freq == 0:
#             logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
#         if i >= epoch_size - 1:
#             break
#
#         n_iter += 1
#
#     return losses.avg[0]


@torch.no_grad()   # Gradient 계산 안함 (test 모드)
def validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers=[]):   # Ground-truth (depth map)가 없는 경우의 평가

    batch_time = AverageMeter()   # Test 지표 클래스 선언
    losses = AverageMeter(i=4, precision=4)
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()   # Test 모드 전환
    pose_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):   # Test data 읽어오기 / Loss만 구할 수 있음
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        # compute output
        tgt_depth = [1 / disp_net(tgt_img)]   # DispNet으로 target 이미지 depth map 추출
        ref_depths = []   # DispNet으로 reference 이미지 depth map 추출
        for ref_img in ref_imgs:
            ref_depth = [1 / disp_net(ref_img)]
            ref_depths.append(ref_depth)

        if log_outputs and i < len(output_writers):   # Visualization
            if epoch == 0:
                output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)

            output_writers[i].add_image('val Dispnet Output Normalized',
                                        tensor2array(1/tgt_depth[0][0], max_value=None, colormap='magma'),
                                        epoch)
            output_writers[i].add_image('val Depth Output',
                                        tensor2array(tgt_depth[0][0], max_value=10),
                                        epoch)

        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)   # Pose 계산

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,   # Photometric loss 및 geometry consistency loss 계산 / Mask 적용
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)   # Smooth loss 계산

        loss_1 = loss_1.item()   # 각 loss 값 추출
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()

        loss = loss_1
        losses.update([loss, loss_1, loss_2, loss_3])   # 각 loss 값 업데이트

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:   # Log 기록
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    logger.valid_bar.update(len(val_loader))   # Test set 길이만큼 평가 완료 알림
    return losses.avg, ['Total loss', 'Photo loss', 'Smooth loss', 'Consistency loss']   # 평균 loss 및 각 loss 값 반환


@torch.no_grad()
def validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers=[]):   # Ground-truth (depth map)가 있는 경우의 평가 / Depth estiamtion accuracy

    batch_time = AverageMeter()   # Test 지표 클래스 선언
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']   # 평가 지표 선택
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()   # Test 모드 전환

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, depth) in enumerate(val_loader):   # Test data 읽어오기
        tgt_img = tgt_img.to(device)
        depth = depth.to(device)

        # check gt
        if depth.nelement() == 0:   # Ground-truth의 전체 element 수 확인
            continue

        # compute output
        output_disp = disp_net(tgt_img)   # DispNet으로 target 이미지 depth map 추출
        output_depth = 1/output_disp[:, 0]   # Inverse된 depth map 복구

        if log_outputs and i < len(output_writers):   # Visualization
            if epoch == 0:
                output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)
                depth_to_show = depth[0]
                output_writers[i].add_image('val target Depth',
                                            tensor2array(depth_to_show, max_value=10),
                                            epoch)
                depth_to_show[depth_to_show == 0] = 1000
                disp_to_show = (1/depth_to_show).clamp(0, 10)
                output_writers[i].add_image('val target Disparity Normalized',
                                            tensor2array(disp_to_show, max_value=None, colormap='magma'),
                                            epoch)

            output_writers[i].add_image('val Dispnet Output Normalized',
                                        tensor2array(output_disp[0], max_value=None, colormap='magma'),
                                        epoch)
            output_writers[i].add_image('val Depth Output',
                                        tensor2array(output_depth[0], max_value=10),
                                        epoch)

        if depth.nelement() != output_depth.nelement():   # Target과 ground-truth의 전체 element 수가 다를 경우
            b, h, w = depth.size()
            output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [h, w]).squeeze(1)   # Interpolation하여 같게 맞춰줌

        errors.update(compute_errors(depth, output_depth, args.dataset))   # 평가 지표 계산 및 업데이트

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:   # Log 기록
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
    logger.valid_bar.update(len(val_loader))   # Test set 길이만큼 평가 완료 알림
    return errors.avg, error_names   # 평균 평가 지표 및 각 loss 값 반환


def compute_depth(disp_net, tgt_img, ref_imgs):   # Depth map 예측
    tgt_depth = [1/disp for disp in disp_net(tgt_img)]   # Target 이미지 inverse하여 입력

    ref_depths = []   # Reference 이미지 inverse하여 입력
    for ref_img in ref_imgs:
        ref_depth = [1/disp for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths   # 예측한 depth map 반환


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):   # Pose 예측
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:   # Pose 및 inverse pose 예측
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))   # poses_inv는 frames를 거꾸로 입력

    return poses, poses_inv   # Pose 및 inverse pose 반환


if __name__ == '__main__':
    main()
