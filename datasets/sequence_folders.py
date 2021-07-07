import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import os


def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, skip_frames=1, dataset='kitti'):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)   # 경로 접근
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'   # Training 및 test set 내 sequences 확인
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]   # Sequence 폴더 접근하기
        self.transform = transform   # 데이터 전처리
        self.dataset = dataset   # 데이터셋 이름
        self.k = skip_frames   # Frames 건너뛰기 (k=1이면 이웃한 frames 활용)
        self.crawl_folders(sequence_length)   # 데이터셋 생성

    def crawl_folders(self, sequence_length):   # 데이터셋 생성 (Dict : {target, reference, intrinsics})
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2   # 추출할 frames 수의 절반 (sequence_length=3일 경우, demi_length=1)
                                               # Target frame을 중심으로 대칭으로 frames을 추출하기 위함
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))   # 추출할 frames index 생성 (demi_length=1일 경우, [-1, 0, 1])
        shifts.pop(demi_length)   # 중간 (기준) index 제거 (demi_length=1일 경우, [-1, 1]) / 이웃한 frames index 의미

        for scene in self.scenes:   # Sequence 폴더 차례대로 접근
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))   # cam.txt 파일로부터 intrinsics matrix 가져오기
            imgs = sorted(scene.files('*.jpg'))

            if len(imgs) < sequence_length:   # Frames 최소 3개 이상이어야 함
                continue
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):   # Target frame, reference image, intrinsics 가져오기
                                                                                    # demi_length=1일 경우, 양 끝 frames는 target frame으로 사용 안 함
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}   # Target frame, reference image, intrinsics 딕셔너리 선언
                for j in shifts:   # 이웃한 두 frames 가져오기
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)   # 데이터 셔플하기
        self.samples = sequence_set

    def __getitem__(self, index):   # 데이터 가져오기
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)   # intrinsics_inv는 intrinsinc의 역행렬로 데이터 불러오면서 생성

    def __len__(self):
        return len(self.samples)
