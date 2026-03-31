# #!/usr/bin/env python3
# #
# # Copyright (c) Facebook, Inc. and its affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.
# #
#
# import json
# import numpy as np
# import os
# from PIL import Image
#
# import torch
# from torch.utils.data import Dataset
#
# from ..utils.utils import truncate_seq_pair, numpy_seed
#
# import random
# class JsonlDataset(Dataset):
#     def __init__(self, data_path, tokenizer, transforms, vocab, args):
#         self.data = [json.loads(l) for l in open(data_path)]
#         self.data_dir = os.path.dirname(data_path)
#         self.tokenizer = tokenizer
#         self.args = args
#         self.vocab = vocab
#         self.n_classes = len(args.labels)
#         self.text_start_token = ["[CLS]"]
#
#         with numpy_seed(0):
#             for row in self.data:
#                 if np.random.random() < args.drop_img_percent:
#                     row["img"] = None
#
#         self.max_seq_len = args.max_seq_len
#         self.transforms = transforms
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         _ = self.tokenizer(self.data[index]["text"])
#         sentence = (
#             self.text_start_token
#             + _[:(self.args.max_seq_len - 1)]
#         )
#         segment = torch.zeros(len(sentence))
#         sentence = torch.LongTensor(
#             [
#                 self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
#                 for w in sentence
#             ]
#         )
#         if self.args.task_type == "multilabel":
#             label = torch.zeros(self.n_classes)
#             label[
#                 [self.args.labels.index(tgt) for tgt in self.data[index]["label"]]
#             ] = 1
#         else:
#             label = torch.LongTensor(
#                 [self.args.labels.index(self.data[index]["label"])]
#             )
#         if self.data[index]["img"]:
#             image = Image.open(
#                 os.path.join(self.data_dir, self.data[index]["img"])
#             ).convert("RGB")
#         else:
#             image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
#         image = self.transforms(image)
#         return sentence, segment, image, label,torch.LongTensor([index])
#
# class AddGaussianNoise(object):
#
#     '''
#     mean:均值
#     variance：方差
#     amplitude：幅值
#     '''
#     def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
#
#         self.mean = mean
#         self.variance = variance
#         self.amplitude = amplitude
#
#     def __call__(self, img):
#
#         img = np.array(img)
#         h, w, c = img.shape
#         np.random.seed(0)
#         N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
#         N = np.repeat(N, c, axis=2)
#         img = N + img
#         img[img > 255] = 255                       # 避免有值超过255而反转
#         img = Image.fromarray(img.astype('uint8')).convert('RGB')
#         return img
#
# class AddSaltPepperNoise(object):
#
#     def __init__(self, density=0,p=0.5):
#         self.density = density
#         self.p = p
#
#     def __call__(self, img):
#         if random.uniform(0, 1) < self.p:  # 概率的判断
#             img = np.array(img)  # 图片转numpy
#             h, w, c = img.shape
#             Nd = self.density
#             Sd = 1 - Nd
#             mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
#             mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
#             img[mask == 0] = 0  # 椒
#             img[mask == 1] = 255  # 盐
#             img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
#             return img
#         else:
#             return img

# !/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from ..utils.utils import truncate_seq_pair, numpy_seed

import random


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _ = self.tokenizer(self.data[index]["text"])
        sentence = (
                self.text_start_token
                + _[:(self.args.max_seq_len - 1)]
        )
        segment = torch.zeros(len(sentence))
        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        # 修复标签处理逻辑
        raw_label = self.data[index]["label"]

        if self.args.task_type == "multilabel":
            # 多标签处理 - 确保label是列表
            if isinstance(raw_label, str):
                # 如果是字符串，检查是否是逗号分隔的多标签
                if ',' in raw_label:
                    labels = [l.strip() for l in raw_label.split(',')]
                else:
                    # 单标签情况，包装成列表
                    labels = [raw_label]
            elif isinstance(raw_label, list):
                labels = raw_label
            else:
                labels = [str(raw_label)]

            label = torch.zeros(self.n_classes)
            try:
                label_indices = [self.args.labels.index(tgt) for tgt in labels]
                label[label_indices] = 1
            except ValueError as e:
                print(f"错误: 无效标签 {labels}，可用标签: {self.args.labels}")
                raise e
        else:
            # 单标签处理
            if isinstance(raw_label, list):
                # 如果是列表，取第一个元素
                if raw_label:
                    raw_label = raw_label[0]
                else:
                    raise ValueError("标签列表为空")

            try:
                label = torch.LongTensor([self.args.labels.index(raw_label)])
            except ValueError as e:
                print(f"错误: 无效标签 '{raw_label}'，可用标签: {self.args.labels}")
                raise e

        # if self.data[index]["img"]:
        #     image = Image.open(
        #         os.path.join(self.data_dir, self.data[index]["img"])
        #     ).convert("RGB")
        # else:
        #     image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))

        if self.data[index]["img"]:
            try:
                # 获取图片路径并规范化
                img_path = self.data[index]["img"]
                # 统一路径分隔符
                img_path = img_path.replace('/', os.sep).replace('\\', os.sep)

                # 构建完整路径
                full_img_path = os.path.join(self.data_dir, img_path)

                # 检查文件是否存在
                if os.path.exists(full_img_path):
                    image = Image.open(full_img_path).convert("RGB")
                else:
                    # 文件不存在，创建默认图像
                    print(f"警告: 图片文件不存在: {full_img_path}")
                    image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            except Exception as e:
                print(f"错误: 无法加载图片: {e}")
                # 创建默认图像
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
        else:
            # 没有图片路径，创建默认图像
            image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))

        image = self.transforms(image)
        return sentence, segment, image, label, torch.LongTensor([index])


# 其他类保持不变...
class AddGaussianNoise(object):
    '''
    mean:均值
    variance：方差
    amplitude：幅值
    '''

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        np.random.seed(0)
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255  # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img


class AddSaltPepperNoise(object):

    def __init__(self, density=0, p=0.5):
        self.density = density
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:  # 概率的判断
            img = np.array(img)  # 图片转numpy
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
            mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
            img[mask == 0] = 0  # 椒
            img[mask == 1] = 255  # 盐
            img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
            return img
        else:
            return img