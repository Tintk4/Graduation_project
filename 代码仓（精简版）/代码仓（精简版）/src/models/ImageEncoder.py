# import os
# import sys
# import collections
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
#
# from transformers import ViTFeatureExtractor, ViTModel
# from ..models.classifier import Classifier
# # from classifier import Classifier
# from transformers import AutoModelForImageClassification
#
# # vit base model from https://huggingface.co/google/vit-base-patch16-224
# # vit large model from https://huggingface.co/google/vit-large-patch16-224
#
# class ViTClf(nn.Module):
#     def __init__(self, args, image_encoder='vit_base'):
#         """
#         image_encoder: base / large
#         """
#         super(ViTClf, self).__init__()
#         assert image_encoder in ['vit_base', 'vit_large']
#
#         # directory is fine
#         if image_encoder in ['vit_base']:
#             # self.tokenizer = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
#             self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
#         else:
#             # self.tokenizer = ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-224")
#             self.image_encoder = ViTModel.from_pretrained("google/vit-large-patch16-224")
#
#         self.clf = Classifier(dropout=args.dropout, in_dim=768, post_dim=256, out_dim=args.n_classes)
#
#
#     def forward(self, pixel_values):
#         """
#         pixel_values:
#         """
#         # pixel_values = self.tokenizer(images=image, return_tensors="pt").pixel_values
#         x = self.image_encoder(pixel_values=pixel_values).last_hidden_state[:, 0, :]
#         out = self.clf(x)
#         return out
#
#
# # torchvision.models.vit_b_16
# class VitImageEncoder(nn.Module):
#     def __init__(self):
#         super(VitImageEncoder, self).__init__()
#
#         # self.model=torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
#         self.model=torchvision.models.vit_b_16(weights="IMAGENET1K_SWAG_E2E_V1")
#         # self.model=torchvision.models.vit_l_16(weights="IMAGENET1K_SWAG_LINEAR_V1")
#         # self.model=torchvision.models.vit_h_14(weights="IMAGENET1K_SWAG_LINEAR_V1")
#
#         # self.model=torch.hub.load("pytorch/vision", "vit_b_16", weights="IMAGENET1K_V1")
#         # self.model=torch.hub.load("pytorch/vision", "vit_b_16", weights="IMAGENET1K_SWAG_E2E_V1")
#         # self.model=torch.hub.load("pytorch/vision", "vit_l_16", weights="IMAGENET1K_SWAG_LINEAR_V1")
#         # self.model=torch.hub.load("pytorch/vision", "vit_h_14", weights="IMAGENET1K_SWAG_LINEAR_V1")
#
#     def forward(self, x):
#         x = self.model._process_input(x)
#         n = x.shape[0]
#
#         # Expand the class token to the full batch
#         batch_class_token = self.model.class_token.expand(n, -1, -1)
#         x = torch.cat([batch_class_token, x], dim=1)
#
#         x = self.model.encoder(x)
#         out = x[:, 0]
#         return out
#
#
# class torchViTClf(nn.Module):
#     def __init__(self, args):
#         """
#         image_encoder: base / large
#         """
#         super(torchViTClf, self).__init__()
#         self.image_encoder = VitImageEncoder()
#         in_features=self.image_encoder.model.heads.head.in_features
#         self.clf = Classifier(dropout=args.dropout, in_dim=in_features, post_dim=256, out_dim=args.n_classes)
#
#     def forward(self, x):
#         """
#         pixel_values:
#         """
#         x = self.image_encoder(x)
#         out = self.clf(x)
#         return out
#
#
# if __name__ == "__main__":
#     x=torch.randn(1,3,224,224)
#     vit_normal = VitImageEncoder()
#     a=vit_normal(x)
#
#

import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from transformers import ViTFeatureExtractor, ViTModel
from ..models.classifier import Classifier

# 设置模型缓存路径
os.environ['TORCH_HOME'] = 'D:/models/torch_hub'
os.environ['TRANSFORMERS_CACHE'] = 'D:/models/transformers'


class ViTClf(nn.Module):
    def __init__(self, args, image_encoder='vit_base'):
        super(ViTClf, self).__init__()
        assert image_encoder in ['vit_base', 'vit_large']

        # 模型配置
        model_configs = {
            'vit_base': {
                'local_path': "D:/models/vit-base-patch16-224",
                'online_name': "google/vit-base-patch16-224"
            },
            'vit_large': {
                'local_path': "D:/models/vit-large-patch16-224",
                'online_name': "google/vit-large-patch16-224"
            }
        }

        config = model_configs[image_encoder]

        # 优先使用本地模型
        if os.path.exists(config['local_path']):
            print(f"使用本地ViT模型: {config['local_path']}")
            try:
                self.image_encoder = ViTModel.from_pretrained(
                    config['local_path'],
                    local_files_only=True
                )
                print("✓ 本地ViT模型加载成功")
            except Exception as e:
                print(f"本地模型加载失败: {e}")
                print(f"回退到在线加载: {config['online_name']}")
                self.image_encoder = ViTModel.from_pretrained(config['online_name'])
        else:
            print(f"本地模型不存在，在线加载: {config['online_name']}")
            self.image_encoder = ViTModel.from_pretrained(config['online_name'])

        self.clf = Classifier(dropout=args.dropout, in_dim=768, post_dim=256, out_dim=args.n_classes)

    def forward(self, pixel_values):
        x = self.image_encoder(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        return self.clf(x)


class VitImageEncoder(nn.Module):
    def __init__(self):
        super(VitImageEncoder, self).__init__()

        # 本地模型路径
        local_path = "D:/models/torchvision/vit_b_16.pth"

        try:
            # 创建模型结构
            self.model = torchvision.models.vit_b_16(weights=None)

            # 尝试加载本地权重
            if os.path.exists(local_path):
                print(f"加载本地TorchVision ViT权重: {local_path}")
                state_dict = torch.load(local_path)
                self.model.load_state_dict(state_dict)
                print("✓ 本地权重加载成功")
            else:
                print("本地权重不存在，使用在线权重")
                # 这会触发下载，但会保存到TORCH_HOME指定的目录
                model_with_weights = torchvision.models.vit_b_16(weights="IMAGENET1K_SWAG_E2E_V1")
                self.model.load_state_dict(model_with_weights.state_dict())

        except Exception as e:
            print(f"ViT模型加载失败: {e}")
            # 使用无预训练权重的模型
            self.model = torchvision.models.vit_b_16(weights=None)
            print("使用无预训练权重的ViT模型")

    def forward(self, x):
        x = self.model._process_input(x)
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.model.encoder(x)
        return x[:, 0]


class torchViTClf(nn.Module):
    def __init__(self, args):
        super(torchViTClf, self).__init__()
        self.image_encoder = VitImageEncoder()
        in_features = self.image_encoder.model.heads.head.in_features
        self.clf = Classifier(dropout=args.dropout, in_dim=in_features, post_dim=256, out_dim=args.n_classes)

    def forward(self, x):
        x = self.image_encoder(x)
        return self.clf(x)


if __name__ == "__main__":
    # 测试代码
    x = torch.randn(1, 3, 224, 224)
    vit_normal = VitImageEncoder()
    a = vit_normal(x)
    print(f"输出形状: {a.shape}")
