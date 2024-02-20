from itertools import combinations
import torch 
import torch.nn as nn
import torch.nn.functional as F
from tvl_enc.tvl import ModalityType
from timm.utils import accuracy

def construct_top_k_mask(affinity_matrix, k=5):
    topk_mat = torch.topk(affinity_matrix, k=k, dim=1)[1]
    topk_bool = torch.zeros_like(affinity_matrix, dtype=torch.bool)
    topk_bool.scatter_(1, topk_mat, True)
    return topk_bool

class TVLLoss(nn.Module):
    def __init__(
        self,
        active_modalities=[ModalityType.VISION, ModalityType.TACTILE, ModalityType.TEXT],
        lang_am_temp=0.05, 
        use_tac_text_loss=False,
        similarity_thres=0.9,
        disable_vision_text_loss=False,
        disable_tactile_text_loss=False,
    ):
        # this loss doesn't work with world size > 1 
        super(TVLLoss, self).__init__()
        self.active_modalities = active_modalities 
        self.lang_am_temp = lang_am_temp
        self.use_tac_text_loss = use_tac_text_loss
        self.similarity_thres = similarity_thres
        self.disable_vision_text_loss = disable_vision_text_loss
        self.disable_tactile_text_loss = disable_tactile_text_loss # this mirrors the decision in imagebind

    def tactile_text_loss(self, tactile_features, text_features, logit_scale):
        text_matrix = text_features @ text_features.T
        text_matrix = text_matrix / self.lang_am_temp
        row_target = F.softmax(text_matrix, dim=0)
        col_target = F.softmax(text_matrix, dim=1)
        
        affinity_matrix = logit_scale * tactile_features @ text_features.T
        row_pred = F.softmax(affinity_matrix, dim=0)
        col_pred = F.softmax(affinity_matrix, dim=1)

        # cross entropy on the rows and columns 
        row_loss = torch.mean(torch.sum(-row_target * torch.log(row_pred), dim=1))
        col_loss = torch.mean(torch.sum(-col_target * torch.log(col_pred), dim=1))
        return (row_loss + col_loss) / 2, affinity_matrix

    def clip_loss(self, class_a_feat, class_b_feat, logit_scale):
        labels = torch.arange(class_a_feat.shape[0], device=class_a_feat.device, dtype=torch.long)
        affinity_matrix = logit_scale * class_a_feat @ class_b_feat.T
        row_loss = F.cross_entropy(affinity_matrix, labels)
        col_loss = F.cross_entropy(affinity_matrix.T, labels)
        return (row_loss + col_loss) / 2, affinity_matrix
    
    def get_acc_from_affinity(self, affinity_matrix, gt_distribution=None):
        if gt_distribution is not None:
            # we assume that gt_distribution is language_feat @ language_feat.T 
            # UPDATE: we previously scales the distribution to [-1,1]. For the ease of visualization / picking a thres
            # we do NOT change the gt_distribution here.
            positive_mask = gt_distribution > self.similarity_thres

            # calculate top 1 and top 5 accuracy
            top1_bool = construct_top_k_mask(affinity_matrix, k=1)
            top5_bool = construct_top_k_mask(affinity_matrix, k=5)
            
            # calculate the transpose as well 
            top1_bool_t = construct_top_k_mask(affinity_matrix.T, k=1)
            top5_bool_t = construct_top_k_mask(affinity_matrix.T, k=5)

            # calculate the accuracy
            acc1 = torch.any(top1_bool & positive_mask, dim=1).float().mean()
            acc5 = torch.any(top5_bool & positive_mask, dim=1).float().mean()
            acc1_t = torch.any(top1_bool_t & positive_mask, dim=1).float().mean() # because positive mask is symmetric
            acc5_t = torch.any(top5_bool_t & positive_mask, dim=1).float().mean() # because positive mask is symmetric
            acc1 = (acc1 + acc1_t) / 2 * 100 # because timm acc is from 0-100%, the acc calculated above is a probability
            acc5 = (acc5 + acc5_t) / 2 * 100 
            return acc1, acc5
            
        labels = torch.arange(affinity_matrix.shape[0], device=affinity_matrix.device, dtype=torch.long)
        acc1, acc5 = accuracy(affinity_matrix, labels, topk=(1, 5))
        acc1_t, acc5_t = accuracy(affinity_matrix.T, labels, topk=(1, 5))
        acc1 = (acc1 + acc1_t) / 2
        acc5 = (acc5 + acc5_t) / 2
        return acc1, acc5

    def forward(self, feature_dict : dict, logit_scale : torch.Tensor, output_dict=False):
        # for every two modality pairs, compute the loss
        total_loss = 0
        losses = {}
        class_pairs = list(combinations(self.active_modalities, 2))
        for class_a, class_b in class_pairs:
            class_a, class_b = sorted([class_a, class_b])
            if {class_a, class_b} == {ModalityType.VISION, ModalityType.TEXT} and self.disable_vision_text_loss:
                continue
            if {class_a, class_b} == {ModalityType.TEXT, ModalityType.TACTILE} and self.disable_tactile_text_loss:
                continue
            class_a_feat = feature_dict[class_a]
            class_b_feat = feature_dict[class_b]
            if {class_a, class_b} == {ModalityType.TACTILE, ModalityType.TEXT} and self.use_tac_text_loss:
                loss, affinity_mat = self.tactile_text_loss(class_a_feat, class_b_feat, logit_scale)
                # the ground truth distribution is based on the language features
            else:
                loss, affinity_mat = self.clip_loss(class_a_feat, class_b_feat, logit_scale)
            if ModalityType.TEXT in {class_a, class_b}:
                acc1, acc5 = self.get_acc_from_affinity(affinity_mat, gt_distribution=feature_dict[ModalityType.TEXT] @ feature_dict[ModalityType.TEXT].T)
            else:
                acc1, acc5 = self.get_acc_from_affinity(affinity_mat)
            losses[f"{class_a}_{class_b}"] = loss 
            losses[f"{class_a}_{class_b}_acc1"] = acc1
            losses[f"{class_a}_{class_b}_acc5"] = acc5
            total_loss += loss
        losses["average_loss"] = total_loss / len(class_pairs)
        losses["average_acc1"] = torch.mean(torch.stack([i for k,i in losses.items() if "acc1" in k]))
        losses["average_acc5"] = torch.mean(torch.stack([i for k,i in losses.items() if "acc5" in k]))
        return losses if output_dict else losses["average_loss"]
