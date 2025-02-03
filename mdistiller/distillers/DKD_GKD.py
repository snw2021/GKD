import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from ._base import Distiller


def dynamic_coefficient(target_confidence, c=0.0, k=10.0):
    return 1 / (1 + torch.exp(-k * (target_confidence - c)))

class CustomGradientFunction(Function):
    @staticmethod
    def forward(ctx, input, scale_factor, truth_index, confidence):
        ctx.scale_factor = scale_factor
        ctx.confidence = dynamic_coefficient(confidence, c=confidence.mean(), k=0.6)
        ctx.truth_index = truth_index
        return input  # 直接返回输入，因为前向传播无需改动

    @staticmethod
    def backward(ctx, grad_output):
        mask = torch.ones_like(grad_output)
        mask[torch.arange(grad_output.shape[0]), ctx.truth_index] = 0
        scaled_grad_output = grad_output * (mask * ctx.scale_factor * ctx.confidence.unsqueeze(1) + (1 - mask))
        return scaled_grad_output, None, None, None

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)

    log_pred_student = torch.log(pred_student)

    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )

    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )

    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False) # size_average=False表示小批量样本上的各样本kl损失这和
        * (temperature**2)
        / target.shape[0]
    )

    return alpha * tckd_loss + beta * nckd_loss

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

class DKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""
    def __init__(self, student, teacher, cfg):
        super(DKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

        self.grad_accumulator = {
            "grad_in": 0,
            "grad_out": 0,
        }
        def hook_fun(module, grad_input, grad_output):
            self.grad_accumulator["grad_out"] = grad_output[0]

        if self.student.__class__.__name__ in ['VGG', 'MobileNetV2']:
            self.student.classifier.register_backward_hook(hook_fun)
        elif self.student.__class__.__name__ in ['ShuffleNet', 'ShuffleNetV2']:
            self.student.linear.register_backward_hook(hook_fun)
        else:  # resnet or wrn
            self.student.fc.register_backward_hook(hook_fun)

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
        confidence_teacher, _ = torch.max(pred_teacher, dim=1)

        scale_factor = 2.0
        custom_grad = CustomGradientFunction.apply
        modified_logits_student = custom_grad(logits_student, scale_factor, target, confidence_teacher.detach())

        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            modified_logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, logits_teacher, losses_dict
