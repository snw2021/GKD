import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.autograd import Function

from ._base import Distiller

def dynamic_coefficient(target_confidence, c=0.0, k=1.0):
    return 1 / (1 + torch.exp(-k * (target_confidence - c)))

class CustomGradientFunction(Function):
    @staticmethod
    def forward(ctx, input, scale_factor, truth_index, confidence):
        ctx.scale_factor = scale_factor
        ctx.confidence = dynamic_coefficient(confidence, c=confidence.mean(), k=0.6)
        ctx.truth_index = truth_index
        return input
    @staticmethod
    def backward(ctx, grad_output):
        mask = torch.ones_like(grad_output)
        mask[torch.arange(grad_output.shape[0]), ctx.truth_index] = 0
        scaled_grad_output = grad_output * (mask * ctx.scale_factor * ctx.confidence.unsqueeze(1) + (1 - mask))
        return scaled_grad_output, None, None, None

def kd_loss(logits_student, logits_teacher, target, temperature):
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    confidence_teacher, _ = torch.max(pred_teacher, dim=1)

    scale_factor = 2.0

    custom_grad = CustomGradientFunction.apply
    modified_logits_student = custom_grad(logits_student, scale_factor, target, confidence_teacher.detach())

    # KL
    log_pred_student = F.log_softmax(modified_logits_student / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2

    return loss_kd

class GradientKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(GradientKD, self).__init__(student, teacher)
        self.temperature = cfg.GradientKD.TEMPERATURE
        self.ce_loss_weight = cfg.GradientKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.GradientKD.LOSS.KD_WEIGHT
        self.warmup = cfg.GradientKD.WARMUP
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
        else: # resnet or wrn
            self.student.fc.register_backward_hook(hook_fun)

    def reset_grad_accumulator(self):
        self.grad_accumulator = {
            "grad_in": 0,
            "grad_out": 0,
        }
    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = min(kwargs["epoch"] / self.warmup, 1.0) * kd_loss(logits_student, logits_teacher, target, self.temperature)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, logits_teacher, losses_dict

