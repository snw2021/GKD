import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.warmup = cfg.KD.WARMUP

        self.grad_accumulator = {
            "grad_in": 0,
            "grad_out": 0,
        }
        def hook_fun(module, grad_input, grad_output):
            # self.grad_accumulator["grad_in"] = grad_input[0]
            self.grad_accumulator["grad_out"] = grad_output[0]

        self.student.fc.register_backward_hook(hook_fun)

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image) # input: [64, 3, 32, 32], output: [64, 100]
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image) # 教师模型前向传播, [64, 100]

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target) # 学生模型的交叉熵损失
        # loss_kd = self.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.temperature) # 学生模型和教师模型的蒸馏损失
        loss_kd = min(kwargs["epoch"] / self.warmup, 1.0) * kd_loss(logits_student, logits_teacher, self.temperature) # 学生模型和教师模型的蒸馏损失

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, logits_teacher, losses_dict
