"""Integration test for the OurMethod branch of KnowledgeDistiller.train_step.

The distiller is built with ``__new__`` so no teacher/tokenizer/data download is
needed; only the attributes the branch touches are populated.
"""

import torch
from torch import nn, optim
from torch.amp import GradScaler

from config import OurMethodConfig
from distiller import KnowledgeDistiller
from src.criterions.our_method import OurMethodDistillation


class _Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeStudent(nn.Module):
    def __init__(self, dim: int = 8, vocab: int = 20):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        return _Output(self.proj(self.embed(input_ids)))


class _Scheduler:
    def step(self):
        pass


def _make_distiller(config: OurMethodConfig) -> KnowledgeDistiller:
    distiller = KnowledgeDistiller.__new__(KnowledgeDistiller)
    distiller.config = config
    distiller.device_s = torch.device("cpu")
    distiller.device_t = torch.device("cpu")
    distiller.model_student = _FakeStudent()
    distiller.criterion = OurMethodDistillation(
        w_task=config.w_task,
        w_fusion=config.w_fusion,
        normalize_target=config.normalize_target,
    )
    distiller.optimizer = optim.AdamW(distiller.model_student.parameters(), lr=1e-3)
    distiller.scheduler = _Scheduler()
    distiller.scaler = GradScaler("cuda", enabled=False)
    return distiller


def _batch(dim: int = 8, vocab: int = 20, two_sides: bool = True) -> dict:
    generator = torch.Generator().manual_seed(1)
    batch = {
        "input_ids1_stu": torch.randint(1, vocab, (4, 6), generator=generator),
        "attention_mask1_stu": torch.ones(4, 6, dtype=torch.long),
        "target1": torch.randn(4, dim),
    }
    if two_sides:
        batch["input_ids2_stu"] = torch.randint(1, vocab, (4, 6), generator=generator)
        batch["attention_mask2_stu"] = torch.ones(4, 6, dtype=torch.long)
        batch["target2"] = torch.randn(4, dim)
    return batch


def test_train_step_reports_fusion_metrics_and_updates_parameters():
    config = OurMethodConfig(w_task=0.5, w_fusion=1.0)
    distiller = _make_distiller(config)
    before = distiller.model_student.proj.weight.detach().clone()

    loss, metrics = distiller.train_step(_batch())

    assert torch.isfinite(loss)
    assert {
        "loss_total",
        "loss_base",
        "loss_fusion",
        "cos_target",
        "grad_norm",
        "lr",
    } <= set(metrics)
    assert metrics["grad_norm"] >= 0.0
    assert not torch.allclose(before, distiller.model_student.proj.weight.detach())


def test_train_step_handles_a_single_target_side():
    config = OurMethodConfig(w_task=0.5, w_fusion=1.0)
    distiller = _make_distiller(config)

    loss, metrics = distiller.train_step(_batch(two_sides=False))

    assert torch.isfinite(loss)
    assert "loss_fusion" in metrics
