from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Union, Mapping, Optional
import random
import torch
from torch import nn, Tensor
import torch.distributed as dist

Losses = Union[Iterable[Tensor], Mapping[str, Tensor]]

class PCGrad:
    """
    PCGrad wrapper for any PyTorch optimizer.

    用法（不变）:
        base_opt = torch.optim.Adam(model.parameters(), lr=...)
        opt_model_pc = PCGrad(base_opt)

        per_task_obj = {"sr": loss_sr, "sod": loss_sod}
        opt_model_pc.pc_backward(per_task_obj, scaler=scaler)

        # (可选)裁剪
        torch.nn.utils.clip_grad_norm_(opt_model_pc.params(), max_norm)

        # step
        if scaler is not None:
            scaler.step(opt_model_pc._optim); scaler.update()
        else:
            opt_model_pc._optim.step()
        opt_model_pc.zero_grad()

        # 统计（新）
        stats = opt_model_pc.get_last_stats()
    """

    def __init__(self, optimizer: torch.optim.Optimizer, eps: float = 1e-12,
                 track_stats: bool = True) -> None:
        self._optim = optimizer
        self.eps = float(eps)
        self._track_stats = bool(track_stats)
        self._last_stats: Dict[str, float] | None = None  # 最近一次 pc_backward 的统计

    # ------- 常用代理 -------
    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optim.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self._optim.step()

    def state_dict(self) -> dict:
        return self._optim.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self._optim.load_state_dict(sd)

    def params(self) -> List[nn.Parameter]:
        ps: List[nn.Parameter] = []
        for g in self._optim.param_groups:
            for p in g["params"]:
                if p is not None and p.requires_grad:
                    ps.append(p)
        return ps

    # ------- 小工具 -------
    @torch.no_grad()
    def _dot(self, gi: List[Tensor], gj: List[Tensor]) -> Tensor:
        s = torch.zeros([], device=gi[0].device)
        for a, b in zip(gi, gj):
            if a is not None and b is not None:
                s = s + (a.flatten() * b.flatten()).sum()
        return s

    @torch.no_grad()
    def _norm(self, g: List[Tensor]) -> Tensor:
        s = torch.zeros([], device=g[0].device)
        for a in g:
            if a is not None:
                s = s + (a.flatten() * a.flatten()).sum()
        return torch.sqrt(s + self.eps)

    def _as_list(self, losses: Losses) -> Tuple[List[Tensor], List[str]]:
        if isinstance(losses, Mapping):
            keys = list(losses.keys())
            vals = [losses[k] for k in keys]
        else:
            vals = list(losses)
            keys = [str(i) for i in range(len(vals))]
        if len(vals) == 0:
            raise ValueError("`losses` is empty.")
        for i, L in enumerate(vals):
            if not torch.is_tensor(L):
                raise TypeError(f"losses[{i}] is not a Tensor.")
            if L.dim() != 0:
                vals[i] = L.mean()
        return vals, keys

    def enable_stats(self, flag: bool = True):
        self._track_stats = bool(flag)

    def get_last_stats(self) -> Optional[Dict[str, float]]:
        """返回最近一次 pc_backward() 的统计（若未启用或尚未产生则为 None）"""
        return None if (not self._track_stats) else self._last_stats

    # ------- 核心：PCGrad + 统计 -------
    def pc_backward(
        self,
        losses: Losses,
        scaler: torch.cuda.amp.GradScaler | None = None,
        retain_graph: bool = False,
        sync_ddp: bool = True,
    ) -> None:
        """
        计算每个任务的梯度，做 PCGrad 投影，然后把合并后的梯度写入 p.grad。
        若提供 scaler，将对每个 loss 先做 scale，并保持与 GradScaler.step() 兼容。
        """
        params = self.params()
        if len(params) == 0:
            return

        loss_list, keys = self._as_list(losses)
        dev = params[0].device

        # 逐任务求梯度（不写入 .grad；autograd.grad 返回列表）
        per_task_grads: List[List[Tensor]] = []
        for L in loss_list:
            if scaler is not None:
                g = torch.autograd.grad(
                    scaler.scale(L), params,
                    retain_graph=True, allow_unused=True
                )
            else:
                g = torch.autograd.grad(
                    L, params, retain_graph=True, allow_unused=True
                )
            g = [torch.zeros_like(p) if (gi is None) else gi for gi, p in zip(g, params)]
            per_task_grads.append(g)

        # ===== 统计：投影前的 pairwise 夹角 & 冲突 =====
        last_stats: Dict[str, float] | None = None
        if self._track_stats:
            try:
                m = len(per_task_grads)
                cos_vals = []
                num_conflicts = 0
                angle_sr_sod = None

                # 快速构建 name->idx
                name2idx = {k: i for i, k in enumerate(keys)}

                # 计算两两 cos / 冲突数
                for i in range(m):
                    for j in range(i + 1, m):
                        gi, gj = per_task_grads[i], per_task_grads[j]
                        dot = self._dot(gi, gj)
                        ni = self._norm(gi)
                        nj = self._norm(gj)
                        cos_ij = (dot / (ni * nj + self.eps)).clamp(-1.0, 1.0)
                        cos_vals.append(cos_ij.item())
                        if dot.item() < 0.0:
                            num_conflicts += 1

                # 特别关心 sr-sod 的夹角（若存在）
                if ("sr" in name2idx) and ("sod" in name2idx):
                    i, j = name2idx["sr"], name2idx["sod"]
                    gi, gj = per_task_grads[i], per_task_grads[j]
                    dot = self._dot(gi, gj)
                    ni = self._norm(gi)
                    nj = self._norm(gj)
                    cos = (dot / (ni * nj + self.eps)).clamp(-1.0, 1.0)
                    angle_sr_sod = torch.rad2deg(torch.arccos(cos)).item()
                    sr_sod_conflict = float(dot.item() < 0.0)
                else:
                    sr_sod_conflict = 0.0

                last_stats = {
                    "projected": float(num_conflicts > 0.0),   # 本次是否发生过投影
                    "conflicts": float(num_conflicts),
                    "cos_min": float(min(cos_vals)) if cos_vals else 1.0,
                    "cos_avg": float(sum(cos_vals) / len(cos_vals)) if cos_vals else 1.0,
                    "angle_sr_sod": float(angle_sr_sod) if angle_sr_sod is not None else 0.0,
                    "conflict_sr_sod": sr_sod_conflict,
                }
            except Exception:
                last_stats = None  # 统计失败不影响训练

        # ===== PCGrad 投影 =====
        m = len(per_task_grads)
        for i in range(m):
            order = list(range(m))
            random.shuffle(order)
            for j in order:
                if j == i:
                    continue
                gi, gj = per_task_grads[i], per_task_grads[j]
                dot = self._dot(gi, gj)
                if dot < 0:  # 仅冲突时投影
                    denom = (self._norm(gj) ** 2) + self.eps
                    coeff = (dot / denom).clamp(-1e6, 1e6)
                    per_task_grads[i] = [a - coeff * b for a, b in zip(gi, gj)]

        # 合并（平均）并写回 p.grad
        self.zero_grad(set_to_none=True)
        for k, p in enumerate(params):
            merged = sum(g[k] for g in per_task_grads) / float(m)
            p.grad = merged  # 若前面使用了 scaler.scale，这里仍然是 scaled grad

        # DDP 同步
        if sync_ddp and dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            for p in params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(world)

        # 保存统计
        self._last_stats = last_stats if self._track_stats else None
