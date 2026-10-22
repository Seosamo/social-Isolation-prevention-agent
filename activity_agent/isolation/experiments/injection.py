"""Paired interventions on source-flow aggregates; never modify production inputs."""
from __future__ import annotations

import numpy as np

from .features import recompute

SCENARIOS = {"S1": "집단 현재 유동 감소", "S2": "낮 시간 선택적 위축", "S3": "집단 위축·전체 완만 감소",
             "S4": "두 달 누적 위축", "S5": "현재 월 일시 위축", "S6": "집단 증가 방향 대조",
             "S7": "소지역 전체 집단의 공통 충격", "S8": "규모 유지·활동 분포 변화"}
SCENARIOS["NC2"] = "시간·요일 비중 유지, 전체 규모만 감소"


def inject(frame, scenario, strength):
    f = frame.copy(deep=True)
    feasible = np.ones(len(f), bool)
    def group_factor(column, factor):
        regional = "r" + column[1:] if column.startswith("g") else "rp" + column[1:]
        old = f[column].copy()
        f[column] *= factor
        f[regional] += f[column] - old
    if scenario == "S1":
        group_factor("g0", 1-strength)
    elif scenario == "S5":
        # Counterfactual prior comparison is stable; this is not a normal label.
        prior_factor = ((f.p2+f.p3)/2) / f.p1.where(f.p1 > 0)
        feasible &= np.isfinite(prior_factor).to_numpy() & f.g0.gt(0).to_numpy()
        group_factor("g1", prior_factor)
        group_factor("p1", prior_factor)
        target = (f.g1+f.g2)/2 * (1-strength)
        group_factor("g0", target/f.g0.where(f.g0>0))
    elif scenario == "S3":
        f["g0"] *= 1-strength
        f["r0"] *= 1-strength/7  # 35% group reduction corresponds to 5% total reduction.
        feasible &= f.r0.ge(f.g0).to_numpy()
    elif scenario == "S4":
        # Fix an actual two-month declining trajectory on the matched grid.
        # Apply the same prior-month factor to both overlapping grid windows.
        prior_factor = f.g2*(1-strength)/f.g1.where(f.g1>0)
        feasible &= np.isfinite(prior_factor).to_numpy() & f.g0.gt(0).to_numpy()
        group_factor("g1", prior_factor)
        group_factor("p1", prior_factor)
        target = f.g1*(1-strength)
        group_factor("g0", target/f.g0.where(f.g0>0))
    elif scenario == "S6":
        group_factor("g0", 1+strength)
    elif scenario == "S7":
        # AGE-family common shock: all groups in a block scale equally. No fabricated
        # synchronization with independently aggregated time/weekday counts.
        f["g0"] *= 1-strength
        f["r0"] *= 1-strength
    elif scenario == "S2":
        hours = [f"h0_{i}" for i in range(24)]
        day = [f"h0_{i}" for i in range(9, 18)]
        other = [h for h in hours if h not in day]
        total = f[hours].sum(axis=1)
        f[day] *= 1-strength
        remaining = total*(1-strength/4) - f[day].sum(axis=1)
        previous = f[other].sum(axis=1)
        feasible &= previous.gt(0).to_numpy() & remaining.ge(0).to_numpy()
        f[other] = f[other].mul(remaining / previous.where(previous > 0), axis=0)
        f["d0"] = f[day].sum(axis=1)
    elif scenario == "NC2":
        for prefix, n in (("h", 24), ("y", 7)):
            columns = [f"{prefix}0_{i}" for i in range(n)]
            f[columns] *= 1-strength
        f["d0"] *= 1-strength
        f["w0"] *= 1-strength
    elif scenario == "S8":
        for prefix, n, shift in (("h", 24, 12), ("y", 7, 2)):
            columns = [f"{prefix}0_{i}" for i in range(n)]
            values = f[columns].to_numpy()
            f[columns] = (1-strength)*values + strength*np.roll(values, shift, axis=1)
        f["d0"] = f[[f"h0_{i}" for i in range(9, 18)]].sum(axis=1)
        f["w0"] = f[["y0_5", "y0_6"]].mean(axis=1)
    else:
        raise ValueError(scenario)
    f = recompute(f)
    f["injection_feasible"] = feasible
    return f
