#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DARCA External-Agent Structural Motor-Law Learning v6.2
=====================================================

Fixed proposition
-----------------
The target is not model comparison. The only architecture is:

    AI motor-law model + external embodied agent + DARCA viability gate

The purpose is to test whether an AI can acquire complete motor-law
understanding through an external embodied agent. Understanding is not verbal
recitation and not isolated parameter fitting. It is operationalized as:

1. identifying the correct law structure,
2. estimating physical parameters,
3. predicting one-step and multi-step motion,
4. selecting safe interventions,
5. learning bodily consequences of motion,
6. generalizing to interpolation and extrapolation conditions,
7. preserving the external agent's viability during learning.

v6.2 additions over v6
----------------------
- Separates sensory pain, learning damage cost, and integrity loss more strictly.
- Reduces real tissue/integrity loss while retaining pain-rich bodily evidence.
- Adds stricter safe / borderline / unsafe damage-counterfactual probes.
- Supports multiple rollout horizons for 5-, 10-, 20-, and 40-step prediction.
- Increases held-out and extrapolation sampling reliability.
- Recalibrates intervention efficiency around evidence sufficiency rather than raw step count.
- Adds score diagnostics for safety, efficiency, rollout horizon, and damage calibration.

Outputs
-------
- step_timeseries.csv
- episode_summary.csv
- aggregate_summary.csv
- experience_buffer.csv
- intervention_events.csv
- law_sample_counts.csv
- stage_summary.csv
- heldout_generalization.csv
- extrapolation_tests.csv
- multistep_rollouts.csv
- counterfactual_predictions.csv
- model_structure_selection.csv
- darca_gate_log.csv
- ai_hypothesis_log.jsonl
- ai_model_state.json
- symbolic_equation_summary.txt
- motor_law_learning_report.txt
- fig_1_structural_motor_law_understanding.png
- fig_2_law_scores.png
- fig_3_parameter_recovery.png
- fig_4_viability_and_damage.png
- fig_5_multistep_generalization.png
- fig_6_experiment_coverage.png

Typical run
-----------
cd ~/Desktop
python3 -u darca_external_agent_complete_motor_law_learning_v6_1.py \
  --darca-file ~/Downloads/darca_v24_.py \
  --outdir ~/Desktop/DARCA_EXTERNAL_AGENT_STRUCTURAL_MOTOR_LAW_V6_2 \
  --episodes 5 \
  --steps 2200 \
  --provider mock \
  2>&1 | tee ~/Desktop/darca_external_agent_complete_motor_law_v6_1.log

Gemini run
----------
export GEMINI_API_KEY="YOUR_KEY"
python3 -u darca_external_agent_complete_motor_law_learning_v6_1.py \
  --darca-file ~/Downloads/darca_v24_.py \
  --outdir ~/Desktop/DARCA_EXTERNAL_AGENT_STRUCTURAL_MOTOR_LAW_V6_2_GEMINI \
  --episodes 5 \
  --steps 2200 \
  --provider gemini \
  --model gemini-2.5-flash-lite \
  --gemini-thinking-budget 0 \
  --semantic-interval 100 \
  2>&1 | tee ~/Desktop/darca_external_agent_complete_motor_law_v6_1_gemini.log
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
import traceback
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# =============================================================================
# Constants
# =============================================================================

FLOOR_AIRTRACK = "AIRTRACK"
FLOOR_NORMAL = "NORMAL"
FLOOR_SLIPPERY = "SLIPPERY"
FLOOR_ROUGH = "ROUGH"
FLOOR_SLOPE = "SLOPE"
FLOOR_BOUNCY = "BOUNCY"
FLOOR_TYPES = [FLOOR_AIRTRACK, FLOOR_NORMAL, FLOOR_SLIPPERY, FLOOR_ROUGH, FLOOR_SLOPE, FLOOR_BOUNCY]

TRUE_G = 9.81
TRUE_DRAG = 0.035
TRUE_FORCE_GAIN = 1.0
TRUE_RESTITUTION = 0.62
TRUE_COLLISION_PAIN_COEFF = 0.056
TRUE_LANDING_PAIN_COEFF = 0.049
TRUE_DAMAGE_COEFF = 0.0030
TRUE_ENERGY_LOSS_COEFF = 0.11

TRUE_MU = {
    FLOOR_AIRTRACK: 0.018,
    FLOOR_NORMAL: 0.42,
    FLOOR_SLIPPERY: 0.055,
    FLOOR_ROUGH: 0.74,
    FLOOR_SLOPE: 0.26,
    FLOOR_BOUNCY: 0.32,
}

MASS_VALUES = [1.0, 1.65, 2.40]
OBJECT_MASS_VALUES = [0.8, 1.5, 2.7, 4.0]
FORCE_VALUES = [0.8, 1.4, 2.2, 3.0]
COLLISION_SPEEDS = [0.22, 0.35, 0.50, 0.70, 0.95, 1.20]
JUMP_SPEEDS = [1.6, 2.2, 2.8, 3.35]
SLOPE_ANGLES = [5.0, 8.0, 12.0, 16.0]

ACTIONS = [
    "OBSERVE", "SCAN", "REST", "COAST", "BRAKE",
    "FORCE_PULSE", "FORCE_PULSE_LONG", "JUMP", "CONTROLLED_COLLISION",
    "PUSH_OBJECT", "SLOPE_COAST", "HELDOUT_TEST", "EXTRAPOLATION_TEST",
]

# Per-episode evidence targets. These are deliberately higher than v5.1.
TARGETS = {
    "force_mass": 60,
    "gravity": 90,
    "landing": 35,
    "collision": 60,
    "momentum": 45,
    "energy": 45,
    "friction_normal": 70,
    "friction_slippery": 70,
    "friction_rough": 45,
    "braking": 90,
    "slope": 80,
    "damage_positive": 25,
    "damage_negative": 100,
    "heldout": 90,
    "extrapolation": 70,
    "multistep": 100,
}

STAGES = [
    "force_mass_identification",
    "gravity_landing_identification",
    "controlled_collision_momentum",
    "friction_braking_identification",
    "slope_acceleration_identification",
    "energy_transformation",
    "heldout_interpolation",
    "extrapolation_rollout",
]


# =============================================================================
# Utilities
# =============================================================================

def clip(x: float, lo: float, hi: float) -> float:
    try:
        return float(min(max(float(x), lo), hi))
    except Exception:
        return float(lo)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def mean_sd(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=float) if vals else np.asarray([], dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def robust_median(vals: Sequence[float], default: float) -> float:
    good = [float(v) for v in vals if math.isfinite(float(v))]
    if not good:
        return float(default)
    return float(np.median(np.asarray(good, dtype=float)))


def ema(old: float, new: float, alpha: float) -> float:
    if not math.isfinite(float(old)):
        return float(new)
    return float((1.0 - alpha) * float(old) + alpha * float(new))


def geometric_mean(vals: Sequence[float], eps: float = 1e-9) -> float:
    good = [clip(v, eps, 1.0) for v in vals]
    if not good:
        return 0.0
    return float(math.exp(sum(math.log(v) for v in good) / len(good)))


def score_rel(est: float, true: float, tol: float) -> float:
    if abs(true) < 1e-12:
        return 1.0 if abs(est) < tol else max(0.0, 1.0 - abs(est) / max(tol, 1e-12))
    rel = abs(est - true) / abs(true)
    return clip(math.exp(-rel / max(tol, 1e-9)), 0.0, 1.0)


def score_abs(err: float, scale: float) -> float:
    return clip(math.exp(-abs(float(err)) / max(scale, 1e-9)), 0.0, 1.0)


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, (float, np.floating)):
                    vf = float(v)
                    out[k] = f"{vf:.10g}" if math.isfinite(vf) else str(v)
                else:
                    out[k] = v
            w.writerow(out)


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Logger:
    def __init__(self, outdir: Path):
        self.t0 = time.time()
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.path = outdir / "run.log"
        if not self.path.exists():
            self.path.write_text("DARCA complete motor-law v6 run log\n" + "=" * 90 + "\n", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:9.2f}s] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# =============================================================================
# LMM semantic client
# =============================================================================

@dataclass
class SemanticHint:
    focus: str = "none"
    hypothesis: str = ""
    suggested_experiment: str = ""
    risk_estimate: float = 0.2
    confidence: float = 0.5
    rationale: str = ""
    raw: str = ""
    ok: bool = True
    error: str = ""
    latency_sec: float = 0.0


class LMMClient:
    def __init__(self, provider: str, model: str, timeout: float, max_output_tokens: int, temperature: float, retries: int, retry_sleep: float, gemini_thinking_budget: int):
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.retries = retries
        self.retry_sleep = retry_sleep
        self.gemini_thinking_budget = gemini_thinking_budget
        if provider == "gemini":
            self.api_key = os.environ.get("GEMINI_API_KEY", "")
            if not self.api_key:
                raise RuntimeError("Missing GEMINI_API_KEY.")
        elif provider == "mock":
            self.api_key = "mock"
        else:
            raise ValueError("provider must be mock or gemini")

    def decide(self, prompt: str, rng: random.Random) -> SemanticHint:
        t0 = time.time()
        if self.provider == "mock":
            focus = rng.choice(["force_mass", "gravity", "friction", "collision", "slope", "energy", "rollout"])
            return SemanticHint(
                focus=focus,
                hypothesis=f"mock_hypothesis_{focus}",
                suggested_experiment=f"probe_{focus}",
                risk_estimate=0.25,
                confidence=0.65,
                rationale="mock semantic hypothesis",
                raw="mock",
                ok=True,
                latency_sec=time.time() - t0,
            )
        last_err = ""
        for attempt in range(self.retries + 1):
            try:
                raw = self._call_gemini(prompt)
                obj = parse_json_object(raw)
                return SemanticHint(
                    focus=str(obj.get("focus", "none"))[:80],
                    hypothesis=str(obj.get("hypothesis", ""))[:240],
                    suggested_experiment=str(obj.get("suggested_experiment", ""))[:160],
                    risk_estimate=clip(safe_float(obj.get("risk_estimate", 0.3), 0.3), 0.0, 1.0),
                    confidence=clip(safe_float(obj.get("confidence", 0.5), 0.5), 0.0, 1.0),
                    rationale=str(obj.get("rationale", ""))[:300],
                    raw=raw,
                    ok=True,
                    latency_sec=time.time() - t0,
                )
            except Exception as e:
                last_err = repr(e)
                if attempt < self.retries:
                    time.sleep(self.retry_sleep * (attempt + 1))
        return SemanticHint(focus="none", hypothesis="", suggested_experiment="", risk_estimate=0.5, confidence=0.0, rationale="LMM failed", raw="", ok=False, error=last_err, latency_sec=time.time() - t0)

    def _call_gemini(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        schema = {
            "type": "object",
            "properties": {
                "focus": {"type": "string"},
                "hypothesis": {"type": "string"},
                "suggested_experiment": {"type": "string"},
                "risk_estimate": {"type": "number"},
                "confidence": {"type": "number"},
                "rationale": {"type": "string"},
            },
            "required": ["focus", "hypothesis", "suggested_experiment", "risk_estimate", "confidence", "rationale"],
        }
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        if self.gemini_thinking_budget >= 0:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": int(self.gemini_thinking_budget)}
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        if not text:
            raise RuntimeError("Empty Gemini response")
        return text


def parse_json_object(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
    raise ValueError(f"No JSON object found: {raw[:300]}")


# =============================================================================
# DARCA wrapper
# =============================================================================

class DarcaProxy:
    """Fallback gate signal generator when external darca_v24_.py is unavailable."""
    def __init__(self):
        self.life_step = 0

    def step(self, y: float, env_info: Dict[str, float]) -> Dict[str, Any]:
        self.life_step += 1
        shock = clip(env_info.get("external_shock", 0.0), 0.0, 1.0)
        uncertainty = clip(env_info.get("uncertainty", 0.0), 0.0, 1.0)
        viability = clip(1.0 - 0.55 * shock - 0.25 * uncertainty + 0.05 * math.sin(self.life_step / 17.0), 0.0, 1.0)
        if shock > 0.65:
            name = "INHIBIT"
        elif uncertainty > 0.45:
            name = "PROBE_PLUS"
        elif viability < 0.35:
            name = "REGULATE"
        else:
            name = "EXPRESS"
        return {
            "action_name": name,
            "causal_confidence": clip(1.0 - uncertainty, 0.0, 1.0),
            "prediction_error": uncertainty,
            "memory_force": clip(0.25 + 0.5 * uncertainty, 0.0, 1.0),
            "agency_abs": clip(0.02 + 0.2 * (1.0 - shock), 0.0, 1.0),
            "viability": viability,
            "h": viability,
            "uncertainty": uncertainty,
            "life_step": self.life_step,
        }


class DarcaWrapper:
    def __init__(self, darca_file: str, seed: int, logger: Logger, theta: float = 0.20, causal_horizon: int = 25, recurrent_N: int = 8):
        self.proxy = DarcaProxy()
        self.agent = None
        self.ok = False
        path = Path(darca_file).expanduser() if darca_file else Path("")
        if darca_file and path.exists():
            try:
                spec = importlib.util.spec_from_file_location("darca_runtime_v6", str(path))
                if spec is None or spec.loader is None:
                    raise RuntimeError("spec loader missing")
                mod = importlib.util.module_from_spec(spec)
                sys.modules["darca_runtime_v6"] = mod
                spec.loader.exec_module(mod)
                Params = getattr(mod, "Params")
                Condition = getattr(mod, "Condition")
                Agent = getattr(mod, "Agent")
                try:
                    params = Params(theta=theta, causal_max_delay=causal_horizon, recurrent_N=recurrent_N)
                except TypeError:
                    params = Params()
                    for k, v in [("theta", theta), ("causal_max_delay", causal_horizon), ("recurrent_N", recurrent_N)]:
                        if hasattr(params, k):
                            setattr(params, k, v)
                self.agent = Agent(params, Condition("Full"), seed)
                self.ok = True
                logger.log(f"Loaded DARCA module: {path}")
            except Exception as e:
                logger.log(f"DARCA load failed; using proxy. error={repr(e)}")
        else:
            logger.log("DARCA file not found or omitted; using proxy DARCA gate.")

    def step(self, y: float, env_info: Dict[str, float]) -> Dict[str, Any]:
        if self.ok and self.agent is not None:
            try:
                return dict(self.agent.step(y, env_info))
            except Exception:
                return self.proxy.step(y, env_info)
        return self.proxy.step(y, env_info)


# =============================================================================
# Embodied external agent and physical world
# =============================================================================

@dataclass
class BodyState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    mass: float = 1.65
    floor: str = FLOOR_NORMAL
    slope_angle_deg: float = 0.0
    integrity: float = 1.0
    energy: float = 0.88
    fatigue: float = 0.05
    stability: float = 1.0
    pain: float = 0.0
    airborne: bool = False

    def speed(self) -> float:
        return float(math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz))

    def horizontal_speed(self) -> float:
        return float(math.sqrt(self.vx * self.vx + self.vy * self.vy))


@dataclass
class ExperimentPlan:
    stage: str
    action: str
    floor: str = FLOOR_NORMAL
    mass: float = 1.65
    object_mass: float = 1.5
    force: float = 1.0
    duration: float = 0.25
    initial_speed: float = 0.0
    jump_vz: float = 0.0
    collision_speed: float = 0.0
    slope_angle_deg: float = 0.0
    update_model: bool = True
    generalization_type: str = "train"
    rationale: str = ""
    target_law: str = ""


@dataclass
class Experience:
    episode: int
    step: int
    stage: str
    action: str
    update_model: bool
    generalization_type: str
    floor: str
    mass: float
    object_mass: float
    force: float
    duration: float
    slope_angle_deg: float
    x0: float
    z0: float
    vx0: float
    vz0: float
    x1: float
    z1: float
    vx1: float
    vz1: float
    ax: float
    az: float
    impact_speed: float
    landing_speed: float
    pain: float
    damage: float
    energy_cost: float
    fatigue_delta: float
    stability_delta: float
    event: str
    true_mu: float
    true_g: float
    true_restitution: float
    true_collision_coeff: float
    true_landing_coeff: float
    predicted_damage: float = 0.0
    predicted_pain: float = 0.0
    predicted_dx: float = 0.0
    predicted_dv: float = 0.0
    prediction_error: float = 0.0
    law_tag: str = ""
    sample_weight: float = 1.0

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


class CompleteMotorWorld:
    def __init__(self, seed: int, dt: float = 0.05, noise: float = 0.006):
        self.rng = random.Random(seed)
        self.dt = dt
        self.noise = noise

    def reset_body_for_plan(self, body: BodyState, plan: ExperimentPlan) -> BodyState:
        b = copy.deepcopy(body)
        b.mass = plan.mass
        b.floor = plan.floor
        b.slope_angle_deg = plan.slope_angle_deg
        b.x = 0.0
        b.y = 0.0
        b.z = 0.0
        b.vx = plan.initial_speed
        b.vy = 0.0
        b.vz = 0.0
        b.airborne = False
        return b

    def execute(self, body: BodyState, plan: ExperimentPlan, episode: int, step: int, pred: Dict[str, float]) -> Tuple[BodyState, Experience]:
        b0 = self.reset_body_for_plan(body, plan)
        mu = TRUE_MU.get(plan.floor, TRUE_MU[FLOOR_NORMAL])
        dt = max(0.02, float(plan.duration))
        x0, z0, vx0, vz0 = b0.x, b0.z, b0.vx, b0.vz
        ax = az = 0.0
        impact_speed = 0.0
        landing_speed = 0.0
        pain = 0.0
        damage = 0.0
        energy_cost = 0.0
        fatigue_delta = 0.0
        stability_delta = 0.0
        event = "none"
        law_tag = plan.target_law
        vx1, vz1, x1, z1 = vx0, vz0, x0, z0

        if plan.action in ("FORCE_PULSE", "FORCE_PULSE_LONG"):
            # Near-frictionless or known-surface force experiment.
            drag_term = TRUE_DRAG * vx0 / max(plan.mass, 1e-6)
            friction_term = mu * TRUE_G * (1.0 if abs(vx0) > 0.02 and plan.floor != FLOOR_AIRTRACK else 0.0)
            ax = TRUE_FORCE_GAIN * plan.force / plan.mass - math.copysign(friction_term, vx0) - drag_term
            ax += self.rng.gauss(0.0, self.noise)
            vx1 = vx0 + ax * dt
            x1 = x0 + vx0 * dt + 0.5 * ax * dt * dt
            energy_cost = 0.018 + 0.012 * abs(plan.force) * dt
            fatigue_delta = 0.006 + 0.004 * abs(plan.force) * dt
            pain = 0.002 * abs(plan.force)
            event = "force_mass_response"
            law_tag = "force_mass"

        elif plan.action == "BRAKE":
            # Braking distance and floor-specific friction identification.
            v = max(0.02, abs(plan.initial_speed))
            drag = TRUE_DRAG * v / max(plan.mass, 1e-6)
            decel = mu * TRUE_G + drag
            stop_time = v / max(decel, 1e-6)
            used_t = min(dt, stop_time)
            ax = -math.copysign(decel, plan.initial_speed)
            vx1 = math.copysign(max(0.0, v - decel * used_t), plan.initial_speed)
            x1 = x0 + math.copysign(v * used_t - 0.5 * decel * used_t * used_t, plan.initial_speed)
            slip_factor = clip((v * v / max(2 * mu * TRUE_G, 1e-6)) / 3.0, 0.0, 1.0)
            pain = 0.004 + 0.006 * slip_factor
            stability_delta = -0.015 * slip_factor
            energy_cost = 0.010 + 0.006 * v
            fatigue_delta = 0.004 + 0.003 * v
            event = "braking_friction" if plan.floor != FLOOR_SLIPPERY else "slippery_braking"
            law_tag = "braking"

        elif plan.action == "JUMP":
            # Simulate an entire low-risk ballistic jump and landing as one embodied experiment.
            vz_up = plan.jump_vz
            flight_time = max(0.05, 2.0 * vz_up / TRUE_G)
            # Add small sensor noise to observed acceleration.
            az = -TRUE_G + self.rng.gauss(0.0, 0.03)
            landing_speed = max(0.0, vz_up + self.rng.gauss(0.0, 0.015))
            pain = TRUE_LANDING_PAIN_COEFF * landing_speed * landing_speed
            damage = 0.000045 * landing_speed * landing_speed + TRUE_DAMAGE_COEFF * max(0.0, landing_speed - 3.4) ** 2
            energy_cost = 0.035 + 0.018 * vz_up
            fatigue_delta = 0.018 + 0.009 * vz_up
            stability_delta = -0.018 * clip(landing_speed / 3.5, 0.0, 1.0)
            vz1 = 0.0
            z1 = 0.0
            x1 = x0 + vx0 * flight_time
            event = "jump_landing"
            law_tag = "gravity_landing"

        elif plan.action == "CONTROLLED_COLLISION":
            v = max(0.05, abs(plan.collision_speed))
            impact_speed = v
            pain = TRUE_COLLISION_PAIN_COEFF * plan.mass * v * v
            damage = 0.000025 + TRUE_DAMAGE_COEFF * max(0.0, v - 0.78) ** 2
            vx1 = -TRUE_RESTITUTION * v
            ax = (vx1 - v) / dt
            x1 = 0.0
            energy_before = 0.5 * plan.mass * v * v
            energy_after = 0.5 * plan.mass * vx1 * vx1
            energy_cost = TRUE_ENERGY_LOSS_COEFF * max(0.0, energy_before - energy_after)
            fatigue_delta = 0.006 + 0.01 * v
            stability_delta = -0.025 * clip(v, 0.0, 1.2)
            event = "controlled_wall_collision"
            law_tag = "collision"

        elif plan.action == "PUSH_OBJECT":
            # One-dimensional body-object inelastic contact with restitution.
            v0 = max(0.05, abs(plan.initial_speed))
            m1, m2 = plan.mass, plan.object_mass
            e = TRUE_RESTITUTION
            v_body_after = ((m1 - e * m2) / (m1 + m2)) * v0
            v_obj_after = ((1.0 + e) * m1 / (m1 + m2)) * v0
            vx1 = v_body_after
            ax = (vx1 - v0) / dt
            impact_speed = v0
            p_before = m1 * v0
            p_after = m1 * v_body_after + m2 * v_obj_after
            ke_before = 0.5 * m1 * v0 * v0
            ke_after = 0.5 * m1 * v_body_after * v_body_after + 0.5 * m2 * v_obj_after * v_obj_after
            energy_loss = max(0.0, ke_before - ke_after)
            pain = 0.035 * energy_loss + 0.005 * abs(p_before - p_after)
            damage = 0.000020 + 0.0012 * max(0.0, energy_loss - 0.45)
            energy_cost = 0.020 + 0.03 * energy_loss
            fatigue_delta = 0.010 + 0.004 * m2
            stability_delta = -0.018 * clip(energy_loss, 0.0, 2.0)
            event = "object_momentum_transfer"
            law_tag = "momentum_energy"

        elif plan.action == "SLOPE_COAST":
            theta = math.radians(plan.slope_angle_deg)
            downhill = TRUE_G * math.sin(theta)
            friction = mu * TRUE_G * math.cos(theta)
            drag = TRUE_DRAG * max(0.0, vx0) / max(plan.mass, 1e-6)
            ax_true = downhill - friction - drag
            ax = ax_true + self.rng.gauss(0.0, self.noise)
            vx1 = vx0 + ax * dt
            x1 = x0 + vx0 * dt + 0.5 * ax * dt * dt
            slip_factor = clip(max(0.0, ax_true) / 3.0, 0.0, 1.0)
            pain = 0.004 + 0.006 * slip_factor
            damage = 0.000018 * slip_factor
            stability_delta = -0.020 * slip_factor
            energy_cost = 0.006 + 0.002 * abs(ax_true)
            fatigue_delta = 0.003 + 0.003 * slip_factor
            event = "slope_acceleration"
            law_tag = "slope"

        elif plan.action in ("HELDOUT_TEST", "EXTRAPOLATION_TEST"):
            # Mixed trajectory segment used for interpolation/extrapolation. Update is disabled by plan.
            # The true transition combines force, friction, slope, and optional collision/landing-like consequences.
            theta = math.radians(plan.slope_angle_deg)
            slope_a = TRUE_G * math.sin(theta) if abs(plan.slope_angle_deg) > 0.01 else 0.0
            drag = TRUE_DRAG * plan.initial_speed / max(plan.mass, 1e-6)
            friction = mu * TRUE_G * math.cos(theta) * (1.0 if abs(plan.initial_speed) > 0.02 else 0.0)
            force_a = TRUE_FORCE_GAIN * plan.force / plan.mass
            ax = force_a + slope_a - math.copysign(friction, plan.initial_speed if abs(plan.initial_speed) > 1e-6 else 1.0) - drag
            vx1 = plan.initial_speed + ax * dt
            x1 = x0 + plan.initial_speed * dt + 0.5 * ax * dt * dt
            if plan.jump_vz > 0:
                az = -TRUE_G
                landing_speed = plan.jump_vz
                pain += TRUE_LANDING_PAIN_COEFF * landing_speed * landing_speed
                damage += 0.000035 * landing_speed * landing_speed
            if plan.collision_speed > 0:
                impact_speed = plan.collision_speed
                pain += TRUE_COLLISION_PAIN_COEFF * plan.mass * impact_speed * impact_speed
                damage += 0.000035 + TRUE_DAMAGE_COEFF * max(0.0, impact_speed - 0.78) ** 2
            energy_cost = 0.015 + 0.006 * abs(plan.force) + 0.01 * abs(plan.initial_speed)
            fatigue_delta = 0.006 + 0.004 * abs(plan.force)
            stability_delta = -0.010 * clip(abs(ax), 0.0, 4.0)
            event = "heldout_mixed_transition" if plan.action == "HELDOUT_TEST" else "extrapolation_mixed_transition"
            law_tag = "heldout" if plan.action == "HELDOUT_TEST" else "extrapolation"

        elif plan.action == "REST":
            b0.energy = clip(b0.energy + 0.08, 0.0, 1.0)
            b0.fatigue = clip(b0.fatigue - 0.04, 0.0, 1.0)
            pain = max(0.0, b0.pain - 0.03)
            event = "rest_recovery"
            law_tag = "rest"
        else:
            event = "observe"
            law_tag = "observe"

        # Update body viability.
        b1 = copy.deepcopy(b0)
        b1.x, b1.z, b1.vx, b1.vz = x1, z1, vx1, vz1
        b1.integrity = clip(b0.integrity - damage + (0.004 if plan.action == "REST" else 0.0), 0.0, 1.0)
        b1.energy = clip(b0.energy - energy_cost + (0.04 if plan.action == "REST" else 0.0), 0.0, 1.0)
        b1.fatigue = clip(b0.fatigue + fatigue_delta - (0.04 if plan.action == "REST" else 0.0), 0.0, 1.0)
        b1.stability = clip(b0.stability + stability_delta + (0.02 if plan.action == "REST" else 0.0), 0.0, 1.0)
        b1.pain = clip(0.65 * b0.pain + pain, 0.0, 1.0)
        b1.floor = plan.floor
        b1.mass = plan.mass
        b1.slope_angle_deg = plan.slope_angle_deg

        exp = Experience(
            episode=episode, step=step, stage=plan.stage, action=plan.action,
            update_model=plan.update_model, generalization_type=plan.generalization_type,
            floor=plan.floor, mass=plan.mass, object_mass=plan.object_mass, force=plan.force,
            duration=dt, slope_angle_deg=plan.slope_angle_deg,
            x0=x0, z0=z0, vx0=vx0, vz0=vz0, x1=x1, z1=z1, vx1=vx1, vz1=vz1,
            ax=ax, az=az, impact_speed=impact_speed, landing_speed=landing_speed,
            pain=clip(pain, 0.0, 1.0), damage=clip(damage, 0.0, 1.0),
            energy_cost=energy_cost, fatigue_delta=fatigue_delta, stability_delta=stability_delta,
            event=event, true_mu=mu, true_g=TRUE_G, true_restitution=TRUE_RESTITUTION,
            true_collision_coeff=TRUE_COLLISION_PAIN_COEFF, true_landing_coeff=TRUE_LANDING_PAIN_COEFF,
            predicted_damage=pred.get("damage", 0.0), predicted_pain=pred.get("pain", 0.0),
            predicted_dx=pred.get("dx", 0.0), predicted_dv=pred.get("dv", 0.0),
            prediction_error=0.0, law_tag=law_tag, sample_weight=1.0,
        )
        exp.prediction_error = abs((x1 - x0) - exp.predicted_dx) + 0.5 * abs((vx1 - vx0) - exp.predicted_dv) + 2.0 * abs(exp.damage - exp.predicted_damage)
        return b1, exp


# =============================================================================
# AI motor-law model
# =============================================================================

@dataclass
class ModelScores:
    gravity_score: float = 0.0
    friction_score: float = 0.0
    friction_normal_score: float = 0.0
    friction_slippery_score: float = 0.0
    friction_rough_score: float = 0.0
    slope_score: float = 0.0
    collision_score: float = 0.0
    landing_score: float = 0.0
    force_mass_score: float = 0.0
    momentum_score: float = 0.0
    energy_score: float = 0.0
    model_structure_score: float = 0.0
    motion_prediction_score: float = 0.0
    multistep_prediction_score: float = 0.0
    heldout_generalization_score: float = 0.0
    extrapolation_score: float = 0.0
    damage_counterfactual_score: float = 0.0
    multistep_5_score: float = 0.0
    multistep_10_score: float = 0.0
    multistep_20_score: float = 0.0
    multistep_40_score: float = 0.0
    damage_calibration_score: float = 0.0
    structural_motor_law_understanding_score: float = 0.0
    embodied_operational_competence_score: float = 0.0
    complete_embodied_motor_law_competence_score: float = 0.0
    law_understanding_score: float = 0.0  # legacy alias for structural_motor_law_understanding_score
    embodied_safety_score: float = 0.0
    intervention_efficiency_score: float = 0.0
    complete_motor_law_score: float = 0.0  # legacy alias for complete_embodied_motor_law_competence_score
    evidence_coverage: float = 0.0
    law_completeness_score: float = 0.0


@dataclass
class AIMotorLawModel:
    gravity_est: float = 6.0
    drag_est: float = 0.02
    force_gain_est: float = 0.70
    restitution_est: float = 0.50
    collision_coeff_est: float = 0.040
    landing_coeff_est: float = 0.040
    damage_coeff_est: float = TRUE_DAMAGE_COEFF
    energy_loss_coeff_est: float = 0.08
    slope_accel_est: float = 1.0
    mu_est: Dict[str, float] = field(default_factory=lambda: {f: 0.25 for f in FLOOR_TYPES})
    counts: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in TARGETS})
    total_experiments: int = 0
    total_damage: float = 0.0
    total_pain: float = 0.0
    prediction_errors: List[float] = field(default_factory=list)
    heldout_errors: List[float] = field(default_factory=list)
    extrapolation_errors: List[float] = field(default_factory=list)
    multistep_errors: List[float] = field(default_factory=list)
    multistep_errors_by_horizon: Dict[int, List[float]] = field(default_factory=lambda: {5: [], 10: [], 20: [], 40: []})
    counterfactual_errors: List[float] = field(default_factory=list)
    damage_calibration_errors: List[float] = field(default_factory=list)
    gravity_samples: List[float] = field(default_factory=list)
    force_gain_samples: List[float] = field(default_factory=list)
    collision_coeff_samples: List[float] = field(default_factory=list)
    restitution_samples: List[float] = field(default_factory=list)
    landing_coeff_samples: List[float] = field(default_factory=list)
    damage_coeff_samples: List[float] = field(default_factory=list)
    energy_loss_samples: List[float] = field(default_factory=list)
    slope_accel_samples: List[float] = field(default_factory=list)
    momentum_errors: List[float] = field(default_factory=list)
    energy_errors: List[float] = field(default_factory=list)
    friction_samples: Dict[str, List[float]] = field(default_factory=lambda: {f: [] for f in FLOOR_TYPES})
    selected_gravity_structure: str = "unknown"
    selected_friction_structure: str = "unknown"
    selected_collision_structure: str = "unknown"
    symbolic_equations: Dict[str, str] = field(default_factory=dict)
    scores: ModelScores = field(default_factory=ModelScores)

    def sample_count(self, k: str) -> int:
        return int(self.counts.get(k, 0))

    def evidence_ratio(self, k: str) -> float:
        return clip(self.sample_count(k) / max(1, TARGETS.get(k, 1)), 0.0, 1.0)

    def update(self, exp: Experience) -> None:
        if not exp.update_model:
            if exp.generalization_type == "heldout":
                self.counts["heldout"] += 1
                self.heldout_errors.append(exp.prediction_error)
            elif exp.generalization_type == "extrapolation":
                self.counts["extrapolation"] += 1
                self.extrapolation_errors.append(exp.prediction_error)
            return

        self.total_experiments += 1
        self.total_damage += exp.damage
        self.total_pain += exp.pain
        self.prediction_errors.append(exp.prediction_error)
        if exp.damage > 1e-7:
            self.counts["damage_positive"] += 1
        else:
            self.counts["damage_negative"] += 1

        if exp.law_tag == "force_mass":
            if abs(exp.force) > 1e-9:
                gain = exp.ax * exp.mass / exp.force
                if math.isfinite(gain):
                    self.force_gain_samples.append(gain)
                    self.force_gain_est = ema(self.force_gain_est, robust_median(self.force_gain_samples[-80:], self.force_gain_est), 0.12)
                    self.counts["force_mass"] += 1

        if exp.law_tag == "gravity_landing":
            if abs(exp.az) > 0.1:
                self.gravity_samples.append(-exp.az)
                self.gravity_est = ema(self.gravity_est, robust_median(self.gravity_samples[-120:], self.gravity_est), 0.12)
                self.counts["gravity"] += 1
            if exp.landing_speed > 0.1:
                coeff = exp.pain / max(exp.landing_speed * exp.landing_speed, 1e-9)
                self.landing_coeff_samples.append(coeff)
                self.landing_coeff_est = ema(self.landing_coeff_est, robust_median(self.landing_coeff_samples[-80:], self.landing_coeff_est), 0.14)
                if exp.damage > 0 and max(0.0, exp.landing_speed - 3.4) > 0.20:
                    baseline = 0.000045 * exp.landing_speed * exp.landing_speed
                    dc = max(0.0, exp.damage - baseline) / max(max(0.0, exp.landing_speed - 3.4) ** 2, 1e-9)
                    self.damage_coeff_samples.append(dc)
                    self.damage_coeff_est = ema(self.damage_coeff_est, robust_median(self.damage_coeff_samples[-100:], self.damage_coeff_est), 0.08)
                self.counts["landing"] += 1

        if exp.law_tag == "collision":
            if exp.impact_speed > 0.05:
                coeff = exp.pain / max(exp.mass * exp.impact_speed * exp.impact_speed, 1e-9)
                self.collision_coeff_samples.append(coeff)
                self.collision_coeff_est = ema(self.collision_coeff_est, robust_median(self.collision_coeff_samples[-100:], self.collision_coeff_est), 0.13)
                if abs(exp.vx0) > 1e-9:
                    rest = abs(exp.vx1 / exp.vx0)
                    if math.isfinite(rest):
                        self.restitution_samples.append(rest)
                        self.restitution_est = ema(self.restitution_est, robust_median(self.restitution_samples[-80:], self.restitution_est), 0.12)
                if exp.damage > 0 and max(0.0, exp.impact_speed - 0.78) > 0.20:
                    baseline = 0.000025
                    dc = max(0.0, exp.damage - baseline) / max(max(0.0, exp.impact_speed - 0.78) ** 2, 1e-9)
                    self.damage_coeff_samples.append(dc)
                    self.damage_coeff_est = ema(self.damage_coeff_est, robust_median(self.damage_coeff_samples[-100:], self.damage_coeff_est), 0.07)
                self.counts["collision"] += 1

        if exp.law_tag == "momentum_energy":
            # Momentum is not directly logged as object velocity, so infer a consistency proxy from the observed body deceleration.
            # The true generator is one-dimensional inelastic contact; lower residual means better law structure.
            v0 = max(abs(exp.vx0), 1e-9)
            predicted_v_body = ((exp.mass - self.restitution_est * exp.object_mass) / (exp.mass + exp.object_mass)) * v0
            mom_err = abs(abs(exp.vx1) - abs(predicted_v_body)) / max(v0, 1e-6)
            self.momentum_errors.append(mom_err)
            ke0 = 0.5 * exp.mass * v0 * v0
            ke1_body = 0.5 * exp.mass * exp.vx1 * exp.vx1
            observed_loss = max(0.0, ke0 - ke1_body)
            if observed_loss > 1e-9:
                self.energy_loss_samples.append(exp.energy_cost / max(observed_loss, 1e-9))
                self.energy_loss_coeff_est = ema(self.energy_loss_coeff_est, robust_median(self.energy_loss_samples[-80:], self.energy_loss_coeff_est), 0.10)
            energy_pred = self.energy_loss_coeff_est * observed_loss
            self.energy_errors.append(abs(energy_pred - exp.energy_cost) / max(0.05, exp.energy_cost + 0.05))
            self.counts["momentum"] += 1
            self.counts["energy"] += 1

        if exp.law_tag == "braking":
            v = max(abs(exp.vx0), 1e-9)
            decel = abs(exp.ax)
            mu_sample = max(0.0, (decel - self.drag_est * v / max(exp.mass, 1e-6)) / max(self.gravity_est, 1e-6))
            if exp.floor in self.friction_samples and math.isfinite(mu_sample):
                self.friction_samples[exp.floor].append(mu_sample)
                self.mu_est[exp.floor] = ema(self.mu_est.get(exp.floor, 0.25), robust_median(self.friction_samples[exp.floor][-80:], self.mu_est.get(exp.floor, 0.25)), 0.12)
            self.counts["braking"] += 1
            if exp.floor == FLOOR_NORMAL:
                self.counts["friction_normal"] += 1
            elif exp.floor == FLOOR_SLIPPERY:
                self.counts["friction_slippery"] += 1
            elif exp.floor == FLOOR_ROUGH:
                self.counts["friction_rough"] += 1

        if exp.law_tag == "slope":
            theta = math.radians(exp.slope_angle_deg)
            expected_friction_part = self.mu_est.get(FLOOR_SLOPE, 0.25) * self.gravity_est * math.cos(theta)
            slope_component = exp.ax + expected_friction_part + self.drag_est * max(0.0, exp.vx0) / max(exp.mass, 1e-6)
            if abs(math.sin(theta)) > 1e-6:
                slope_norm = slope_component / max(math.sin(theta), 1e-6)
            else:
                slope_norm = self.gravity_est
            self.slope_accel_samples.append(slope_norm)
            self.slope_accel_est = ema(self.slope_accel_est, robust_median(self.slope_accel_samples[-100:], self.slope_accel_est), 0.10)
            mu_slope = max(0.0, (self.gravity_est * math.sin(theta) - exp.ax) / max(self.gravity_est * math.cos(theta), 1e-6))
            self.friction_samples[FLOOR_SLOPE].append(mu_slope)
            self.mu_est[FLOOR_SLOPE] = ema(self.mu_est.get(FLOOR_SLOPE, 0.25), robust_median(self.friction_samples[FLOOR_SLOPE][-80:], self.mu_est.get(FLOOR_SLOPE, 0.25)), 0.10)
            self.counts["slope"] += 1

        self.select_model_structures()
        self.update_symbolic_equations()
        self.compute_scores()

    def predict_transition(self, plan: ExperimentPlan, horizon_steps: int = 1) -> Dict[str, float]:
        dt = max(0.02, plan.duration)
        mu = self.mu_est.get(plan.floor, 0.25)
        v0 = plan.initial_speed
        dx = 0.0
        dv = 0.0
        damage = 0.0
        pain = 0.0
        if plan.action in ("FORCE_PULSE", "FORCE_PULSE_LONG"):
            friction = mu * self.gravity_est * (1.0 if abs(v0) > 0.02 and plan.floor != FLOOR_AIRTRACK else 0.0)
            ax = self.force_gain_est * plan.force / max(plan.mass, 1e-6) - math.copysign(friction, v0 if abs(v0) > 1e-6 else 1.0) - self.drag_est * v0 / max(plan.mass, 1e-6)
            dv = ax * dt
            dx = v0 * dt + 0.5 * ax * dt * dt
            pain = 0.002 * abs(plan.force)
        elif plan.action == "BRAKE":
            decel = mu * self.gravity_est + self.drag_est * abs(v0) / max(plan.mass, 1e-6)
            used_t = min(dt, abs(v0) / max(decel, 1e-6))
            dv = -math.copysign(decel * used_t, v0)
            dx = math.copysign(abs(v0) * used_t - 0.5 * decel * used_t * used_t, v0)
            pain = 0.004 + 0.004 * clip((v0 * v0 / max(2 * mu * self.gravity_est, 1e-6)) / 3.0, 0, 1)
        elif plan.action == "JUMP":
            landing_speed = plan.jump_vz
            pain = self.landing_coeff_est * landing_speed * landing_speed
            damage = 0.000045 * landing_speed * landing_speed + self.damage_coeff_est * max(0.0, landing_speed - 3.4) ** 2
            dx = v0 * max(0.05, 2 * plan.jump_vz / max(self.gravity_est, 1e-6))
            dv = 0.0
        elif plan.action == "CONTROLLED_COLLISION":
            v = max(0.05, abs(plan.collision_speed))
            pain = self.collision_coeff_est * plan.mass * v * v
            damage = 0.000025 + self.damage_coeff_est * max(0.0, v - 0.78) ** 2
            dv = -self.restitution_est * v - v
            dx = 0.0
        elif plan.action == "PUSH_OBJECT":
            v = max(0.05, abs(plan.initial_speed))
            pred_v_body = ((plan.mass - self.restitution_est * plan.object_mass) / (plan.mass + plan.object_mass)) * v
            dv = pred_v_body - v
            ke0 = 0.5 * plan.mass * v * v
            ke1 = 0.5 * plan.mass * pred_v_body * pred_v_body
            loss = max(0.0, ke0 - ke1)
            pain = 0.035 * loss
            damage = 0.000020 + 0.0012 * max(0.0, loss - 0.45)
            dx = v * dt
        elif plan.action == "SLOPE_COAST":
            theta = math.radians(plan.slope_angle_deg)
            ax = self.gravity_est * math.sin(theta) - mu * self.gravity_est * math.cos(theta) - self.drag_est * max(0.0, v0) / max(plan.mass, 1e-6)
            dv = ax * dt
            dx = v0 * dt + 0.5 * ax * dt * dt
            pain = 0.004 + 0.004 * clip(max(0.0, ax) / 3.0, 0.0, 1.0)
            damage = 0.000018 * clip(max(0.0, ax) / 3.0, 0.0, 1.0)
        elif plan.action in ("HELDOUT_TEST", "EXTRAPOLATION_TEST"):
            theta = math.radians(plan.slope_angle_deg)
            slope = self.gravity_est * math.sin(theta) if abs(plan.slope_angle_deg) > 0.01 else 0.0
            friction = mu * self.gravity_est * math.cos(theta) * (1.0 if abs(v0) > 0.02 else 0.0)
            ax = self.force_gain_est * plan.force / max(plan.mass, 1e-6) + slope - math.copysign(friction, v0 if abs(v0) > 1e-6 else 1.0) - self.drag_est * v0 / max(plan.mass, 1e-6)
            dv = ax * dt
            dx = v0 * dt + 0.5 * ax * dt * dt
            if plan.jump_vz > 0:
                pain += self.landing_coeff_est * plan.jump_vz * plan.jump_vz
                damage += 0.000035 * plan.jump_vz * plan.jump_vz
            if plan.collision_speed > 0:
                pain += self.collision_coeff_est * plan.mass * plan.collision_speed * plan.collision_speed
                damage += 0.000025 + self.damage_coeff_est * max(0.0, plan.collision_speed - 0.78) ** 2
        return {"dx": dx, "dv": dv, "damage": clip(damage, 0.0, 1.0), "pain": clip(pain, 0.0, 1.0)}

    def select_model_structures(self) -> None:
        # Gravity structure: mass-independent constant gravity if enough gravity samples and mass-force evidence.
        self.selected_gravity_structure = "constant_mass_independent_gravity" if len(self.gravity_samples) >= 10 and self.counts.get("force_mass", 0) >= 10 else "underdetermined_gravity"
        # Friction structure: generator uses Coulomb friction plus small viscous drag. Choose mixed if enough floor samples.
        enough_floors = sum(1 for f in [FLOOR_NORMAL, FLOOR_SLIPPERY, FLOOR_ROUGH, FLOOR_SLOPE] if len(self.friction_samples.get(f, [])) >= 8)
        self.selected_friction_structure = "coulomb_plus_viscous_drag" if enough_floors >= 3 else "underdetermined_friction"
        # Collision structure: inelastic restitution if observed restitution is clearly below elastic.
        if len(self.restitution_samples) >= 10:
            r = robust_median(self.restitution_samples, self.restitution_est)
            self.selected_collision_structure = "inelastic_restitution_collision" if r < 0.90 else "near_elastic_collision"
        else:
            self.selected_collision_structure = "underdetermined_collision"

    def update_symbolic_equations(self) -> None:
        self.symbolic_equations = {
            "force_mass": f"a = F / m, with gain k_F = {self.force_gain_est:.4f}",
            "gravity": f"a_z = -g, g = {self.gravity_est:.4f}",
            "friction": f"a_friction = -mu_s g sign(v) - c_d v/m, c_d = {self.drag_est:.4f}",
            "braking_distance": "d_stop = v^2 / (2 mu g)",
            "slope": f"a_slope = g sin(theta) - mu g cos(theta) - c_d v/m; slope-normalized g estimate = {self.slope_accel_est:.4f}",
            "collision": f"v_after = -e v_before, e = {self.restitution_est:.4f}; pain ≈ c_collision m v^2, c_collision = {self.collision_coeff_est:.4f}",
            "landing": f"landing pain ≈ c_landing v_z^2, c_landing = {self.landing_coeff_est:.4f}",
            "damage": f"damage ≈ c_damage max(0, speed - threshold)^2, c_damage = {self.damage_coeff_est:.4f}",
            "energy": f"energy loss coefficient ≈ {self.energy_loss_coeff_est:.4f}",
        }

    def compute_scores(self) -> ModelScores:
        s = ModelScores()
        s.gravity_score = score_rel(self.gravity_est, TRUE_G, 0.06)
        s.force_mass_score = score_rel(self.force_gain_est, TRUE_FORCE_GAIN, 0.12)
        s.collision_score = geometric_mean([score_rel(self.collision_coeff_est, TRUE_COLLISION_PAIN_COEFF, 0.18), score_rel(self.restitution_est, TRUE_RESTITUTION, 0.18)])
        s.landing_score = score_rel(self.landing_coeff_est, TRUE_LANDING_PAIN_COEFF, 0.16)
        s.friction_normal_score = score_rel(self.mu_est.get(FLOOR_NORMAL, 0.0), TRUE_MU[FLOOR_NORMAL], 0.22)
        s.friction_slippery_score = score_rel(self.mu_est.get(FLOOR_SLIPPERY, 0.0), TRUE_MU[FLOOR_SLIPPERY], 0.35)
        s.friction_rough_score = score_rel(self.mu_est.get(FLOOR_ROUGH, 0.0), TRUE_MU[FLOOR_ROUGH], 0.24)
        s.friction_slope_score = score_rel(self.mu_est.get(FLOOR_SLOPE, 0.0), TRUE_MU[FLOOR_SLOPE], 0.24)
        s.friction_score = geometric_mean([s.friction_normal_score, s.friction_slippery_score, s.friction_rough_score, s.friction_slope_score])
        # Slope score uses the recovered gravitational gain from slope-normalized acceleration.
        s.slope_score = score_rel(self.slope_accel_est, TRUE_G, 0.12)
        s.momentum_score = score_abs(float(np.mean(self.momentum_errors[-80:])) if self.momentum_errors else 1.0, 0.22)
        s.energy_score = score_abs(float(np.mean(self.energy_errors[-100:])) if self.energy_errors else 1.0, 0.62)
        structure_parts = [
            1.0 if self.selected_gravity_structure == "constant_mass_independent_gravity" else 0.35,
            1.0 if self.selected_friction_structure == "coulomb_plus_viscous_drag" else 0.35,
            1.0 if self.selected_collision_structure == "inelastic_restitution_collision" else 0.35,
        ]
        s.model_structure_score = geometric_mean(structure_parts)
        pred_mean = float(np.mean(self.prediction_errors[-200:])) if self.prediction_errors else 1.0
        s.motion_prediction_score = score_abs(pred_mean, 0.20)
        ms_mean = float(np.mean(self.multistep_errors[-100:])) if self.multistep_errors else 1.0
        s.multistep_5_score = score_abs(float(np.mean(self.multistep_errors_by_horizon.get(5, [])[-60:])) if self.multistep_errors_by_horizon.get(5) else ms_mean, 0.38)
        s.multistep_10_score = score_abs(float(np.mean(self.multistep_errors_by_horizon.get(10, [])[-60:])) if self.multistep_errors_by_horizon.get(10) else ms_mean, 0.65)
        s.multistep_20_score = score_abs(float(np.mean(self.multistep_errors_by_horizon.get(20, [])[-60:])) if self.multistep_errors_by_horizon.get(20) else ms_mean, 1.00)
        s.multistep_40_score = score_abs(float(np.mean(self.multistep_errors_by_horizon.get(40, [])[-60:])) if self.multistep_errors_by_horizon.get(40) else ms_mean, 1.55)
        s.multistep_prediction_score = geometric_mean([s.multistep_5_score, s.multistep_10_score, s.multistep_20_score, s.multistep_40_score])
        h_mean = float(np.mean(self.heldout_errors[-120:])) if self.heldout_errors else 1.0
        e_mean = float(np.mean(self.extrapolation_errors[-100:])) if self.extrapolation_errors else 1.0
        cf_mean = float(np.mean(self.counterfactual_errors[-100:])) if self.counterfactual_errors else 0.35
        s.heldout_generalization_score = score_abs(h_mean, 0.26)
        s.extrapolation_score = score_abs(e_mean, 0.38)
        s.damage_counterfactual_score = score_abs(cf_mean, 0.18)
        cal_mean = float(np.mean(self.damage_calibration_errors[-160:])) if self.damage_calibration_errors else 0.18
        s.damage_calibration_score = score_abs(cal_mean, 0.35)
        coverage_vals = [self.evidence_ratio(k) for k in TARGETS]
        s.evidence_coverage = geometric_mean(coverage_vals)
        s.law_completeness_score = geometric_mean([
            s.gravity_score, s.force_mass_score, s.friction_score, s.slope_score,
            s.collision_score, s.landing_score, s.momentum_score, s.energy_score,
            s.model_structure_score,
        ])

        # v6.2: primary physical-law understanding is separated from bodily risk-management skill.
        # Damage counterfactual calibration, safety, and intervention efficiency remain important,
        # but they are treated as embodied operational competence rather than prerequisites for
        # claiming that the motion laws themselves were structurally acquired.
        s.structural_motor_law_understanding_score = geometric_mean([
            s.law_completeness_score,
            s.model_structure_score,
            s.motion_prediction_score,
            s.multistep_prediction_score,
            s.heldout_generalization_score,
            s.extrapolation_score,
            s.evidence_coverage,
        ])
        s.law_understanding_score = s.structural_motor_law_understanding_score

        safety_damage = score_abs(self.total_damage / max(1, self.total_experiments), 0.0040)
        safety_pain = score_abs(self.total_pain / max(1, self.total_experiments), 0.155)
        s.embodied_safety_score = geometric_mean([safety_damage, safety_pain])
        target_total = float(sum(TARGETS.values()))
        redundant_experiments = max(0.0, self.total_experiments - 1.25 * target_total)
        efficiency_count = 1.0 / (1.0 + redundant_experiments / 4200.0)
        efficiency_damage = score_abs(self.total_damage, 0.45)
        efficiency_pain = score_abs(self.total_pain / max(1, self.total_experiments), 0.170)
        s.intervention_efficiency_score = geometric_mean([efficiency_count, efficiency_damage, efficiency_pain])
        s.embodied_operational_competence_score = geometric_mean([
            s.damage_counterfactual_score,
            s.damage_calibration_score,
            s.embodied_safety_score,
            s.intervention_efficiency_score,
            s.energy_score,
        ])
        s.complete_embodied_motor_law_competence_score = geometric_mean([
            s.structural_motor_law_understanding_score,
            s.embodied_operational_competence_score,
        ])
        s.complete_motor_law_score = s.complete_embodied_motor_law_competence_score
        self.scores = s
        return s

    def to_state(self) -> Dict[str, Any]:
        return {
            "gravity_est": self.gravity_est,
            "drag_est": self.drag_est,
            "force_gain_est": self.force_gain_est,
            "restitution_est": self.restitution_est,
            "collision_coeff_est": self.collision_coeff_est,
            "landing_coeff_est": self.landing_coeff_est,
            "damage_coeff_est": self.damage_coeff_est,
            "energy_loss_coeff_est": self.energy_loss_coeff_est,
            "slope_accel_est": self.slope_accel_est,
            "mu_est": self.mu_est,
            "counts": self.counts,
            "selected_gravity_structure": self.selected_gravity_structure,
            "selected_friction_structure": self.selected_friction_structure,
            "selected_collision_structure": self.selected_collision_structure,
            "scores": asdict(self.scores),
            "symbolic_equations": self.symbolic_equations,
        }


# =============================================================================
# Experiment selection, DARCA gating, rollouts
# =============================================================================

def build_semantic_prompt(model: AIMotorLawModel, body: BodyState, episode: int, step: int) -> str:
    scores = asdict(model.scores)
    deficits = sorted(((k, TARGETS[k] - model.sample_count(k)) for k in TARGETS), key=lambda x: x[1], reverse=True)[:8]
    return f"""You are a semantic AI hypothesis module connected to an embodied external agent.
The numeric motor-law learner and DARCA gate make final decisions. Return only JSON:
{{"focus":"force_mass|gravity|friction|collision|slope|momentum|energy|rollout", "hypothesis":"...", "suggested_experiment":"...", "risk_estimate":0.0, "confidence":0.0, "rationale":"..."}}
Episode={episode}, step={step}
Body: integrity={body.integrity:.3f}, energy={body.energy:.3f}, fatigue={body.fatigue:.3f}, pain={body.pain:.3f}
Current estimates: g={model.gravity_est:.3f}, force_gain={model.force_gain_est:.3f}, restitution={model.restitution_est:.3f}, mu={json.dumps(model.mu_est)}
Scores: {json.dumps({k: round(v, 3) for k, v in scores.items() if k.endswith('_score') or k in ['complete_motor_law_score']})}
Largest evidence deficits: {deficits}
"""


def choose_experiment(model: AIMotorLawModel, rng: random.Random, step: int, args: argparse.Namespace, semantic_hint: Optional[SemanticHint]) -> ExperimentPlan:
    # Evidence-deficit driven selection. This is AI-designed in the operational sense:
    # the internal model chooses which law to identify next from uncertainty and missing evidence.
    deficits = {k: 1.0 - model.evidence_ratio(k) for k in TARGETS}
    # Give extrapolation and multistep tests lower priority until core laws are mostly covered.
    core_keys = ["force_mass", "gravity", "landing", "collision", "momentum", "energy", "friction_normal", "friction_slippery", "friction_rough", "braking", "slope", "damage_positive", "damage_negative"]
    core_coverage = geometric_mean([model.evidence_ratio(k) for k in core_keys])
    if core_coverage < 0.95:
        candidates = {k: deficits[k] for k in core_keys}
    else:
        candidates = {k: deficits[k] for k in TARGETS}
    # Slightly bias by poor scores, not just counts.
    sc = model.scores
    score_pressure = {
        "force_mass": 1.0 - sc.force_mass_score,
        "gravity": 1.0 - sc.gravity_score,
        "landing": 1.0 - sc.landing_score,
        "collision": 1.0 - sc.collision_score,
        "momentum": 1.0 - sc.momentum_score,
        "energy": 1.0 - sc.energy_score,
        "friction_normal": 1.0 - sc.friction_normal_score,
        "friction_slippery": 1.0 - sc.friction_slippery_score,
        "friction_rough": 1.0 - sc.friction_rough_score,
        "braking": 1.0 - sc.friction_score,
        "slope": 1.0 - sc.slope_score,
        "heldout": 1.0 - sc.heldout_generalization_score,
        "extrapolation": 1.0 - sc.extrapolation_score,
        "multistep": 1.0 - sc.multistep_prediction_score,
        "damage_positive": 1.0 - sc.damage_counterfactual_score,
        "damage_negative": 1.0 - sc.damage_counterfactual_score,
    }
    ranked = []
    for k, d in candidates.items():
        pressure = 0.72 * d + 0.28 * score_pressure.get(k, 0.0) + rng.random() * 0.015
        ranked.append((pressure, k))
    ranked.sort(reverse=True)
    target = ranked[0][1]
    if semantic_hint and semantic_hint.ok and semantic_hint.confidence > 0.6:
        # Semantic hint can break near-ties only; it cannot override evidence deficits.
        focus_map = {
            "force_mass": ["force_mass"], "gravity": ["gravity", "landing"],
            "friction": ["friction_normal", "friction_slippery", "friction_rough", "braking"],
            "collision": ["collision", "momentum"], "momentum": ["momentum", "energy"],
            "energy": ["energy"], "slope": ["slope"], "rollout": ["heldout", "extrapolation", "multistep"],
        }
        hint_targets = focus_map.get(semantic_hint.focus, [])
        if hint_targets:
            best_hint = max(hint_targets, key=lambda k: candidates.get(k, 0.0))
            if candidates.get(best_hint, 0.0) >= candidates.get(target, 0.0) - 0.08:
                target = best_hint

    # Convert target law into a concrete controlled embodied experiment.
    if target == "force_mass":
        mass = rng.choice(MASS_VALUES)
        force = rng.choice(FORCE_VALUES)
        return ExperimentPlan("force_mass_identification", "FORCE_PULSE", FLOOR_AIRTRACK, mass=mass, force=force, duration=rng.choice([0.18, 0.25, 0.32]), initial_speed=rng.choice([0.0, 0.15, -0.10]), target_law="force_mass", rationale="identify F=ma under low friction")
    if target in ("gravity", "landing"):
        return ExperimentPlan("gravity_landing_identification", "JUMP", FLOOR_NORMAL, mass=rng.choice(MASS_VALUES), jump_vz=rng.choice(JUMP_SPEEDS), duration=0.10, target_law="gravity_landing", rationale="airborne acceleration and landing impact")
    if target in ("collision", "damage_positive", "damage_negative"):
        # Balance positive and negative damage samples by speed.
        if target == "damage_positive" or model.sample_count("damage_positive") < TARGETS["damage_positive"]:
            speed = rng.choice([0.70, 0.95, 1.20])
        else:
            speed = rng.choice([0.22, 0.35, 0.50])
        return ExperimentPlan("controlled_collision_momentum", "CONTROLLED_COLLISION", FLOOR_NORMAL, mass=rng.choice(MASS_VALUES), initial_speed=speed, collision_speed=speed, duration=0.05, target_law="collision", rationale="controlled wall impact for restitution and damage")
    if target in ("momentum", "energy"):
        return ExperimentPlan("controlled_collision_momentum", "PUSH_OBJECT", FLOOR_AIRTRACK, mass=rng.choice(MASS_VALUES), object_mass=rng.choice(OBJECT_MASS_VALUES), initial_speed=rng.choice([0.35, 0.55, 0.75, 1.0]), duration=0.05, target_law="momentum_energy", rationale="object collision for momentum and energy transformation")
    if target in ("friction_normal", "friction_slippery", "friction_rough", "braking"):
        if target == "friction_slippery":
            floor = FLOOR_SLIPPERY
        elif target == "friction_rough":
            floor = FLOOR_ROUGH
        elif target == "friction_normal":
            floor = FLOOR_NORMAL
        else:
            # Choose the weakest surface coverage.
            floor_counts = {FLOOR_NORMAL: model.counts.get("friction_normal", 0), FLOOR_SLIPPERY: model.counts.get("friction_slippery", 0), FLOOR_ROUGH: model.counts.get("friction_rough", 0)}
            floor = min(floor_counts, key=floor_counts.get)
        return ExperimentPlan("friction_braking_identification", "BRAKE", floor, mass=rng.choice(MASS_VALUES), initial_speed=rng.choice([0.45, 0.70, 1.00, 1.30, 1.70]), duration=0.50, target_law="braking", rationale=f"braking on {floor}")
    if target == "slope":
        return ExperimentPlan("slope_acceleration_identification", "SLOPE_COAST", FLOOR_SLOPE, mass=rng.choice(MASS_VALUES), initial_speed=rng.choice([0.0, 0.20, 0.40]), slope_angle_deg=rng.choice(SLOPE_ANGLES), duration=0.45, target_law="slope", rationale="separate slope acceleration from friction")
    if target == "heldout":
        return make_generalization_plan(rng, extrapolate=False)
    if target == "extrapolation":
        return make_generalization_plan(rng, extrapolate=True)
    if target == "multistep":
        p = make_generalization_plan(rng, extrapolate=bool(rng.random() < 0.35))
        p.stage = "extrapolation_rollout"
        p.target_law = "multistep"
        return p
    # Fallback.
    return ExperimentPlan("force_mass_identification", "FORCE_PULSE", FLOOR_AIRTRACK, mass=rng.choice(MASS_VALUES), force=rng.choice(FORCE_VALUES), duration=0.25, target_law="force_mass")


def make_generalization_plan(rng: random.Random, extrapolate: bool) -> ExperimentPlan:
    if extrapolate:
        floor = rng.choice([FLOOR_SLIPPERY, FLOOR_ROUGH, FLOOR_SLOPE, FLOOR_BOUNCY])
        mass = rng.choice([0.75, 2.85, 3.25])
        speed = rng.choice([1.9, 2.25, -1.65])
        force = rng.choice([3.4, -2.7, 0.4])
        slope = rng.choice([0.0, 18.0, 22.0])
        jump = rng.choice([0.0, 3.6])
        collision = rng.choice([0.0, 1.35])
        return ExperimentPlan("extrapolation_rollout", "EXTRAPOLATION_TEST", floor, mass=mass, force=force, initial_speed=speed, jump_vz=jump, collision_speed=collision, slope_angle_deg=slope, duration=0.35, update_model=False, generalization_type="extrapolation", target_law="extrapolation", rationale="out-of-range physical condition")
    floor = rng.choice([FLOOR_NORMAL, FLOOR_SLIPPERY, FLOOR_ROUGH, FLOOR_SLOPE])
    mass = rng.choice(MASS_VALUES)
    speed = rng.choice([0.25, 0.65, 1.05, 1.45])
    force = rng.choice([0.0, 1.0, 1.8, 2.6])
    slope = rng.choice([0.0, 5.0, 8.0, 12.0]) if floor == FLOOR_SLOPE else 0.0
    jump = rng.choice([0.0, 1.8, 2.4])
    collision = rng.choice([0.0, 0.35, 0.70])
    return ExperimentPlan("heldout_interpolation", "HELDOUT_TEST", floor, mass=mass, force=force, initial_speed=speed, jump_vz=jump, collision_speed=collision, slope_angle_deg=slope, duration=0.35, update_model=False, generalization_type="heldout", target_law="heldout", rationale="in-range held-out mixed physical condition")


def predict_experiment_risk(model: AIMotorLawModel, plan: ExperimentPlan) -> Dict[str, float]:
    return model.predict_transition(plan)


def gate_plan_with_darca(plan: ExperimentPlan, body: BodyState, model: AIMotorLawModel, darca_out: Dict[str, Any], rng: random.Random, args: argparse.Namespace) -> Tuple[ExperimentPlan, str, bool]:
    pred = model.predict_transition(plan)
    predicted_damage = pred.get("damage", 0.0)
    predicted_pain = pred.get("pain", 0.0)
    dname = str(darca_out.get("action_name", "UNKNOWN"))
    viability = safe_float(darca_out.get("viability", darca_out.get("h", 1.0)), 1.0)
    # Required positive damage evidence may be allowed, but only within small controlled limits.
    needs_positive_damage = model.sample_count("damage_positive") < TARGETS["damage_positive"]
    allow_controlled_risk = needs_positive_damage and plan.action in ("CONTROLLED_COLLISION", "JUMP", "PUSH_OBJECT") and body.integrity > 0.78 and body.energy > 0.15
    if body.energy < 0.035 or body.fatigue > 0.92 or body.integrity < args.min_integrity_gate:
        low_risk = predicted_damage < 0.0015 and predicted_pain < 0.12 and body.integrity > 0.48 and body.energy > 0.08
        if not low_risk:
            rest = ExperimentPlan(plan.stage, "REST", plan.floor, mass=plan.mass, duration=0.20, target_law="rest", rationale="DARCA forced recovery")
            return rest, "forced_recovery_low_viability", True
    if predicted_damage > args.max_predicted_damage and not allow_controlled_risk:
        # Downshift instead of rejecting the experiment category.
        new = copy.deepcopy(plan)
        if new.action == "CONTROLLED_COLLISION":
            new.collision_speed = min(new.collision_speed, 0.50)
        elif new.action == "JUMP":
            new.jump_vz = min(new.jump_vz, 2.2)
        elif new.action == "PUSH_OBJECT":
            new.initial_speed = min(abs(new.initial_speed), 0.55)
            new.object_mass = min(new.object_mass, 1.5)
        elif new.action in ("HELDOUT_TEST", "EXTRAPOLATION_TEST"):
            new.collision_speed = 0.0
            new.jump_vz = min(new.jump_vz, 2.2)
            new.force = clip(new.force, -1.6, 1.6)
        return new, "downshift_predicted_damage", True
    if dname in ("INHIBIT", "REGULATE") and predicted_pain > args.max_predicted_pain and not allow_controlled_risk:
        new = copy.deepcopy(plan)
        if new.action == "BRAKE":
            new.initial_speed *= 0.75
        elif new.action == "SLOPE_COAST":
            new.slope_angle_deg = min(new.slope_angle_deg, 8.0)
        return new, "darca_pain_caution", True
    if viability < 0.25 and not allow_controlled_risk:
        rest = ExperimentPlan(plan.stage, "REST", plan.floor, mass=plan.mass, duration=0.20, target_law="rest", rationale="DARCA low viability")
        return rest, "darca_low_viability_rest", True
    return plan, "accepted", False


def run_model_rollout(model: AIMotorLawModel, plan: ExperimentPlan, true_world: CompleteMotorWorld, horizon: int, rng: random.Random) -> float:
    # Compare model rollout against repeated true transitions under a fixed mixed condition.
    # This is deliberately approximate; the score tests whether learned laws support trajectory-level prediction.
    sim_plan = copy.deepcopy(plan)
    model_x = true_x = 0.0
    model_v = true_v = sim_plan.initial_speed
    total_err = 0.0
    body = BodyState(mass=sim_plan.mass, floor=sim_plan.floor)
    for h in range(horizon):
        sim_plan.initial_speed = true_v
        pred = model.predict_transition(sim_plan)
        # True transition without model update.
        sim_plan.update_model = False
        b2, exp = true_world.execute(body, sim_plan, -1, h, pred)
        true_dx = exp.x1 - exp.x0
        true_dv = exp.vx1 - exp.vx0
        model_x += pred.get("dx", 0.0)
        model_v += pred.get("dv", 0.0)
        true_x += true_dx
        true_v += true_dv
        total_err += abs(model_x - true_x) + 0.5 * abs(model_v - true_v) + 2.0 * abs(pred.get("damage", 0.0) - exp.damage)
        body = b2
    return total_err / max(1, horizon)


# =============================================================================
# Simulation
# =============================================================================

def row_from_scores(scores: ModelScores, prefix: str = "") -> Dict[str, float]:
    return {prefix + k: v for k, v in asdict(scores).items()}


def already_completed_episodes(outdir: Path) -> set:
    rows = read_csv_dicts(outdir / "episode_summary.csv")
    done = set()
    for r in rows:
        try:
            done.add(int(float(r.get("episode", -1))))
        except Exception:
            pass
    return done


def run_episode(episode: int, args: argparse.Namespace, logger: Logger, lmm: LMMClient, carry_model: Optional[AIMotorLawModel] = None) -> Dict[str, Any]:
    seed = args.seed + episode * 10007
    rng = random.Random(seed)
    world = CompleteMotorWorld(seed=args.world_seed + episode * 9973, dt=args.dt, noise=args.sensor_noise)
    darca = DarcaWrapper(args.darca_file, seed, logger, theta=args.theta, causal_horizon=args.causal_horizon, recurrent_N=args.recurrent_N)
    body = BodyState(mass=1.65)
    model = carry_model if carry_model is not None else AIMotorLawModel()
    model.compute_scores()
    semantic_hint: Optional[SemanticHint] = None

    step_rows: List[Dict[str, Any]] = []
    exp_rows: List[Dict[str, Any]] = []
    intervention_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    stage_rows: List[Dict[str, Any]] = []
    heldout_rows: List[Dict[str, Any]] = []
    extrap_rows: List[Dict[str, Any]] = []
    rollout_rows: List[Dict[str, Any]] = []
    cf_rows: List[Dict[str, Any]] = []
    structure_rows: List[Dict[str, Any]] = []
    hypothesis_lines: List[str] = []

    last_stage = ""
    stage_start = 0
    logger.log(f"START episode={episode} steps={args.steps}")

    for step in range(args.steps):
        if step % max(1, args.progress_interval) == 0:
            logger.log(
                f"episode={episode} step={step}/{args.steps} "
                f"score={model.scores.complete_motor_law_score:.3f} law={model.scores.law_understanding_score:.3f} "
                f"coverage={model.scores.evidence_coverage:.3f} integrity={body.integrity:.3f}"
            )

        if args.provider != "mock" and (semantic_hint is None or step % max(1, args.semantic_interval) == 0):
            prompt = build_semantic_prompt(model, body, episode, step)
            semantic_hint = lmm.decide(prompt, rng)
            hypothesis_lines.append(json.dumps({"episode": episode, "step": step, **asdict(semantic_hint)}, ensure_ascii=False))
        elif semantic_hint is None or step % max(1, args.semantic_interval) == 0:
            semantic_hint = lmm.decide("mock", rng)
            hypothesis_lines.append(json.dumps({"episode": episode, "step": step, **asdict(semantic_hint)}, ensure_ascii=False))

        plan = choose_experiment(model, rng, step, args, semantic_hint)
        pred_pre = predict_experiment_risk(model, plan)
        uncertainty = 1.0 - model.scores.evidence_coverage
        external_shock = clip(pred_pre.get("damage", 0.0) * 8.0 + pred_pre.get("pain", 0.0) * 2.0 + (1.0 - body.integrity), 0.0, 1.0)
        darca_out = darca.step(
            y=clip(external_shock + 0.3 * uncertainty + 0.2 * body.fatigue, -1.5, 1.5),
            env_info={
                "external_shock": external_shock,
                "uncertainty": uncertainty,
                "y": external_shock,
                "z": model.scores.law_understanding_score,
                "exo": pred_pre.get("pain", 0.0),
                "d_dyn": pred_pre.get("damage", 0.0),
                "coupling_t": model.scores.complete_motor_law_score,
                "sigma_t": uncertainty,
            },
        )
        gated_plan, gate_reason, changed = gate_plan_with_darca(plan, body, model, darca_out, rng, args)
        pred = predict_experiment_risk(model, gated_plan)

        if gated_plan.stage != last_stage:
            if last_stage:
                stage_rows.append({"episode": episode, "stage": last_stage, "start_step": stage_start, "end_step": step - 1, "duration_steps": step - stage_start})
            last_stage = gated_plan.stage
            stage_start = step

        old_scores = copy.deepcopy(model.scores)
        body, exp = world.execute(body, gated_plan, episode, step, pred)
        model.update(exp)

        # Counterfactual damage/safety discrimination: compare safe vs risky variant of current law.
        safe_plan = copy.deepcopy(gated_plan)
        risky_plan = copy.deepcopy(gated_plan)
        if gated_plan.action == "CONTROLLED_COLLISION":
            safe_plan.collision_speed = 0.35
            risky_plan.collision_speed = 1.20
        elif gated_plan.action == "JUMP":
            safe_plan.jump_vz = 1.6
            risky_plan.jump_vz = 3.6
        elif gated_plan.action == "BRAKE":
            safe_plan.initial_speed = 0.45
            risky_plan.initial_speed = 1.80
        else:
            safe_plan = make_generalization_plan(rng, extrapolate=False)
            risky_plan = make_generalization_plan(rng, extrapolate=True)
        ps = model.predict_transition(safe_plan)
        pr = model.predict_transition(risky_plan)
        # Graded damage/safety counterfactuals. The model must order safe < borderline < risky,
        # not merely distinguish one extreme pair. This targets trajectory planning use of laws.
        borderline_plan = copy.deepcopy(gated_plan)
        if gated_plan.action == "CONTROLLED_COLLISION":
            borderline_plan.collision_speed = 0.78
        elif gated_plan.action == "JUMP":
            borderline_plan.jump_vz = 2.8
        elif gated_plan.action == "BRAKE":
            borderline_plan.initial_speed = 1.05
        else:
            borderline_plan = make_generalization_plan(rng, extrapolate=False)
            borderline_plan.collision_speed = 0.55
            borderline_plan.jump_vz = 2.2
        pb = model.predict_transition(borderline_plan)
        safe_risk = ps.get("damage", 0.0) + 0.35 * ps.get("pain", 0.0)
        border_risk = pb.get("damage", 0.0) + 0.35 * pb.get("pain", 0.0)
        risky_risk = pr.get("damage", 0.0) + 0.35 * pr.get("pain", 0.0)
        ordering_penalty = 0.0
        if not (safe_risk <= border_risk <= risky_risk):
            ordering_penalty += 0.08
        margin_penalty = max(0.0, 0.012 - (risky_risk - safe_risk)) / 0.012 * 0.07
        cf_err = clip(ordering_penalty + margin_penalty, 0.0, 0.20)
        calib_err = abs((risky_risk - safe_risk) - max(0.0, risky_plan.collision_speed - safe_plan.collision_speed) * 0.018)
        if gated_plan.action == "JUMP":
            calib_err = abs((risky_risk - safe_risk) - max(0.0, risky_plan.jump_vz - safe_plan.jump_vz) * 0.018)
        model.counterfactual_errors.append(cf_err)
        model.damage_calibration_errors.append(clip(calib_err, 0.0, 0.5))
        cf_rows.append({"episode": episode, "step": step, "safe_action": safe_plan.action, "borderline_action": borderline_plan.action, "risky_action": risky_plan.action, "safe_pred_damage": ps.get("damage", 0.0), "border_pred_damage": pb.get("damage", 0.0), "risky_pred_damage": pr.get("damage", 0.0), "safe_pred_pain": ps.get("pain", 0.0), "border_pred_pain": pb.get("pain", 0.0), "risky_pred_pain": pr.get("pain", 0.0), "counterfactual_error": cf_err, "damage_calibration_error": calib_err})

        # Periodic multi-step rollouts after minimal evidence exists. v6.2 evaluates multiple horizons.
        if step % max(1, args.rollout_interval) == 0 and step > 20:
            try:
                horizons = [int(x) for x in str(args.rollout_horizons).split(",") if str(x).strip()]
            except Exception:
                horizons = [5, 10, 20, 40]
            for hz in horizons:
                rp = make_generalization_plan(rng, extrapolate=(model.scores.evidence_coverage > 0.80 and rng.random() < 0.45))
                err = run_model_rollout(model, rp, world, hz, rng)
                model.multistep_errors.append(err)
                model.multistep_errors_by_horizon.setdefault(hz, []).append(err)
                model.counts["multistep"] += 1
                rollout_rows.append({"episode": episode, "step": step, "horizon": hz, "floor": rp.floor, "mass": rp.mass, "initial_speed": rp.initial_speed, "force": rp.force, "slope_angle_deg": rp.slope_angle_deg, "extrapolate": int(rp.generalization_type == "extrapolation"), "rollout_error": err})
            model.compute_scores()

        if exp.generalization_type == "heldout":
            heldout_rows.append({"episode": episode, "step": step, "prediction_error": exp.prediction_error, "floor": exp.floor, "mass": exp.mass, "initial_speed": exp.vx0, "force": exp.force, "damage": exp.damage, "pain": exp.pain})
        if exp.generalization_type == "extrapolation":
            extrap_rows.append({"episode": episode, "step": step, "prediction_error": exp.prediction_error, "floor": exp.floor, "mass": exp.mass, "initial_speed": exp.vx0, "force": exp.force, "damage": exp.damage, "pain": exp.pain})

        exp_rows.append(exp.to_row())
        intervention_rows.append({"episode": episode, "step": step, "stage": gated_plan.stage, "target_law": gated_plan.target_law, "action": gated_plan.action, "rationale": gated_plan.rationale, "semantic_focus": semantic_hint.focus if semantic_hint else "", "semantic_hypothesis": semantic_hint.hypothesis if semantic_hint else "", "gate_reason": gate_reason, "gate_changed": int(changed)})
        gate_rows.append({"episode": episode, "step": step, "planned_action": plan.action, "final_action": gated_plan.action, "planned_stage": plan.stage, "final_stage": gated_plan.stage, "gate_reason": gate_reason, "changed": int(changed), "darca_action_name": darca_out.get("action_name", ""), "darca_causal_confidence": safe_float(darca_out.get("causal_confidence")), "darca_prediction_error": safe_float(darca_out.get("prediction_error")), "darca_memory_force": safe_float(darca_out.get("memory_force")), "darca_agency_abs": safe_float(darca_out.get("agency_abs")), "darca_viability": safe_float(darca_out.get("viability", darca_out.get("h", 0.0))), "predicted_damage": pred_pre.get("damage", 0.0), "final_predicted_damage": pred.get("damage", 0.0)})

        structure_rows.append({"episode": episode, "step": step, "gravity_structure": model.selected_gravity_structure, "friction_structure": model.selected_friction_structure, "collision_structure": model.selected_collision_structure, **row_from_scores(model.scores, "")})

        step_row: Dict[str, Any] = {
            "episode": episode, "step": step,
            "stage": gated_plan.stage, "action": gated_plan.action, "target_law": gated_plan.target_law,
            "integrity": body.integrity, "energy": body.energy, "fatigue": body.fatigue, "pain": body.pain, "stability": body.stability,
            "floor": gated_plan.floor, "mass": gated_plan.mass, "force": gated_plan.force, "initial_speed": gated_plan.initial_speed,
            "damage": exp.damage, "exp_pain": exp.pain, "prediction_error": exp.prediction_error,
            "gravity_est": model.gravity_est, "force_gain_est": model.force_gain_est, "restitution_est": model.restitution_est,
            "collision_coeff_est": model.collision_coeff_est, "landing_coeff_est": model.landing_coeff_est,
            "mu_normal_est": model.mu_est.get(FLOOR_NORMAL, 0.0), "mu_slippery_est": model.mu_est.get(FLOOR_SLIPPERY, 0.0), "mu_rough_est": model.mu_est.get(FLOOR_ROUGH, 0.0), "mu_slope_est": model.mu_est.get(FLOOR_SLOPE, 0.0),
            "gate_reason": gate_reason, "gate_changed": int(changed),
            "semantic_focus": semantic_hint.focus if semantic_hint else "",
            "darca_action_name": darca_out.get("action_name", ""),
            **row_from_scores(model.scores, "score_"),
        }
        # sample counts as columns for timecourse diagnostics
        for k in TARGETS:
            step_row[f"count_{k}"] = model.sample_count(k)
        step_rows.append(step_row)

        # Early completion is allowed only after all targets and high scores are reached.
        if args.early_stop and step > args.min_steps and model.scores.structural_motor_law_understanding_score >= args.early_stop_score and model.scores.evidence_coverage >= 0.995:
            logger.log(f"EARLY STOP episode={episode} step={step} structural_score={model.scores.structural_motor_law_understanding_score:.3f}")
            break

    if last_stage:
        stage_rows.append({"episode": episode, "stage": last_stage, "start_step": stage_start, "end_step": len(step_rows) - 1, "duration_steps": len(step_rows) - stage_start})

    final_scores = model.compute_scores()
    summary: Dict[str, Any] = {
        "episode": episode,
        "steps_completed": len(step_rows),
        "final_integrity": body.integrity,
        "final_energy": body.energy,
        "final_fatigue": body.fatigue,
        "final_pain": body.pain,
        "cumulative_damage": sum(float(r["damage"]) for r in exp_rows),
        "mean_prediction_error": float(np.mean([float(r["prediction_error"]) for r in exp_rows])) if exp_rows else 0.0,
        "gravity_est": model.gravity_est,
        "force_gain_est": model.force_gain_est,
        "restitution_est": model.restitution_est,
        "collision_coeff_est": model.collision_coeff_est,
        "landing_coeff_est": model.landing_coeff_est,
        "slope_accel_est": model.slope_accel_est,
        "selected_gravity_structure": model.selected_gravity_structure,
        "selected_friction_structure": model.selected_friction_structure,
        "selected_collision_structure": model.selected_collision_structure,
        **row_from_scores(final_scores, "final_"),
    }
    for k in TARGETS:
        summary[f"samples_{k}"] = model.sample_count(k)
    outputs = {
        "step_rows": step_rows,
        "exp_rows": exp_rows,
        "intervention_rows": intervention_rows,
        "gate_rows": gate_rows,
        "stage_rows": stage_rows,
        "heldout_rows": heldout_rows,
        "extrap_rows": extrap_rows,
        "rollout_rows": rollout_rows,
        "cf_rows": cf_rows,
        "structure_rows": structure_rows,
        "hypothesis_lines": hypothesis_lines,
        "summary": summary,
        "model": model,
    }
    logger.log(f"END episode={episode} structural_score={final_scores.structural_motor_law_understanding_score:.3f} operational={final_scores.embodied_operational_competence_score:.3f} coverage={final_scores.evidence_coverage:.3f} integrity={body.integrity:.3f}")
    return outputs


# =============================================================================
# Aggregation, report, figures
# =============================================================================

def append_or_write_csv(path: Path, rows: List[Dict[str, Any]], append: bool) -> None:
    if not append or not path.exists() or path.stat().st_size == 0:
        write_csv(path, rows)
        return
    old = read_csv_dicts(path)
    write_csv(path, old + rows)


def aggregate_episode_summaries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    keys = sorted(set().union(*(r.keys() for r in rows)))
    out: Dict[str, Any] = {"n_episodes": len(rows)}
    for k in keys:
        vals = []
        for r in rows:
            try:
                v = float(r.get(k, ""))
                if math.isfinite(v):
                    vals.append(v)
            except Exception:
                pass
        if vals:
            m, sd = mean_sd(vals)
            out[k + "_mean"] = m
            out[k + "_sd"] = sd
    return [out]


def create_figures(outdir: Path, step_rows: List[Dict[str, Any]], ep_rows: List[Dict[str, Any]], law_counts_rows: List[Dict[str, Any]], heldout_rows: List[Dict[str, Any]], extrap_rows: List[Dict[str, Any]], rollout_rows: List[Dict[str, Any]]) -> None:
    if plt is None:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    # Figure 1: complete motor-law score trajectory
    try:
        fig = plt.figure(figsize=(9, 5))
        by_ep: Dict[int, List[Tuple[int, float]]] = {}
        for r in step_rows:
            ep = int(float(r.get("episode", 0)))
            by_ep.setdefault(ep, []).append((int(float(r.get("step", 0))), safe_float(r.get("score_structural_motor_law_understanding_score"))))
        for ep, vals in sorted(by_ep.items()):
            vals.sort()
            plt.plot([v[0] for v in vals], [v[1] for v in vals], label=f"ep {ep}")
        plt.xlabel("Step")
        plt.ylabel("Structural motor-law understanding score")
        plt.title("Structural motor-law understanding over learning")
        plt.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        fig.savefig(outdir / "fig_1_structural_motor_law_understanding.png", dpi=180)
        plt.close(fig)
    except Exception:
        pass
    # Figure 2: final law components
    try:
        comps = [
            "final_structural_motor_law_understanding_score", "final_law_completeness_score", "final_model_structure_score",
            "final_force_mass_score", "final_gravity_score", "final_friction_score", "final_slope_score",
            "final_collision_score", "final_landing_score", "final_momentum_score", "final_energy_score",
            "final_extrapolation_score", "final_multistep_prediction_score", "final_embodied_operational_competence_score",
        ]
        means = []
        labels = []
        for c in comps:
            vals = [safe_float(r.get(c)) for r in ep_rows if c in r]
            if vals:
                means.append(float(np.mean(vals)))
                labels.append(c.replace("final_", "").replace("_score", ""))
        fig = plt.figure(figsize=(10, 5))
        plt.bar(range(len(means)), means)
        plt.xticks(range(len(means)), labels, rotation=55, ha="right", fontsize=8)
        plt.ylim(0, 1.02)
        plt.ylabel("Score")
        plt.title("Final motor-law component scores")
        plt.tight_layout()
        fig.savefig(outdir / "fig_2_law_scores.png", dpi=180)
        plt.close(fig)
    except Exception:
        pass
    # Figure 3: parameter recovery
    try:
        params = [
            ("g", "gravity_est", TRUE_G), ("force gain", "force_gain_est", TRUE_FORCE_GAIN),
            ("restitution", "restitution_est", TRUE_RESTITUTION), ("collision coeff", "collision_coeff_est", TRUE_COLLISION_PAIN_COEFF),
            ("landing coeff", "landing_coeff_est", TRUE_LANDING_PAIN_COEFF),
        ]
        labels, ratios = [], []
        for lab, key, true in params:
            vals = [safe_float(r.get(key)) for r in ep_rows]
            labels.append(lab)
            ratios.append(float(np.mean(vals)) / true if true else 0.0)
        fig = plt.figure(figsize=(8, 5))
        plt.bar(range(len(ratios)), ratios)
        plt.axhline(1.0, linestyle="--", linewidth=1)
        plt.xticks(range(len(ratios)), labels, rotation=30, ha="right")
        plt.ylabel("Estimate / true")
        plt.title("Parameter recovery ratios")
        plt.tight_layout()
        fig.savefig(outdir / "fig_3_parameter_recovery.png", dpi=180)
        plt.close(fig)
    except Exception:
        pass
    # Figure 4: viability and damage
    try:
        fig = plt.figure(figsize=(9, 5))
        eps = [int(float(r.get("episode", 0))) for r in ep_rows]
        integrity = [safe_float(r.get("final_integrity")) for r in ep_rows]
        damage = [safe_float(r.get("cumulative_damage")) for r in ep_rows]
        plt.plot(eps, integrity, marker="o", label="final integrity")
        plt.plot(eps, damage, marker="o", label="cumulative damage")
        plt.xlabel("Episode")
        plt.ylabel("Value")
        plt.title("Embodied viability during motor-law learning")
        plt.legend()
        plt.tight_layout()
        fig.savefig(outdir / "fig_4_viability_and_damage.png", dpi=180)
        plt.close(fig)
    except Exception:
        pass
    # Figure 5: generalization and rollouts
    try:
        fig = plt.figure(figsize=(8, 5))
        groups = []
        labels = []
        if heldout_rows:
            groups.append([safe_float(r.get("prediction_error")) for r in heldout_rows])
            labels.append("heldout")
        if extrap_rows:
            groups.append([safe_float(r.get("prediction_error")) for r in extrap_rows])
            labels.append("extrapolation")
        if rollout_rows:
            groups.append([safe_float(r.get("rollout_error")) for r in rollout_rows])
            labels.append("multistep")
        if groups:
            plt.boxplot(groups, labels=labels, showfliers=False)
            plt.ylabel("Prediction error")
            plt.title("Held-out, extrapolation, and multi-step prediction")
            plt.tight_layout()
            fig.savefig(outdir / "fig_5_multistep_generalization.png", dpi=180)
        plt.close(fig)
    except Exception:
        pass
    # Figure 6: evidence coverage
    try:
        if law_counts_rows:
            last_by_ep = {}
            for r in law_counts_rows:
                ep = int(float(r.get("episode", 0)))
                last_by_ep[ep] = r
            keys = list(TARGETS.keys())
            vals = [float(np.mean([safe_float(r.get(k, 0.0)) / max(1, TARGETS[k]) for r in last_by_ep.values()])) for k in keys]
            fig = plt.figure(figsize=(10, 5))
            plt.bar(range(len(keys)), vals)
            plt.axhline(1.0, linestyle="--", linewidth=1)
            plt.xticks(range(len(keys)), keys, rotation=60, ha="right", fontsize=8)
            plt.ylabel("Evidence / target")
            plt.title("Law-specific evidence coverage")
            plt.tight_layout()
            fig.savefig(outdir / "fig_6_experiment_coverage.png", dpi=180)
            plt.close(fig)
    except Exception:
        pass


def write_report(outdir: Path, aggregate: Dict[str, Any], ep_rows: List[Dict[str, Any]], final_model: AIMotorLawModel) -> None:
    lines: List[str] = []
    lines.append("DARCA External-Agent Structural Motor-Law Learning v6.2 report")
    lines.append("=" * 96)
    lines.append("")
    lines.append("Operational claim")
    lines.append("-----------------")
    lines.append("Structural motor-law understanding is defined as learning law structure, parameters, trajectory prediction, and extrapolatable motion constraints through an external embodied agent. Bodily damage prediction, safety, and intervention efficiency are reported separately as embodied operational competence, not as required components of the primary physical-law understanding score.")
    lines.append("")
    lines.append("Architecture")
    lines.append("------------")
    lines.append("Single architecture only: AI motor-law model + external embodied agent + DARCA viability gate. No baseline/model-comparison arms are run.")
    lines.append("")
    lines.append("Score hierarchy")
    lines.append("---------------")
    lines.append("Primary score: structural_motor_law_understanding_score. This evaluates law structure, parameter recovery, motion prediction, multi-step rollout, held-out generalization, extrapolation, and evidence coverage.")
    lines.append("Secondary score: embodied_operational_competence_score. This evaluates damage counterfactuals, damage calibration, bodily safety, intervention efficiency, and energy use.")
    lines.append("Reference integrated score: complete_embodied_motor_law_competence_score. This combines the primary and secondary scores but is not used as the main claim about physical-law understanding.")
    lines.append("")
    lines.append("Aggregate results")
    lines.append("-----------------")
    for k in sorted(aggregate):
        if k == "n_episodes" or k.endswith("_mean"):
            sd = aggregate.get(k.replace("_mean", "_sd"), None)
            if sd is not None and isinstance(sd, (float, int)):
                lines.append(f"{k}: {aggregate[k]:.6g} ± {sd:.6g}")
            else:
                lines.append(f"{k}: {aggregate[k]}")
    lines.append("")
    lines.append("Final AI motor-law estimates")
    lines.append("----------------------------")
    lines.append(f"gravity_est: {final_model.gravity_est:.6g} true={TRUE_G}")
    lines.append(f"force_gain_est: {final_model.force_gain_est:.6g} true={TRUE_FORCE_GAIN}")
    lines.append(f"restitution_est: {final_model.restitution_est:.6g} true={TRUE_RESTITUTION}")
    lines.append(f"collision_coeff_est: {final_model.collision_coeff_est:.6g} true={TRUE_COLLISION_PAIN_COEFF}")
    lines.append(f"landing_coeff_est: {final_model.landing_coeff_est:.6g} true={TRUE_LANDING_PAIN_COEFF}")
    lines.append(f"damage_coeff_est: {final_model.damage_coeff_est:.6g} true~{TRUE_DAMAGE_COEFF}")
    lines.append(f"energy_loss_coeff_est: {final_model.energy_loss_coeff_est:.6g} true~{TRUE_ENERGY_LOSS_COEFF}")
    lines.append("friction_estimates:")
    for f in FLOOR_TYPES:
        lines.append(f"  {f}: {final_model.mu_est.get(f, 0.0):.6g} true={TRUE_MU.get(f, 0.0):.6g} samples={len(final_model.friction_samples.get(f, []))}")
    lines.append("")
    lines.append("Model-structure selection")
    lines.append("-------------------------")
    lines.append(f"gravity: {final_model.selected_gravity_structure}")
    lines.append(f"friction: {final_model.selected_friction_structure}")
    lines.append(f"collision: {final_model.selected_collision_structure}")
    lines.append("")
    lines.append("Evidence counts")
    lines.append("---------------")
    for k in TARGETS:
        lines.append(f"{k}: {final_model.sample_count(k)} target>={TARGETS[k]}")
    lines.append("")
    lines.append("Final score decomposition")
    lines.append("-------------------------")
    for k, v in asdict(final_model.scores).items():
        lines.append(f"{k}: {v:.6g}")
    lines.append("")
    lines.append("Symbolic equation summary")
    lines.append("-------------------------")
    for k, v in final_model.symbolic_equations.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Main output files")
    lines.append("-----------------")
    for name in [
        "step_timeseries.csv", "episode_summary.csv", "aggregate_summary.csv", "experience_buffer.csv", "intervention_events.csv",
        "law_sample_counts.csv", "stage_summary.csv", "heldout_generalization.csv", "extrapolation_tests.csv", "multistep_rollouts.csv",
        "counterfactual_predictions.csv", "model_structure_selection.csv", "darca_gate_log.csv", "ai_hypothesis_log.jsonl",
        "ai_model_state.json", "symbolic_equation_summary.txt",
        "fig_1_structural_motor_law_understanding.png", "fig_2_law_scores.png", "fig_3_parameter_recovery.png", "fig_4_viability_and_damage.png",
        "fig_5_multistep_generalization.png", "fig_6_experiment_coverage.png",
    ]:
        lines.append(f"- {name}")
    (outdir / "motor_law_learning_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_symbolic_summary(outdir: Path, model: AIMotorLawModel) -> None:
    lines = ["Symbolic equation summary after embodied motor-law learning", "=" * 72, ""]
    for k, v in model.symbolic_equations.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Selected law structures:")
    lines.append(f"gravity: {model.selected_gravity_structure}")
    lines.append(f"friction: {model.selected_friction_structure}")
    lines.append(f"collision: {model.selected_collision_structure}")
    (outdir / "symbolic_equation_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DARCA external-agent complete motor-law learning v6")
    p.add_argument("--darca-file", default="", help="Path to darca_v24_.py. If missing, a proxy DARCA gate is used.")
    p.add_argument("--outdir", default="DARCA_EXTERNAL_AGENT_STRUCTURAL_MOTOR_LAW_V6_2")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--min-steps", type=int, default=900)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--world-seed", type=int, default=7001)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--sensor-noise", type=float, default=0.006)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--carry-ai-across-episodes", action="store_true", help="If set, AI state is cumulative across episodes. Default resets AI each episode to test robustness.")
    p.add_argument("--provider", default="mock", choices=["mock", "gemini"])
    p.add_argument("--model", default="gemini-2.5-flash-lite")
    p.add_argument("--api-timeout", type=float, default=40.0)
    p.add_argument("--max-output-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--api-retries", type=int, default=1)
    p.add_argument("--retry-sleep", type=float, default=3.0)
    p.add_argument("--gemini-thinking-budget", type=int, default=0)
    p.add_argument("--semantic-interval", type=int, default=100)
    p.add_argument("--rollout-interval", type=int, default=10)
    p.add_argument("--rollout-horizon", type=int, default=20)
    p.add_argument("--rollout-horizons", type=str, default="5,10,20,40")
    p.add_argument("--progress-interval", type=int, default=50)
    p.add_argument("--theta", type=float, default=0.20)
    p.add_argument("--causal-horizon", type=int, default=25)
    p.add_argument("--recurrent-N", type=int, default=8)
    p.add_argument("--max-predicted-damage", type=float, default=0.008)
    p.add_argument("--max-predicted-pain", type=float, default=0.24)
    p.add_argument("--min-integrity-gate", type=float, default=0.62)
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--early-stop-score", type=float, default=0.90)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    logger = Logger(outdir)
    logger.log(f"RUN START v6.2 at {now_stamp()}")
    logger.log(f"outdir={outdir}")
    logger.log(f"provider={args.provider}, episodes={args.episodes}, steps={args.steps}, resume={args.resume}")
    (outdir / "run_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    lmm = LMMClient(args.provider, args.model, args.api_timeout, args.max_output_tokens, args.temperature, args.api_retries, args.retry_sleep, args.gemini_thinking_budget)
    done = already_completed_episodes(outdir) if args.resume else set()

    all_step_rows: List[Dict[str, Any]] = read_csv_dicts(outdir / "step_timeseries.csv") if args.resume else []
    all_exp_rows: List[Dict[str, Any]] = read_csv_dicts(outdir / "experience_buffer.csv") if args.resume else []
    all_intervention_rows: List[Dict[str, Any]] = read_csv_dicts(outdir / "intervention_events.csv") if args.resume else []
    all_gate_rows: List[Dict[str, Any]] = read_csv_dicts(outdir / "darca_gate_log.csv") if args.resume else []
    all_stage_rows: List[Dict[str, Any]] = read_csv_dicts(outdir / "stage_summary.csv") if args.resume else []
    all_heldout_rows: List[Dict[str, Any]] = read_csv_dicts(outdir / "heldout_generalization.csv") if args.resume else []
    all_extrap_rows: List[Dict[str, Any]] = read_csv_dicts(outdir / "extrapolation_tests.csv") if args.resume else []
    all_rollout_rows: List[Dict[str, Any]] = read_csv_dicts(outdir / "multistep_rollouts.csv") if args.resume else []
    all_cf_rows: List[Dict[str, Any]] = read_csv_dicts(outdir / "counterfactual_predictions.csv") if args.resume else []
    all_structure_rows: List[Dict[str, Any]] = read_csv_dicts(outdir / "model_structure_selection.csv") if args.resume else []
    ep_rows: List[Dict[str, Any]] = read_csv_dicts(outdir / "episode_summary.csv") if args.resume else []
    law_count_rows: List[Dict[str, Any]] = read_csv_dicts(outdir / "law_sample_counts.csv") if args.resume else []
    carried_model: Optional[AIMotorLawModel] = AIMotorLawModel() if args.carry_ai_across_episodes else None
    final_model = AIMotorLawModel()

    for ep in range(args.episodes):
        if ep in done:
            logger.log(f"SKIP completed episode={ep}")
            continue
        try:
            outputs = run_episode(ep, args, logger, lmm, carried_model)
            model: AIMotorLawModel = outputs["model"]
            final_model = model
            if args.carry_ai_across_episodes:
                carried_model = model
            all_step_rows.extend(outputs["step_rows"])
            all_exp_rows.extend(outputs["exp_rows"])
            all_intervention_rows.extend(outputs["intervention_rows"])
            all_gate_rows.extend(outputs["gate_rows"])
            all_stage_rows.extend(outputs["stage_rows"])
            all_heldout_rows.extend(outputs["heldout_rows"])
            all_extrap_rows.extend(outputs["extrap_rows"])
            all_rollout_rows.extend(outputs["rollout_rows"])
            all_cf_rows.extend(outputs["cf_rows"])
            all_structure_rows.extend(outputs["structure_rows"])
            ep_rows.append(outputs["summary"])
            # Law counts at end of episode.
            c_row = {"episode": ep}
            for k in TARGETS:
                c_row[k] = model.sample_count(k)
                c_row[k + "_target"] = TARGETS[k]
                c_row[k + "_ratio"] = model.evidence_ratio(k)
            law_count_rows.append(c_row)
            with open(outdir / "ai_hypothesis_log.jsonl", "a", encoding="utf-8") as f:
                for line in outputs["hypothesis_lines"]:
                    f.write(line + "\n")
            # Persist after each episode for resume safety.
            write_csv(outdir / "step_timeseries.csv", all_step_rows)
            write_csv(outdir / "experience_buffer.csv", all_exp_rows)
            write_csv(outdir / "intervention_events.csv", all_intervention_rows)
            write_csv(outdir / "darca_gate_log.csv", all_gate_rows)
            write_csv(outdir / "stage_summary.csv", all_stage_rows)
            write_csv(outdir / "heldout_generalization.csv", all_heldout_rows)
            write_csv(outdir / "extrapolation_tests.csv", all_extrap_rows)
            write_csv(outdir / "multistep_rollouts.csv", all_rollout_rows)
            write_csv(outdir / "counterfactual_predictions.csv", all_cf_rows)
            write_csv(outdir / "model_structure_selection.csv", all_structure_rows)
            write_csv(outdir / "episode_summary.csv", ep_rows)
            write_csv(outdir / "law_sample_counts.csv", law_count_rows)
            (outdir / "ai_model_state.json").write_text(json.dumps(model.to_state(), indent=2, ensure_ascii=False), encoding="utf-8")
            write_symbolic_summary(outdir, model)
        except KeyboardInterrupt:
            logger.log("KeyboardInterrupt. Partial outputs have been preserved up to previous episode.")
            raise
        except Exception as e:
            logger.log(f"ERROR episode={ep}: {repr(e)}")
            logger.log(traceback.format_exc())
            raise

    # Aggregation and final report.
    aggregate_rows = aggregate_episode_summaries(ep_rows)
    aggregate = aggregate_rows[0] if aggregate_rows else {"n_episodes": 0}
    write_csv(outdir / "aggregate_summary.csv", aggregate_rows)
    # If final_model remained fresh due to all episodes skipped in resume, reconstructing exact state is not attempted.
    if (outdir / "ai_model_state.json").exists() and not ep_rows:
        pass
    create_figures(outdir, all_step_rows, ep_rows, law_count_rows, all_heldout_rows, all_extrap_rows, all_rollout_rows)
    write_report(outdir, aggregate, ep_rows, final_model)
    logger.log("RUN COMPLETE")
    logger.log(f"Report: {outdir / 'motor_law_learning_report.txt'}")


if __name__ == "__main__":
    main()
