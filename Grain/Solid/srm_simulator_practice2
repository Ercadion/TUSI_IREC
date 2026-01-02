
"""
srm_design_report.py
====================

One-command SRM design pre-check:
- Load design numbers from an Excel sheet (loose label matching).
- Run nominal 0D SRM simulation + generate plots.
- Run Monte Carlo uncertainty analysis to produce P05/P50/P95 bands.
- Output a Markdown report + CSV/JSON artifacts.

Requires: numpy, pandas, matplotlib
Also requires: srm_simulator.py (same folder or on PYTHONPATH)

Usage (examples)
----------------
1) Basic (using your existing design sheet):
   python srm_design_report.py --design_xlsx "엔진설계_250912(그레인변경사항적용).xlsx" ^
     --expr "circle(x,y,0,0,3.974359)" --xlim -10 10 --ylim -10 10 ^
     --eps 5.6 --out_dir out --name my_engine

2) Provide/override any value:
   python srm_design_report.py --design_xlsx design.xlsx --expr "circle(x,y,0,0,4.2)" --eps 5.6 ^
     --override "cstar_m_s=885.22" --override "At_m2=3.4856e-5"

3) Add targets / constraints (simple PASS/FAIL flags in report):
   python srm_design_report.py ... --target_F_mean 200 --max_Pc_peak 60

Notes
-----
- a,n are not typically in your design sheet. Provide them via:
  --a_SI and --n
  or supply a_mm and n with unit hint:
     rdot[mm/s] = a_mm * (Pc[MPa])^n  --> a_SI = a_mm*1e-3*(1e-6)^n
  Use: --a_mm_MPa 3.5 --n 0.35

- For complex ports, MC is approximated by a fast scaling surrogate by default.
  For simple center circle ports, MC runs full fast 0D+circle model.

Outputs
-------
(out_dir)/(name)_REPORT.md
(out_dir)/(name)_Pc_nominal.png
(out_dir)/(name)_Thrust_nominal.png
(out_dir)/(name)_MC_hist.png
(out_dir)/(name)_MC_percentiles.csv
(out_dir)/(name)_MC_samples.csv
(out_dir)/(name)_summary.json
"""

from __future__ import annotations
import argparse
import os
import json
import math
import re
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the simulator module (expects srm_simulator.py nearby)
try:
    import srm_simulator as sim
except Exception as e:
    # Try local directory import
    import importlib.util, sys
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "srm_simulator.py")
    spec = importlib.util.spec_from_file_location("srm_simulator", path)
    sim = importlib.util.module_from_spec(spec)
    sys.modules["srm_simulator"] = sim
    spec.loader.exec_module(sim)

_CIRCLE_RE = re.compile(r"circle\(\s*x\s*,\s*y\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*\)\s*$")

def a_mmMPa_to_SI(a_mm: float, n: float) -> float:
    """rdot[mm/s] = a_mm*(Pc[MPa])^n  -> rdot[m/s] = a_SI*(Pc[Pa])^n"""
    return (a_mm * 1e-3) * (1e-6 ** n)

def load_design_from_excel(path: str) -> Dict[str, float]:
    """
    Loose parser: finds Korean labels and returns adjacent cell values.
    Keys returned (if found):
      D_grain_m, core_diam_m, L_grain_m, rho_kg_m3, cstar_m_s, At_m2
    """
    raw = pd.read_excel(path, sheet_name=0, header=None)

    def find(label: str) -> Optional[float]:
        for r in range(raw.shape[0]):
            for c in range(raw.shape[1]):
                v = raw.iat[r, c]
                if isinstance(v, str) and label in v:
                    if c+1 < raw.shape[1]:
                        try:
                            return float(raw.iat[r, c+1])
                        except Exception:
                            pass
        return None

    D_mm = find("그레인 직경")
    core_mm = find("코어 직경")
    L_mm = find("그레인 길이")
    rho_gcc = find("그레인 밀도")
    cstar = find("c*")
    At_mm2 = find("노즐목 면적")

    out: Dict[str, float] = {}
    if D_mm is not None: out["D_grain_m"] = D_mm/1000.0
    if core_mm is not None: out["core_diam_m"] = core_mm/1000.0
    if L_mm is not None: out["L_grain_m"] = L_mm/1000.0
    if rho_gcc is not None: out["rho_kg_m3"] = rho_gcc*1000.0
    if cstar is not None: out["cstar_m_s"] = cstar
    if At_mm2 is not None: out["At_m2"] = At_mm2*1e-6
    return out

def parse_overrides(items: List[str]) -> Dict[str, float]:
    """
    --override key=value (repeatable). Example: --override "At_m2=3.48e-5"
    """
    out = {}
    for s in items or []:
        if "=" not in s:
            raise ValueError(f"override must be key=value, got: {s}")
        k, v = s.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out

def is_simple_center_circle(exprs: List[str]):
    if len(exprs) != 1:
        return None
    m = _CIRCLE_RE.match(exprs[0].strip())
    if not m:
        return None
    cx, cy, r = map(float, m.groups())
    if abs(cx) > 1e-9 or abs(cy) > 1e-9:
        return None
    return r

def run_circle_ode(
    t_end: float, dt: float,
    D_grain: float, L_grain: float,
    r0_coord: float, xlim: Tuple[float,float],
    rho: float, a_SI: float, n: float, cstar: float,
    At0: float, eps: float, Cd: float, gamma: float,
    V_dead: float = 0.0, pa: float = 101325.0,
    Tc: float = 3000.0, Rgas: float = 300.0,
    erosion_k: float = 0.0, erosion_m: float = 0.0
) -> Dict[str, np.ndarray]:
    """Fast 0D+circle model (no grid)."""
    scale = D_grain / (xlim[1]-xlim[0])
    Rgrain = D_grain/2.0
    r = r0_coord * scale

    steps = int(math.ceil(t_end/dt)) + 1
    t = np.linspace(0.0, t_end, steps)
    Pc = max(pa, 1e5)

    Pc_bar = np.zeros_like(t)
    Thrust = np.zeros_like(t)

    def At(Pc, tt):
        if erosion_k <= 0:
            return At0
        if erosion_m == 0:
            return At0 + erosion_k*tt
        return At0 + erosion_k*(Pc**erosion_m)*tt

    for i, ti in enumerate(t):
        # Geometry
        Pport = 2*math.pi*r
        Ab = Pport*L_grain
        Aport = math.pi*r*r
        V = Aport*L_grain + V_dead

        rdot = a_SI*(Pc**n) if Pc>0 else 0.0
        mgen = rho*rdot*Ab
        mout = Cd*At(Pc, ti)*Pc/cstar
        dVdt = (Pport*rdot)*L_grain

        dPdt = (Rgas*Tc/max(V,1e-12))*(mgen-mout) - (Pc/max(V,1e-12))*dVdt
        Pc = max(Pc + dPdt*dt, 0.0)

        # Advance radius
        r = min(Rgrain, r + rdot*dt)
        if r >= Rgrain:
            Pc = pa

        cf = sim.nozzle_Cf(gamma, Pc, pa, eps)
        Thrust[i] = cf * Pc * At(Pc, ti)
        Pc_bar[i] = Pc/1e5

    return {"t": t, "Pc_bar": Pc_bar, "Thrust_N": Thrust}

def lognormal_sample(rng, mu: float, rel: float) -> float:
    sigma = math.sqrt(math.log(1 + rel*rel))
    m = math.log(mu) - 0.5*sigma*sigma
    return float(rng.lognormal(m, sigma))

def make_design_report(
    exprs: List[str],
    xlim: Tuple[float,float], ylim: Tuple[float,float],
    D_grain: float, L_grain: float,
    rho: float, cstar: float,
    At0: float, eps: float,
    a_SI: float, n: float,
    Cd: float = 0.95, gamma: float = 1.2,
    V_dead: float = 0.0, pa: float = 101325.0,
    t_end: float = 3.0, dt: float = 0.002,
    # uncertainty
    mc_draws: int = 800,
    rel_a: float = 0.25, rel_cstar: float = 0.06, rel_Cd: float = 0.05, rel_At: float = 0.03, n_sd: float = 0.03,
    erosion_k: float = 0.0, erosion_m: float = 0.0, rel_erosion_k: float = 0.5,
    seed: int = 7,
    out_dir: str = "out",
    name: str = "design",
    # targets
    target_F_mean: Optional[float] = None,
    max_Pc_peak: Optional[float] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, name)

    # Nominal run
    r0_coord = is_simple_center_circle(exprs)
    if r0_coord is not None:
        ts = run_circle_ode(
            t_end=t_end, dt=dt, D_grain=D_grain, L_grain=L_grain,
            r0_coord=r0_coord, xlim=xlim,
            rho=rho, a_SI=a_SI, n=n, cstar=cstar,
            At0=At0, eps=eps, Cd=Cd, gamma=gamma,
            V_dead=V_dead, pa=pa, erosion_k=erosion_k, erosion_m=erosion_m
        )
        t = ts["t"]; Pc_bar = ts["Pc_bar"]; Thrust = ts["Thrust_N"]
    else:
        geom = sim.PortGeometry(exprs=exprs, xlim=xlim, ylim=ylim, D_grain=D_grain, grid=650)
        prop = sim.Propellant(rho=rho, a=a_SI, n=n, Tc=3000.0, Rgas=300.0)
        noz  = sim.Nozzle(At0=At0, eps=eps, Cd=Cd, gamma=gamma, erosion_k=erosion_k, erosion_m=erosion_m)
        motor = sim.Motor(geom=geom, prop=prop, noz=noz, cstar=cstar, L_grain=L_grain, V_dead=V_dead, pa=pa, use_ode=True)
        ts_full = sim.run_simulation(motor, t_end=t_end, dt=dt)
        t = ts_full["t"]; Pc_bar = ts_full["Pc_bar"]; Thrust = ts_full["Thrust_N"]

    Pc_peak = float(np.max(Pc_bar))
    F_peak  = float(np.max(Thrust))
    F_mean  = float(np.mean(Thrust))
    It      = float(np.trapz(Thrust, t))

    # PASS/FAIL
    pass_flags = {}
    if target_F_mean is not None:
        pass_flags["target_F_mean_met"] = bool(F_mean >= target_F_mean)
    if max_Pc_peak is not None:
        pass_flags["max_Pc_peak_ok"] = bool(Pc_peak <= max_Pc_peak)

    # Nominal plots
    pc_png = f"{prefix}_Pc_nominal.png"
    th_png = f"{prefix}_Thrust_nominal.png"
    plt.figure(figsize=(7,4.2)); plt.plot(t, Pc_bar); plt.xlabel("Time [s]"); plt.ylabel("Pc [bar]"); plt.grid(True); plt.tight_layout(); plt.savefig(pc_png, dpi=200); plt.close()
    plt.figure(figsize=(7,4.2)); plt.plot(t, Thrust); plt.xlabel("Time [s]"); plt.ylabel("Thrust [N]"); plt.grid(True); plt.tight_layout(); plt.savefig(th_png, dpi=200); plt.close()

    # Monte Carlo
    rng = np.random.default_rng(seed)
    samples = []
    mc_eff = mc_draws if r0_coord is not None else min(mc_draws, 200)  # safeguard for complex ports

    for _ in range(mc_eff):
        a_i = lognormal_sample(rng, a_SI, rel_a)
        c_i = lognormal_sample(rng, cstar, rel_cstar)
        Cd_i = min(1.0, max(0.5, lognormal_sample(rng, Cd, rel_Cd)))
        At_i = lognormal_sample(rng, At0, rel_At)
        n_i = float(rng.normal(n, n_sd))
        ek_i = erosion_k if erosion_k <= 0 else lognormal_sample(rng, erosion_k, rel_erosion_k)

        if r0_coord is not None:
            tsi = run_circle_ode(
                t_end=t_end, dt=dt, D_grain=D_grain, L_grain=L_grain,
                r0_coord=r0_coord, xlim=xlim,
                rho=rho, a_SI=a_i, n=n_i, cstar=c_i,
                At0=At_i, eps=eps, Cd=Cd_i, gamma=gamma,
                V_dead=V_dead, pa=pa, erosion_k=ek_i, erosion_m=erosion_m
            )
            Pc_pk = float(np.max(tsi["Pc_bar"]))
            F_pk  = float(np.max(tsi["Thrust_N"]))
            F_mn  = float(np.mean(tsi["Thrust_N"]))
            It_i  = float(np.trapz(tsi["Thrust_N"], tsi["t"]))
        else:
            # surrogate scaling (fast)
            scale = (a_i/a_SI) * (c_i/cstar) / (Cd_i/Cd) / (At_i/At0)
            Pc_scale = scale ** (1.0/(1.0-max(0.05, min(0.95, n_i))))
            Pc_pk = float(Pc_peak * Pc_scale)
            F_pk  = float(F_peak  * Pc_scale)
            F_mn  = float(F_mean  * Pc_scale)
            It_i  = float(It      * Pc_scale)

        samples.append((Pc_pk, F_pk, F_mn, It_i))

    sdf = pd.DataFrame(samples, columns=["Pc_peak_bar","F_peak_N","F_mean_N","It_Ns"])
    sdf_path = f"{prefix}_MC_samples.csv"
    sdf.to_csv(sdf_path, index=False)

    pct = sdf.quantile([0.05, 0.5, 0.95]).rename(index={0.05:"P05", 0.5:"P50", 0.95:"P95"})
    pct_path = f"{prefix}_MC_percentiles.csv"
    pct.to_csv(pct_path)

    hist_png = f"{prefix}_MC_hist.png"
    plt.figure(figsize=(10,7))
    plt.subplot(2,2,1); plt.hist(sdf["Pc_peak_bar"], bins=30); plt.title("Pc peak [bar]"); plt.grid(True)
    plt.subplot(2,2,2); plt.hist(sdf["F_peak_N"], bins=30); plt.title("Thrust peak [N]"); plt.grid(True)
    plt.subplot(2,2,3); plt.hist(sdf["F_mean_N"], bins=30); plt.title("Thrust mean [N]"); plt.grid(True)
    plt.subplot(2,2,4); plt.hist(sdf["It_Ns"], bins=30); plt.title("Total impulse [N·s]"); plt.grid(True)
    plt.tight_layout(); plt.savefig(hist_png, dpi=200); plt.close()

    # Markdown report
    md_path = f"{prefix}_REPORT.md"
    checks = ""
    if target_F_mean is not None:
        checks += f"- Target mean thrust ≥ {target_F_mean:.1f} N: **{'PASS' if pass_flags.get('target_F_mean_met') else 'FAIL'}**\n"
    if max_Pc_peak is not None:
        checks += f"- Max Pc peak ≤ {max_Pc_peak:.1f} bar: **{'PASS' if pass_flags.get('max_Pc_peak_ok') else 'FAIL'}**\n"
    if checks:
        checks = "## Quick checks (PASS/FAIL)\n" + checks + "\n"

    md = f"""# SRM Design Pre-Check Report (범용)

{checks}## Inputs (Nominal)
- Port exprs: {exprs}
- Coord bounds: x={xlim}, y={ylim}
- Grain: D={D_grain:.6g} m, L={L_grain:.6g} m
- Propellant: rho={rho:.2f} kg/m³, a={a_SI:.6e} (m/s)/Pa^n, n={n:.3f}
- Nozzle: At0={At0:.6e} m², eps(Ae/At)={eps:.3f}, Cd={Cd:.3f}, gamma={gamma:.2f}
- Env/Chamber: pa={pa:.0f} Pa, V_dead={V_dead:.3e} m³
- Sim: t_end={t_end:.3f}s, dt={dt:.4f}s
- MC({mc_eff} runs): a±{rel_a*100:.0f}%, c*±{rel_cstar*100:.0f}%, Cd±{rel_Cd*100:.0f}%, At±{rel_At*100:.0f}%, n±{n_sd:.3f}

## Nominal results
- Pc_peak: **{Pc_peak:.2f} bar**
- F_peak: **{F_peak:.1f} N**
- F_mean: **{F_mean:.1f} N**
- Total impulse: **{It:.1f} N·s**

### Nominal time histories
![Pc](./{os.path.basename(pc_png)})
![Thrust](./{os.path.basename(th_png)})

## Monte Carlo results
- Percentiles CSV: `{os.path.basename(pct_path)}`
- Samples CSV: `{os.path.basename(sdf_path)}`

![MC](./{os.path.basename(hist_png)})

### Key bands (P05 / P50 / P95)
- Pc_peak_bar: {pct.loc["P05","Pc_peak_bar"]:.2f} / {pct.loc["P50","Pc_peak_bar"]:.2f} / {pct.loc["P95","Pc_peak_bar"]:.2f}
- F_mean_N:    {pct.loc["P05","F_mean_N"]:.1f} / {pct.loc["P50","F_mean_N"]:.1f} / {pct.loc["P95","F_mean_N"]:.1f}
- It_Ns:       {pct.loc["P05","It_Ns"]:.1f} / {pct.loc["P50","It_Ns"]:.1f} / {pct.loc["P95","It_Ns"]:.1f}

## Interpretation (practical)
- If **P95(Pc_peak)** exceeds case/nozzle limits → increase At, inhibit more area, choose slower propellant, or use conservative Cd.
- If **P05(F_mean)** is below target → decrease At, increase initial perimeter (port), or increase c* (propellant/nozzle).
- Wide bands usually mean uncertainty in **a, Cd, At** dominates → tighten measurement/manufacturing.
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    # JSON summary
    js_path = f"{prefix}_summary.json"
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump({
            "nominal": {"Pc_peak_bar": Pc_peak, "F_peak_N": F_peak, "F_mean_N": F_mean, "It_Ns": It},
            "pass_flags": pass_flags,
            "percentiles": pct.to_dict(),
            "mc_draws": int(mc_eff),
            "inputs": {
                "exprs": exprs, "xlim": xlim, "ylim": ylim,
                "D_grain_m": D_grain, "L_grain_m": L_grain,
                "rho_kg_m3": rho, "a_SI": a_SI, "n": n,
                "cstar_m_s": cstar, "At0_m2": At0, "eps": eps, "Cd": Cd, "gamma": gamma,
            }
        }, f, ensure_ascii=False, indent=2)

    return {
        "report_md": md_path,
        "summary_json": js_path,
        "pc_png": pc_png,
        "thrust_png": th_png,
        "mc_hist_png": hist_png,
        "mc_percentiles_csv": pct_path,
        "mc_samples_csv": sdf_path,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design_xlsx", required=True, help="Design Excel file path")
    ap.add_argument("--expr", action="append", required=True, help="Port implicit expression (repeatable). e.g. circle(x,y,0,0,3.9)")
    ap.add_argument("--xlim", nargs=2, type=float, required=True)
    ap.add_argument("--ylim", nargs=2, type=float, required=True)
    ap.add_argument("--eps", type=float, required=True, help="Nozzle area ratio Ae/At")
    ap.add_argument("--Cd", type=float, default=0.95)
    ap.add_argument("--gamma", type=float, default=1.2)
    ap.add_argument("--V_dead", type=float, default=0.0)
    ap.add_argument("--pa", type=float, default=101325.0)
    ap.add_argument("--t_end", type=float, default=3.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--mc_draws", type=int, default=800)
    ap.add_argument("--seed", type=int, default=7)

    # Burn-rate inputs
    ap.add_argument("--a_SI", type=float, default=None, help="a in SI: (m/s)/Pa^n")
    ap.add_argument("--a_mm_MPa", type=float, default=None, help="a in (mm/s)/(MPa^n)")
    ap.add_argument("--n", type=float, required=True)

    # Uncertainty knobs
    ap.add_argument("--rel_a", type=float, default=0.25)
    ap.add_argument("--rel_cstar", type=float, default=0.06)
    ap.add_argument("--rel_Cd", type=float, default=0.05)
    ap.add_argument("--rel_At", type=float, default=0.03)
    ap.add_argument("--n_sd", type=float, default=0.03)

    ap.add_argument("--erosion_k", type=float, default=0.0)
    ap.add_argument("--erosion_m", type=float, default=0.0)
    ap.add_argument("--rel_erosion_k", type=float, default=0.5)

    ap.add_argument("--override", action="append", default=[], help='Override design value: key=value (repeatable).')
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--name", default="design")

    # Targets
    ap.add_argument("--target_F_mean", type=float, default=None)
    ap.add_argument("--max_Pc_peak", type=float, default=None)

    args = ap.parse_args()

    design = load_design_from_excel(args.design_xlsx)
    design.update(parse_overrides(args.override))

    # Require the basics from design
    required = ["D_grain_m", "L_grain_m", "rho_kg_m3", "cstar_m_s", "At_m2"]
    missing = [k for k in required if k not in design]
    if missing:
        raise SystemExit(f"Missing from design sheet (or overrides): {missing}")

    if args.a_SI is None:
        if args.a_mm_MPa is None:
            raise SystemExit("Provide either --a_SI or --a_mm_MPa")
        a_SI = a_mmMPa_to_SI(args.a_mm_MPa, args.n)
    else:
        a_SI = args.a_SI

    artifacts = make_design_report(
        exprs=args.expr,
        xlim=(args.xlim[0], args.xlim[1]),
        ylim=(args.ylim[0], args.ylim[1]),
        D_grain=design["D_grain_m"],
        L_grain=design["L_grain_m"],
        rho=design["rho_kg_m3"],
        cstar=design["cstar_m_s"],
        At0=design["At_m2"],
        eps=args.eps,
        a_SI=a_SI,
        n=args.n,
        Cd=args.Cd,
        gamma=args.gamma,
        V_dead=args.V_dead,
        pa=args.pa,
        t_end=args.t_end,
        dt=args.dt,
        mc_draws=args.mc_draws,
        rel_a=args.rel_a,
        rel_cstar=args.rel_cstar,
        rel_Cd=args.rel_Cd,
        rel_At=args.rel_At,
        n_sd=args.n_sd,
        erosion_k=args.erosion_k,
        erosion_m=args.erosion_m,
        rel_erosion_k=args.rel_erosion_k,
        seed=args.seed,
        out_dir=args.out_dir,
        name=args.name,
        target_F_mean=args.target_F_mean,
        max_Pc_peak=args.max_Pc_peak,
    )

    print("Generated artifacts:")
    for k, v in artifacts.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
