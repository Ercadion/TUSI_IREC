
"""
Generic Solid Rocket Motor (SRM) 0D+Geometry Simulator
======================================================

Goal
----
A reusable simulator that can be used before TMS/hardware tests to sanity-check
whether a target thrust/pressure is plausible from design numbers.

Core features
-------------
1) Geometry-driven Ab(t), Aport(t) using a signed-distance field (SDF) on a grid.
   - Supports arbitrary implicit port shapes via safe expression parser.
   - Computes port area and perimeter from a binary mask (fast).
2) 0D chamber pressure dynamics (ODE form, integrated explicitly):
      dP/dt = (R*T/V) * (mgen - mout) - (P/V)*dV/dt
   with:
      mgen = rho_p * rdot(P,T) * Ab
      rdot = a * (P**n) * f_T(T)   (default f_T = 1)
      mout = Cd * At(t) * P / c*
3) Nozzle throat erosion model: At(t) = At0 + dAt_dt(P)*t (simple options)
4) Thrust model: F = Cf(P, pa, eps, gamma) * P * At
   - Uses ideal nozzle relations (frozen, 1D) for Cf.
5) Uncertainty wrapper: Monte Carlo over {a, n, c*, Cd, At0, erosion}.

Units (SI)
----------
- Pressure P: Pa
- Lengths: m
- Areas: m^2
- Mass: kg
- Time: s
- a in rdot = a*P^n must be SI: [a] = (m/s) / Pa^n

Dependencies
------------
numpy, scipy (optional), matplotlib (optional), pandas (for excel loader), skimage (optional)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

# ----------------------------
# Safe expression evaluation
# ----------------------------
_SAFE_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "arctan2": np.arctan2,
    "sqrt": np.sqrt, "abs": np.abs,
    "exp": np.exp, "log": np.log,
    "pi": np.pi, "e": np.e,
    "min": np.minimum,
    "max": np.maximum,
}

def circle(x, y, cx=0.0, cy=0.0, r=1.0):
    """Implicit circle: <=0 is inside."""
    return np.sqrt((x - cx)**2 + (y - cy)**2) - r

def box(x, y, cx=0.0, cy=0.0, w=1.0, h=1.0):
    """Axis-aligned box centered at (cx,cy), half-width w/2, half-height h/2."""
    return np.maximum(np.abs(x - cx) - w/2.0, np.abs(y - cy) - h/2.0)

def star(x, y, cx=0.0, cy=0.0, r_in=1.0, r_out=2.0, k=5):
    """
    Simple star-ish implicit boundary using polar modulation.
    Not a perfect SDF, but works for port region definition.
    """
    X = x - cx
    Y = y - cy
    th = np.arctan2(Y, X)
    r = np.sqrt(X*X + Y*Y)
    r_target = r_in + (r_out - r_in) * (0.5 * (1.0 + np.cos(k*th)))
    return r - r_target

def _normalize_expr(expr: str) -> str:
    expr = expr.strip()
    # allow "x^2 + y^2 = 25" style
    if "=" in expr and "==" not in expr:
        left, right = expr.split("=", 1)
        expr = f"({left})-({right})"
    expr = expr.replace("^", "**")
    return expr

def eval_expr(expr: str, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    expr = _normalize_expr(expr)
    scope = dict(_SAFE_FUNCS)
    scope.update({"x": X, "y": Y, "circle": circle, "box": box, "star": star})
    return eval(expr, {"__builtins__": {}}, scope)

# ----------------------------
# Geometry utilities
# ----------------------------
def make_grid(xlim: Tuple[float,float], ylim: Tuple[float,float], grid: int) -> Tuple[np.ndarray,np.ndarray,float,float]:
    xs = np.linspace(xlim[0], xlim[1], grid)
    ys = np.linspace(ylim[0], ylim[1], grid)
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    X, Y = np.meshgrid(xs, ys)
    return X, Y, dx, dy

def area_from_mask(mask: np.ndarray, dx: float, dy: float) -> float:
    return float(mask.sum()) * dx * dy

def perimeter_4n(mask: np.ndarray, dx: float, dy: float) -> float:
    """
    Fast perimeter estimate using 4-neighborhood edge counting.
    Assumes square-ish pixels (dx~dy). Uses dx for horizontal and dy for vertical.
    """
    m = mask.astype(bool)
    # differences along axes
    horiz = (m[:, 1:] != m[:, :-1]).sum()
    vert  = (m[1:, :] != m[:-1, :]).sum()
    # each differing pair corresponds to an edge segment
    return float(horiz) * dy + float(vert) * dx

@dataclass
class PortGeometry:
    """
    Geometry built from implicit expressions (union of inside regions).
    """
    exprs: List[str]
    xlim: Tuple[float,float]
    ylim: Tuple[float,float]
    D_grain: float          # grain diameter [m] used to scale coordinate units -> meters
    grid: int = 700

    def __post_init__(self):
        X, Y, dx, dy = make_grid(self.xlim, self.ylim, self.grid)
        self.X0, self.Y0 = X, Y
        self.dx0, self.dy0 = dx, dy
        # scale from coord units to meters
        self.scale = self.D_grain / (self.xlim[1] - self.xlim[0])  # [m/coord]
        self.dx = dx * self.scale
        self.dy = dy * self.scale
        # outer grain boundary mask (circle with radius = half of coord width)
        R_coord = (self.xlim[1] - self.xlim[0]) / 2.0
        outer = circle(X, Y, 0.0, 0.0, R_coord) <= 0
        self.mask_outer = outer

        # initial port SDF-like field as min of expressions
        F = np.full_like(X, 1e9, dtype=float)
        for e in self.exprs:
            F = np.minimum(F, eval_expr(e, X, Y))
        self.F = F

    def port_mask(self) -> np.ndarray:
        # "gas" region = inside port but within outer grain
        gas = (self.F <= 0.0) & self.mask_outer
        return gas

    def A_port(self) -> float:
        return area_from_mask(self.port_mask(), self.dx, self.dy)

    def P_port(self) -> float:
        return perimeter_4n(self.port_mask(), self.dx, self.dy)

    def expand_port(self, dr: float) -> None:
        """
        Expand port boundary by dr [m] using the SDF field in coord-units.
        F is in coord units approximately, so convert dr -> coord units.
        For exact SDF you'd subtract dr/scale. Here works well for smooth shapes.
        """
        dcoord = dr / self.scale
        self.F = self.F - dcoord

# ----------------------------
# Propellant, nozzle, motor model
# ----------------------------
@dataclass
class Propellant:
    rho: float            # kg/m^3
    a: float              # (m/s)/Pa^n
    n: float              # dimensionless
    Tc: float = 3000.0    # K (only used if you enable pressure ODE with RT/V)
    Rgas: float = 300.0   # J/kg-K (effective), rough for SRM gases
    temp_coeff: float = 0.0  # 1/K; if nonzero, rdot *= exp(temp_coeff*(Tc-Tref))
    Tref: float = 3000.0

    def rdot(self, Pc: float) -> float:
        if Pc <= 0:
            return 0.0
        fT = math.exp(self.temp_coeff * (self.Tc - self.Tref)) if self.temp_coeff != 0.0 else 1.0
        return self.a * (Pc ** self.n) * fT

@dataclass
class Nozzle:
    At0: float            # m^2
    eps: float            # Ae/At
    Cd: float = 0.95      # effective discharge coefficient
    gamma: float = 1.2    # for Cf model
    erosion_k: float = 0.0   # m^2/s baseline throat-area growth (simple)
    erosion_m: float = 0.0   # exponent vs Pc (optional)
    At_min: float = 1e-8

    def At(self, Pc: float, t: float) -> float:
        # simple power-law erosion in area:
        if self.erosion_k <= 0.0:
            return max(self.At0, self.At_min)
        if self.erosion_m == 0.0:
            return max(self.At0 + self.erosion_k * t, self.At_min)
        return max(self.At0 + self.erosion_k * (Pc**self.erosion_m) * t, self.At_min)

def nozzle_Cf(gamma: float, Pc: float, pa: float, eps: float) -> float:
    """
    Ideal thrust coefficient for choked nozzle, 1D isentropic expansion.
    Solves for exit pressure ratio via area ratio eps.
    """
    if Pc <= 0:
        return 0.0
    g = gamma
    # function to match eps = Ae/At for given Me
    def area_ratio(Me):
        term1 = (2/(g+1))*(1 + (g-1)/2 * Me*Me)
        return (1/Me) * (term1 ** ((g+1)/(2*(g-1))))
    # find Me by simple bracketed search
    # Me must be >=1
    lo, hi = 1.0001, 20.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if area_ratio(mid) < eps:
            lo = mid
        else:
            hi = mid
    Me = 0.5*(lo+hi)
    pe_over_pc = (1 + (g-1)/2 * Me*Me) ** (-g/(g-1))
    pe = pe_over_pc * Pc
    # momentum term
    Cf_mom = math.sqrt((2*g*g/(g-1)) * (2/(g+1))**((g+1)/(g-1)) * (1 - pe_over_pc**((g-1)/g)))
    Cf = Cf_mom + (pe - pa) / Pc * eps
    return Cf

@dataclass
class Motor:
    geom: PortGeometry
    prop: Propellant
    noz: Nozzle
    cstar: float          # m/s
    L_grain: float        # m
    V_dead: float = 0.0   # m^3 additional free volume (chamber+headspace)
    pa: float = 101325.0  # ambient pressure Pa
    use_ode: bool = True  # if False, quasi-steady Pc is used

    def initial_volume(self) -> float:
        # port volume = A_port * L, plus dead volume
        return self.geom.A_port() * self.L_grain + self.V_dead

    def burn_area(self) -> float:
        # side-burning only by default; user can extend as needed
        return self.geom.P_port() * self.L_grain

    def mdot_out(self, Pc: float, t: float) -> float:
        At = self.noz.At(Pc, t)
        return self.noz.Cd * At * Pc / self.cstar

    def mdot_gen(self, Pc: float) -> float:
        Ab = self.burn_area()
        rdot = self.prop.rdot(Pc)
        return self.prop.rho * rdot * Ab

    def step(self, Pc: float, t: float, dt: float) -> Tuple[float, Dict[str,float]]:
        # Geometry at beginning of step
        Aport = self.geom.A_port()
        Pport = self.geom.P_port()
        Ab = Pport * self.L_grain
        V = Aport * self.L_grain + self.V_dead

        # mass flows
        mgen = self.prop.rho * self.prop.rdot(Pc) * Ab
        mout = self.mdot_out(Pc, t)

        # pressure update
        if self.use_ode:
            # dV/dt from port expansion approx: dA/dt * L
            # compute dA for a small virtual expansion based on current rdot
            dr = self.prop.rdot(Pc) * dt
            # approximate dA via perimeter * dr (good for smooth shapes)
            dA = Pport * dr
            dVdt = (dA * self.L_grain) / dt if dt > 0 else 0.0
            dPdt = (self.prop.Rgas * self.prop.Tc / max(V, 1e-12)) * (mgen - mout) - (Pc / max(V, 1e-12)) * dVdt
            Pc_next = max(Pc + dPdt * dt, 0.0)
        else:
            # quasi-steady Pc from mgen=mout -> Pc = (c* rho a Ab /(Cd At))^(1/(1-n))
            At = self.noz.At(Pc, t)
            Pc_next = (self.cstar * self.prop.rho * self.prop.a * Ab / (self.noz.Cd * max(At, 1e-12))) ** (1.0 / (1.0 - self.prop.n))

        # geometry advance using current regression rate (explicit)
        dr = self.prop.rdot(Pc_next) * dt
        self.geom.expand_port(dr)

        # thrust
        At_now = self.noz.At(Pc_next, t+dt)
        Cf = nozzle_Cf(self.noz.gamma, Pc_next, self.pa, self.noz.eps)
        F = Cf * Pc_next * At_now

        diag = {
            "Pc_Pa": Pc_next,
            "Pc_bar": Pc_next/1e5,
            "A_port": Aport,
            "P_port": Pport,
            "A_burn": Ab,
            "V": V,
            "mdot_gen": mgen,
            "mdot_out": mout,
            "At": At_now,
            "Cf": Cf,
            "Thrust_N": F,
            "rdot_m_s": self.prop.rdot(Pc_next),
            "dr_step_m": dr,
        }
        return Pc_next, diag

def run_simulation(
    motor: Motor,
    t_end: float,
    dt: float,
    Pc0: Optional[float] = None,
    stop_on_burnout: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run SRM simulation. Returns timeseries dict of arrays.
    """
    steps = int(math.ceil(t_end / dt)) + 1
    t = np.linspace(0.0, t_end, steps)

    # initialize Pc
    if Pc0 is None:
        # small initial pressure to start flow (or choose ambient)
        Pc = max(motor.pa, 1e5)  # 1 bar
    else:
        Pc = float(Pc0)

    out: Dict[str, List[float]] = {k: [] for k in [
        "t", "Pc_Pa", "Pc_bar", "A_port", "P_port", "A_burn", "V",
        "mdot_gen", "mdot_out", "At", "Cf", "Thrust_N", "rdot_m_s"
    ]}

    for i in range(steps):
        ti = float(t[i])
        out["t"].append(ti)

        Pc, diag = motor.step(Pc, ti, dt)
        for k in out.keys():
            if k == "t": 
                continue
            out[k].append(float(diag[k]))

        # burnout check: if port area nearly equals outer grain area, or Ab near 0
        if stop_on_burnout:
            # if perimeter collapses or gas region touches outer boundary everywhere, A_burn might still exist;
            # use a simple threshold on remaining web: if expansion makes port fill most of grain cross-section
            # Here: stop when A_port exceeds 98% of outer grain area.
            A_outer = float(motor.geom.mask_outer.sum()) * motor.geom.dx * motor.geom.dy
            if out["A_port"][-1] > 0.98 * A_outer:
                # pressure decays (set to ambient) and stop
                break

    return {k: np.asarray(v) for k, v in out.items()}

# ----------------------------
# Excel loaders (optional)
# ----------------------------
def load_design_from_excel(path: str) -> Dict[str, float]:
    """
    A loose parser for your design sheet style:
    finds Korean labels and reads adjacent cell values.
    """
    import pandas as pd
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

    out = {}
    if D_mm is not None: out["D_grain_m"] = D_mm/1000.0
    if core_mm is not None: out["core_diam_m"] = core_mm/1000.0
    if L_mm is not None: out["L_grain_m"] = L_mm/1000.0
    if rho_gcc is not None: out["rho_kg_m3"] = rho_gcc*1000.0
    if cstar is not None: out["cstar_m_s"] = cstar
    if At_mm2 is not None: out["At_m2"] = At_mm2*1e-6
    return out

# ----------------------------
# Monte Carlo wrapper
# ----------------------------
def monte_carlo(
    base_motor_factory: Callable[[Dict[str,float]], Motor],
    draws: int,
    t_end: float,
    dt: float,
    param_dists: Dict[str, Tuple[float,float]],
    seed: int = 0,
) -> Dict[str, Any]:
    """
    param_dists: dict of parameter -> (mu, sigma) for lognormal on positive params,
                 or for n you can pass ("normal", mu, sigma) via a small convention below.
    Example:
      {"a": (a0, 0.20), "cstar": (c0, 0.05), "Cd": (0.95, 0.03), "At0": (At0, 0.02)}
      where sigma is relative (20% etc) for lognormal.
    """
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(draws):
        p = {}
        for k, spec in param_dists.items():
            if isinstance(spec, tuple) and len(spec)==3 and spec[0]=="normal":
                _, mu, sd = spec
                p[k] = float(rng.normal(mu, sd))
            else:
                mu, rel = spec
                # lognormal with given relative sigma (approx)
                sigma = math.sqrt(math.log(1 + rel*rel))
                m = math.log(mu) - 0.5*sigma*sigma
                p[k] = float(rng.lognormal(m, sigma))
        motor = base_motor_factory(p)
        ts = run_simulation(motor, t_end=t_end, dt=dt)
        results.append({
            "Pc_peak_bar": float(np.max(ts["Pc_bar"])),
            "F_peak_N": float(np.max(ts["Thrust_N"])),
            "F_mean_N": float(np.mean(ts["Thrust_N"])),
            "It_Ns": float(np.trapz(ts["Thrust_N"], ts["t"])),
        })
    df = __import__("pandas").DataFrame(results)
    return {
        "summary": df.describe(percentiles=[0.05,0.5,0.95]).to_dict(),
        "samples": df,
    }
