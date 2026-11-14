# actuator-sensed-yoyo · Simulation physics spec

English export of the project's physics spec (source doc dated 2026-11-13), plus the
amendments made while implementing v0–v3 (marked **[A#]**, summarized in §11).
This file is the only physics reference for the simulator. Every formula in the code
cites its section (e.g. `# §3.2`). To change a physical assumption, change it here
first, then the code. Anything uncertain is a `TODO(sysid)` in `config/params.yaml`,
never an invented number.

## 0. Current design, conventions

### Current design
- **Joint 1** (base): GM4108 through one 6:1 GT2 belt (20T → 120T) driving link 1. The only belt on the machine.
- **Link 1**: 10 mm carbon tube, l₁ = 0.15 m.
- **Wrist**: GM3506 direct drive. Stator on one side of a U-fork at the end of link 1, a 608 bearing and stub shaft on the other side, funnel hung in between. String exit right under the wrist axis, l₂ ≈ 1.5 cm.
- **Funnel**: PETG body + TPU liner + neodymium magnet in the seat, about 60 × 85 × 45 mm.
- **Yo-yo**: Duncan Imperial, fixed axle, string tied to the axle, L = 0.8 m. It cannot sleep: it always rebounds at the bottom.

### Coordinates and conventions
- 2-D vertical plane, x right, y up, `g_vec = (0, −g)`, `g = 9.81`. SI units throughout.
- **String exit H**: the string hole at the top (apex) of the funnel. The end-effector of the arm kinematics is H.
- **String direction u**: unit vector from H to the yo-yo centre. θ is the string angle from straight down: `u = (sinθ, −cosθ)`, `u_θ = (cosθ, sinθ)`.
- **ℓ**: paid-out string length from H to the yo-yo centre; L is the total length. `p = H + ℓ·u`.
- **ω**: signed spin rate of the yo-yo.
- **σ ∈ {+1, −1}**: winding direction. When taut the rolling constraint is `ℓ̇ = σ·r·ω`. At every bottom the string flips to the other side of the axle and σ changes sign; ω keeps its direction at the bottom and only reverses after the apex (ω = 0).
- The arm plane is the yo-yo's spin plane (axle ⟂ plane). "In-plane" funnel dimensions are along the yo-yo diameter.

## 1. Parameters
All in one `Params` dataclass (`sim/params.py`), values in `config/params.yaml` with randomization ranges.

### Yo-yo and string
| Name | Symbol | Default | Randomization | Source |
|---|---|---|---|---|
| mass | `m` | 0.050 kg | ±5 % | scale |
| outer radius | `R` | 0.028 m | — | calipers |
| spin inertia | `I` | 0.6·m·R² ≈ 2.35e-5 kg·m² (`I_coef` = 0.6) | ±20 % | `TODO(sysid)` drop test |
| axle radius, string fully out | `r0` | 0.006 m | 0.004–0.008 | `TODO(sysid)` |
| string length | `L` | 0.80 m | ±1 cm | tape |
| string diameter | `d_s` | 1.2e-3 m | — | calipers |
| fully wound radius | `r_full` | 0.015 m | — | calipers, gives `w_eff` (§3.4) |
| bottom loss correction | `c_loss` | 1.0 | 0.85–1.0 | `TODO(sysid)` |
| damping along string | `c_l` | 0 1/s | 0–0.5 | `TODO(sysid)` |
| snap restitution | `e_taut` | 0.1 | 0–0.3 | estimate |
| min snap speed for rebound **[A3]** | `v_snap_min` | 0.05 m/s | — | numerical |

### Arm
| Name | Symbol | Default | Randomization | Source |
|---|---|---|---|---|
| link 1 (joint-1 axis → wrist axis) | `l1` | 0.15 m | design 0.15–0.18 | stage-2 sweep |
| wrist axis → string exit | `l2` | 0.015 m | design 0.01–0.03 | stage-2 sweep |
| link 1 mass | `m1` | 0.04 kg | ±10 % | scale |
| link 2 adapter mass | `m2` | 0.005 kg | ±20 % | scale |
| funnel mass | `mc` | 0.02 kg | ±10 % | scale |
| funnel COM from wrist axis | `lc` | `l2 + 0.02` m | — | CAD |
| elbow mass (GM3506 + U-fork + 608) | `m_e` | 0.08 kg | ±10 % | scale |
| joint-1 hard limits | `q1_min, q1_max` | −180°, +180° | — | design |

### Transmission, motors, drivers
| Name | Symbol | Default | Randomization | Source |
|---|---|---|---|---|
| joint-1 ratio | `N1` | 6 | — | design |
| joint-1 belt efficiency | `eta1` | 0.9 | 0.8–0.95 | estimate |
| wrist | — | direct drive, N = 1, η = 1 | — | design |
| rotor inertia GM4108 | `Jr1` | 4e-5 kg·m² | ±20 % | `TODO` datasheet |
| rotor inertia GM3506 | `Jr2` | 2e-5 kg·m² | ±20 % | `TODO` datasheet |
| torque constants | `Kt1, Kt2` | placeholders 0.09, 0.05 N·m/A | ±10 % | `TODO(sysid)` weights |
| phase resistance | `R1, R2` | 11.1 Ω, 5.57 Ω | ±5 % | datasheet |
| continuous torque | `tau_max1, tau_max2` | 0.15, 0.08 N·m | — | datasheet; 2–3× short-term |
| bus voltage | `V_bus` | 24 V | — | PSU |
| motor-2 voltage limit | `V_lim2` | 12 V | — | GM3506 2–3S |
| viscous friction | `b1, b2` | placeholders | ±50 % | `TODO(sysid)` |
| Coulomb friction | `tc1, tc2` | placeholders | ±50 % | `TODO(sysid)` |
| current-loop time constant | `tau_i` | 0.8 ms | 0.5–1.5 ms | estimate |
| joint-1 belt stiffness (v4) | `k_belt, c_belt` | — | — | `TODO(sysid)` tap test |

### Funnel
| Name | Symbol | Default | Randomization | Source |
|---|---|---|---|---|
| total depth | `d_cup` | 0.045 m | — | CAD |
| hole → yo-yo top gap | `d_apex` | 0.003 m | — | CAD (swivel) |
| seat (in-plane × out-of-plane) | `seat_width × seat_thickness` | 59 × 36 mm | — | calipers + 1–2 mm |
| straight seat depth **[A4]** | `seat_depth` | 0.028 m | — | CAD |
| mouth | `mouth_width × mouth_thickness` | 80 × 55 mm | — | CAD |
| flare exponent **[A4]** | `flare_exponent` | 2 | — | design |
| gate model: max misalignment | `alpha_cap` | 10° | 5–15° | hand test |
| max seat speed to stow | `v_cap_max` | 0.5 m/s | 0.3–0.8 | hand test |
| seat restitution | `e_cup` | 0.3 | 0.1–0.5 | hand test |
| magnet pull when seated | `F_mag0` | 1.0 N | 0.6–1.5 | spring scale |
| magnet decay distance | `d_mag` | 0.003 m | — | measured |
| wall penalty stiffness / restitution / friction **[A4]** | `k_wall, e_wall, mu_wall` | 5e4 N/m, 0.3, 0.3 | μ 0.1–0.6 | measure |

### Derived quantities (functions, never stored)
```
kappa(r)  = m r² / (I + m r²)          # falling-acceleration ratio
zeta      = (I − m r0²) / (I + m r0²)  # Zlajpah bottom speed ratio
ell_min   = R + d_apex                  # H → yo-yo centre when stowed
ell_mouth = R + d_cup                   # yo-yo top reaches the mouth plane
w_eff     = L d_s² / (4 (r_full² − r0²))
```
Defaults: `kappa(r0) ≈ 0.071`, `zeta ≈ 0.858`, `ell_min = 0.031 m`, `ell_mouth = 0.073 m`.
Released at ℓ = 0 from rest with a fixed hook and r ≡ r0: bottom speed ≈ 1.06 m/s,
spin ≈ 176 rad/s, fall time ≈ 1.5 s (unit-test numbers).

## 2. Hybrid system: modes and events
Four modes: `STOWED`, `FREE`, `TAUT`, `FAIL`. The bottom and the seat rebound are
events inside TAUT (or FREE, [A2]), not modes; after a bottom a timer tracks the
remaining hook force pulse (§3.6).

| Event | In | Condition | Result | Rule |
|---|---|---|---|---|
| release | STOWED | inward force needed to keep the yo-yo seated > F_mag0 | → FREE | §4.1 |
| snap | FREE | `|p − H| ≥ ℓ_s` | impulse, → TAUT (or rebound) | §3.5 |
| slack | TAUT | computed `T < −0.01 N` | → FREE, `ℓ_s = ℓ` | §3.2 |
| bottom | TAUT | `ℓ ≥ L` and `ℓ̇ > 0` | stay TAUT, flip σ, start force pulse | §3.6 |
| mouth | TAUT/FREE **[A6]** | `|p − H|` crosses `ell_mouth` inwards | gate: aligned → inside, else FAIL; geometry: inside | §4.2 |
| seat | TAUT/FREE **[A2]** | `|p − H| ≤ ell_min`, moving inwards | slow → STOWED, fast → rebound | §4.2 |
| hit rim **[A4]** | TAUT/FREE | disc top above the mouth plane with the centre outside the inner wall | → FAIL | §4.3 |
| joint limit | any | `q1 ∉ [q1_min, q1_max]` | → FAIL | §5 |

- Integer `mode`; the JAX port will branch with `lax.switch` / `jnp.where`.
- Event location: fixed step, RK4 inside a mode, zero crossing found by linear
  interpolation of the event function; integrate to the event, apply the map,
  integrate the rest of the step. At most one interpolated event per step.
  Slack, release, mouth and rim are checked at the step end.
- Physics step `dt = 2e-4 s` (5 kHz): the bottom half-turn only lasts ≈ 18 ms.
- Slack hysteresis: switch to FREE only for `T < −0.01 N`.

## 3. Yo-yo dynamics

### 3.1 State
`(mode, ℓ, ℓ̇, θ, θ̇, ω, σ, p, ṗ, ℓ_s, pulse_t, pulse_J, pulse_dt, inside)`
- TAUT: `p, ṗ` follow from `(ℓ, ℓ̇, θ, θ̇)` and H; spin from the constraint `ω = σ·ℓ̇ / r(ℓ)`.
- FREE: `p, ṗ, ω` are integrated; `ℓ_s` is the paid-out length, which keeps winding/unwinding with the spin, `ℓ̇_s = σ r(ℓ_s) ω` **[A7]**; σ is unchanged.
- `pulse_*` track the bottom force pulse (§3.6), `inside` whether the yo-yo is inside the mouth (§4.2).

### 3.2 TAUT: paying out and winding up are the same equation
The hook acceleration `Ḧ` comes from the arm (§5). Define
```
F_mag = magnet force (§4.2), 0 unless inside
F_c   = funnel wall contact force on the yo-yo, τ_c its spin torque (§4.3 [A4])
A     = g cosθ − Ḧ·u + ℓ θ̇² − F_mag/m + F_c·u/m
rho   = (ṙ / r) ℓ̇                               # winding-radius term, 0 for constant r
ℓ̈  = kappa(r)·A + (1 − kappa(r))·rho + σ r τ_c / (I + m r²) − c_l·ℓ̇
θ̈  = ( −g sinθ − Ḧ·u_θ − 2 ℓ̇ θ̇ + F_c·u_θ/m ) / ℓ
T   = m ( A − ℓ̈ )                                # string tension
F_H = (T + F_mag)·u                              # string + magnet on the hook
```
Derivation (also in code comments):
- translation `m p̈ = −(T + F_mag)·u + m g_vec + F_c`; spin `I ω̇ = σ·T·r + τ_c`. The magnet acts on the axle: no torque.
- differentiate `ℓ̇ = σ r ω` twice and combine with the radial equation; σ drops out, so paying out and winding up need no separate cases.
- vertical string, vertical hook motion, no magnet: `ℓ̈ = κ (g + Ḧ_y)`, `T = m (g + Ḧ_y)(1 − κ)` (a unit test).
- `c_l` is phenomenological (string rubbing the yo-yo sides), default 0, identified later.

### 3.3 1-D simplification (v0, `planar: false`)
θ ≡ 0, H moves vertically only, no funnel: `ℓ̈ = κ (g + z̈) − c_l ℓ̇`, `T = m (g + z̈)(1 − κ)`.

### 3.4 Winding radius r(ℓ) (v4, `variable_radius: true`; before that r ≡ r0)
```
r(ℓ)²  = r0² + (L − ℓ)·d_s² / (4 w_eff)
dr/dℓ  = −d_s² / (8 w_eff r),   ṙ = (dr/dℓ)·ℓ̇
```
κ changes a lot from r0 to r_full (≈ 0.07 → 0.32), so the first part of a throw falls
much faster. The winding constraint is ideal (no slip), so energy is still conserved.

### 3.5 FREE (slack string) and snapping taut
- Free flight: `p̈ = g_vec (+ F_c/m − F_mag/m·u_r [A2])`, `ω̇ = τ_c / I`; σ unchanged.
- **[A7]** The string is tied to the axle, so a spinning slack yo-yo keeps winding or paying out string: `ℓ̇_s = σ·r(ℓ_s)·ω` (clamped to `[ell_min, L]`). With ℓ_s frozen (the original spec), a few ms of slack after a bottom would throw away the whole climb.
- Snap when `|p − H| ≥ ℓ_s`:
```
u     = (p − H) / |p − H|
v_r   = (ṗ − Ḣ)·u                    # positive away from the hook
Δ     = v_r − σ·r·ω                   # how much faster the yo-yo moves than the string pays out
Δ ≤ 0 : no impulse, switch to TAUT
else  :
  m_eff = 1 / (1/m + r²/I + u·Λ_H⁻¹·u)   # Λ_H: arm operational-space inertia at H (0 for a kinematic hook)
  P     = (1 + e)·m_eff·Δ,   e = e_taut if Δ ≥ v_snap_min else 0      [A3]
  ṗ⁺    = ṗ⁻ − (P/m)·u,   ω⁺ = ω⁻ + σ·P·r / I,   hook gets +P·u (§5.3)
```
- `e = 0`: afterwards `v_r = σ r ω`, switch to TAUT with ℓ = ℓ_s.
- `e > 0`: the yo-yo rebounds relative to the string and stays FREE until the next snap.

### 3.6 Bottom event
When ℓ reaches L with `ℓ̇ > 0`:
```
v⁻ = ℓ̇⁻,  ζ_e = c_loss·zeta
ℓ̇⁺ = −ζ_e·v⁻,   σ⁺ = −σ⁻,   ω⁺ = ζ_e·ω⁻,   E⁺ = ζ_e²·E⁻
J        = m (1 + ζ_e) v⁻         # total string impulse on the hook
pulse_dt = π r0 / v⁻              # half a turn around the axle
```
- Yo-yo side: instantaneous change of the radial velocity; in 2-D only the along-string component reverses, θ̇ is unchanged.
- Hook side: **not** an instantaneous impulse. J is spread as a raised-cosine force so the arm has time to react and the current shows a shape:
```
F_b(t) = (J / pulse_dt)·(1 − cos(2π t / pulse_dt))·u,  0 ≤ t ≤ pulse_dt,   peak = 2 J / pulse_dt
```
- The yo-yo reverses instantly while the hook receives the same total impulse over ≈ 18 ms. Intentional simplification.
- **[A1]** Consequently the yo-yo equations (§3.2) use the hook acceleration *without* the part caused by F_b (`Ḧ_yoyo = Ḧ − Λ_H⁻¹ F_b`): the yo-yo was already reversed by that impulse. Without this, the ~60 m/s² hook kick makes T < 0 and the string goes "slack" 3 ms after every bottom.
- Defaults: `J ≈ 0.098 N·s`, `pulse_dt ≈ 18 ms`, mean ≈ 5.5 N, peak ≈ 11 N, plus T.

### 3.7 Passive bounce relations
Fixed hook, constant r: interval between bottoms `T_b` and apex height above the bottom `h`:
```
h = κ g T_b² / 8,    h_{k+1} / h_k = ζ_e²
```
The first one is the core of the stage-4 proprioceptive observer: the time between two current spikes gives the apex height.

## 4. Funnel: throw and catch
The funnel is fixed to the wrist rotor; its apex is H. Opening direction
`a = (cos φ2, sin φ2)` (from the wrist axis towards H). Bounce posture: link 2 down,
φ2 = −90°, `a = (0, −1)`. Funnel frame: axial z along a from H (0 at H, d_cup at the
mouth plane), lateral x along `n = (−sin φ2, cos φ2)`.

### 4.1 STOWED and the throw
- STOWED: yo-yo and funnel are one rigid body — a point mass on link 2 at `p = H + ell_min·a` (distance `l2 + ell_min` from the wrist axis), merged into M and G of §5.2. `ω = 0`, `ℓ = ell_min`.
- Every step compute the force the funnel must exert to carry the yo-yo:
```
p̈_s  = acceleration of that point of link 2 (incl. tangential and centripetal terms)
F_req = m (p̈_s − g_vec)
F_in  = −F_req·a          # inward ("pull") component: only the magnet can supply it
```
- `F_in > F_mag0` → release: FREE with the point's velocity, `ω = 0`, `σ = +1`, `ℓ_s = ell_min`, `inside = false`, mass removed from link 2 (velocities are continuous).
- At rest `F_in = m g ≈ 0.49 N < F_mag0`: it holds. The funnel can push the yo-yo out but can only pull it in through the magnet, so the natural throw is "accelerate down, then stop hard": the stop is an upward acceleration, F_in exceeds the magnet and the yo-yo leaves downwards.
- Catch (§4.2): the merge is perfectly inelastic, conserving the arm's generalized momentum `M_new q̇⁺ = M_old q̇⁻ + Jac_sᵀ m ṗ`.

### 4.2 Catch: three steps (the "gate" model, `capture_model: gate`)
**Step 1, mouth**: when `|p − H|` crosses `ell_mouth` inwards from outside **[A6, A8]**:
`β = angle(u, a)`. `β ≥ alpha_cap` → hits the rim, FAIL. Otherwise `inside = true`,
and from then on the funnel holds the string on its axis: `u = a`, θ and θ̇ follow the funnel.

**Step 2, inside** (`ell_min < ℓ < ell_mouth`): the magnet acts (subtracted from A in §3.2):
```
d = ℓ − ell_min,   F_mag = F_mag0 / (1 + d / d_mag)²
```
If the yo-yo turns around and leaves past `ell_mouth`, `inside = false`.
**[A2]** If the magnet out-pulls the string (T < 0) the yo-yo goes FREE; the magnet keeps acting in FREE while inside (along `−u_r`, reaction on the hook), and the seat event below is also detected in FREE.

**Step 3, seat** (`|p − H| ≤ ell_min`, moving inwards): `v = −ℓ̇` (or `−v_r` in FREE)
- `v < v_cap_max` → `STOWED`, caught.
- `v ≥ v_cap_max` → rebound `ℓ̇⁺ = e_cup·v`, `ω⁺ = σ·ℓ̇⁺ / r` (forcing the spin onto the constraint is a simplification; the real one slips). In FREE: the radial relative velocity is reversed and scaled by e_cup.
- The hook receives `−m (1 + e_cup) v·u` (§5.3).

### 4.3 Funnel geometry and wall contact **[A4]** (`capture_model: geometry`, default)
Design intent: a seat that fits the yo-yo **loosely**, and a **streamlined (trumpet) mouth**
along which the yo-yo slides down to the seat. In-plane inner half-width:
```
w_s = seat_width / 2,  w_m = mouth_width / 2
w(z) = w_s                                            0 ≤ z ≤ seat_depth
w(z) = w_s + (w_m − w_s)·ξⁿ,  ξ = (z − seat_depth)/(d_cup − seat_depth)    seat_depth < z ≤ d_cup
```
`n = flare_exponent`: n = 1 is a cone (corner at the seat), n = 2 leaves the seat wall
tangentially and steepens to ≈ 51° at the lip with the default numbers (no edge for the rim to catch on).
The same law is used for the out-of-plane section (`seat_thickness → mouth_thickness`), which
the 2-D sim does not simulate.

Contact: the yo-yo is a disc of radius R in the plane; the wall is the polyline w(z)
(both sides, lip included as the end point). For penetration δ along the normal N
(from wall to disc centre):
```
F_n  = max(0, k_wall δ − c_wall v_n)          # c_wall from e_wall for a linear spring-damper
v_t  = (rim-point velocity − wall-point velocity)·t,   rim point = p − R N, rim velocity = ṗ + ω × (−R N)
F_f  = −mu_wall F_n tanh(v_t / 0.01)·t
F_c  = F_n N + F_f t  on the yo-yo,  τ_c = ((−R N) × F_f)_z,   −F_c on link 2 at the contact point (generalized force Jac_cᵀ(−F_c))
```
F_c and τ_c enter §3.2 (TAUT) and §3.5 (FREE). The friction term matters: the climbing
yo-yo spins fast (rim speed ≈ (R/r)·|ℓ̇| ≈ 4.7 |ℓ̇|), so rubbing the wall brakes the
spin — which, through the string constraint, brakes the climb.
FAIL "hit_rim": disc top above the mouth plane (`z − R < d_cup`) while the centre is
outside the inner wall (`|x| > w(min(z, d_cup))`).
The gate thresholds `alpha_cap` are not used in this model; `v_cap_max`, `e_cup` and the
magnet are. If the success rate still disagrees with hand tests, move to MuJoCo convex-hull contact.

## 5. Arm dynamics

### 5.1 Kinematics
- `q1`: link-1 angle from horizontal, joint-1 motor through the 6:1 belt.
- `q2`: wrist angle relative to link 1 = GM3506 motor angle (stator on link 1, so it controls the relative angle).
- `φ2 = q1 + q2`: absolute angle of link 2 (funnel). Dynamics use `(q1, φ2)`; control and sensors use `(q1, q2)`.
```
E   = l1 (cos q1, sin q1)
H   = E + l2 (cos φ2, sin φ2)
Jac = [[−l1 sin q1, −l2 sin φ2],
       [ l1 cos q1,  l2 cos φ2]]         # ∂H/∂(q1, φ2)
```
- Bounce posture: link 1 about horizontal, link 2 down (φ2 = −90°); the wrist counter-rotates (`q2 = −90° − q1`) to keep the funnel pointing down.
- With l₂ ≈ 1.5 cm the hook moves on an arc of radius l₁; ±5 cm stroke → ≈ 0.9 cm horizontal drift, ±10 cm → ≈ 3.8 cm.
- String torque on the wrist `T·l2·sinδ` (δ = link 2 vs. string) is logged as `tau_wrist_rope = Jac[:,1]·F_H`.

### 5.2 Dynamics
```
M(q) q̈ + C(q, q̇) q̇ + G(q) = Q_drive + Q_fric + Jacᵀ·F_H (+ Q_contact)     q = (q1, φ2)
```
Rigid bodies (closed form in `arm/dynamics.py`, checked against the sympy Lagrangian in `arm/derive_sympy.py`):
- link 1: uniform rod `m1`, COM `l1/2`; elbow point mass `m_e` at E; reflected rotor `N1²·Jr1` on `M[0,0]`.
- link 2 (rotates with φ2): adapter rod `m2` (COM `l2/2`), funnel point mass `mc` at `lc` plus `Ic = mc·0.02²`, GM3506 rotor `Jr2` on `M[1,1]` (direct drive: absolute angle φ2, no ratio), and when STOWED the yo-yo point mass `m` at `l2 + ell_min`.
```
q̇2      = φ̇2 − q̇1
f_i(v)   = b_i·v + tc_i·tanh(v / 0.01)
Q_drive  = ( N1·eta1·τ_m1 − τ_m2 ,  τ_m2 )
Q_fric   = ( −f1(q̇1) + f2(q̇2) ,  −f2(q̇2) )
```
External `F_H`: TAUT `(T + F_mag)·u`, plus `F_b(t)` during a pulse; FREE `F_b + F_mag u_r` [A2]; STOWED 0 (merged).

### 5.3 Impulse transfer
Snap (§3.5) and seat rebound (§4.2) apply an impulse `P_H` at H:
```
Δq̇ = M⁻¹·Jacᵀ·P_H,    Λ_H⁻¹ = Jac·M⁻¹·Jacᵀ
```

### 5.4 Arm–yo-yo coupling
With the wrist motor on the elbow the arm's effective mass at H is ≈ 0.18 kg
(`1 / (u·Λ_H⁻¹·u)`), 3–4× the yo-yo: the bottom pulls the hook down noticeably, so from v3
the hook is **not** kinematic.
**[A5]** Implemented as the simultaneous solve (the spec's fallback option) in every
derivative evaluation: `T = T0 − μ (Ḧ·u)` with `μ = m(1 − κ)` and
`Ḧ = a0 + Λ_H⁻¹ F_H` are linear in T, so
```
T = (T0 − μ (a0·u + λ F_mag)) / (1 + μ λ),   λ = u·Λ_H⁻¹·u
```
where `a0` is the hook acceleration with no string force. No one-step lag, no lag oscillation.

## 6. Motors and drivers
Both motors run FOC current control (SimpleFOC Mini v2 / DRV8316); the current loop is a first-order lag:
```
i̇_k  = (clip(i_cmd_k, limits_k) − i_k) / tau_i,     τ_m_k = Kt_k·i_k
```
### Position mode (default action interface)
Firmware PD at the physics rate, on measured angles (§7):
```
joint 1: i_cmd1 = (Kp1 (q1_ref − q1) − Kd1 q̇1) / (Kt1 N1 eta1)
wrist  : i_cmd2 = (Kp2 (q2_ref − q2) − Kd2 q̇2) / Kt2          # relative q2
```
Gravity feed-forward only if the real firmware has it (it does not yet). Host → driver command latency `cmd_delay` (5 ms, roadmap).
### Current and voltage limits
- `i_max_k = i_peak_ratio · tau_max_k / Kt_k` (default ratio 1 = continuous); time above continuous is accumulated for heating estimates.
- **v4** back-EMF (`use_back_emf`): `(−V − Ke ω_m)/R ≤ i ≤ (V − Ke ω_m)/R`, `V_1 = 24 V`, `V_2 = 12 V`, `ω_m1 = N1 q̇1`, `ω_m2 = q̇2`, `Ke = Kt`.
### Joint-1 belt compliance (v4, not implemented yet)
```
τ_belt   = k_belt (θ_m/N1 − q1) + c_belt (θ̇_m/N1 − q̇1)
Jr1 θ̈_m = τ_m1 − τ_belt / (N1 eta1)
```
Link-side `Q_drive[0]` becomes `τ_belt − τ_m2` and `N1²·Jr1` leaves `M[0,0]`.

## 7. Sensors and observations
The policy sees only this section. Cameras are for development, sysid and evaluation.
| Signal | Model | Rate |
|---|---|---|
| joint-1 angle `q1` | motor encoder: `q1_meas = θ_m / N1`, LSB 2π/(16384·N1), 1 ms delay. With belt compliance (v4) it reads the motor side | 1 kHz (sim: every physics step) |
| wrist angle `q2` | direct drive: LSB 2π/16384, 1 ms delay | 1 kHz |
| joint velocities | angle difference + first-order low-pass (100 Hz) | 1 kHz |
| motor currents `i_q` | true + Gaussian σ = 30 mA + per-episode bias U(±20 mA) + 0.5 ms delay | 1 kHz |
| cogging residual (DR) | on τ_m: sinusoids of the motor angle, 1–3 % of continuous torque | — |
| momentum observer (optional) | generalized-momentum observer `τ_ext` estimate | 1 kHz |

Joint-1 current is the main channel that feels the bottom: through l₁ and the 6:1 belt.
The wrist barely feels the string because l₂ is short.
**Privileged** (critic, teacher, logs): `mode, ℓ, ℓ̇, θ, θ̇, ω, σ, T`, predicted time to the next bottom.
Policy rate 200 Hz; stacked history or a GRU.

## 8. Environment interface, logs, visualization
```
state = env.reset(seed, params)                       # params may be randomized
state, obs, priv, reward, done, info = env.step(state, action)
```
- `action`: position mode `(Δq1_ref, Δq2_ref)`, increments of the current reference, clipped. Wrist uses q2 like the firmware.
- One `env.step` = 25 physics steps (200 Hz policy over 5 kHz physics).
- `reset` starts STOWED, or "after the k-th bottom" for skill-wise training.
- `info`: events of the step (bottom time and J, snap, slack, catch outcome), T peak, motor torque peaks, wrist string torque peak.
- Logs: one row per physics step (`sim/log.py:FIELDS`) plus an event table; the same field names as the hardware logger.
- `viz/animate.py`: matplotlib animation (stand, link 1, U-fork and funnel, string, yo-yo coloured by mode) with ℓ(t), T(t), i1(t), i2(t), mode background and event lines.

## 9. Required tests (float64, `tests/`)
1. Free fall (§3.3): `ℓ(t) = ½ κ g t²` within 1e-6 m.
2. Tension (§3.2): hook accelerating up at a → `T = m (g + a)(1 − κ)`.
3. Energy (§3.2, §3.4): fixed hook, no damping, TAUT with swing and r(ℓ), drift < 0.1 % over 2 s.
4. Bottom (§3.6): `E⁺/E⁻ = ζ_e²` (1e-6), σ flips, ω keeps sign, ∫F_b dt = J.
5. Passive bounce (§3.7): apex ratio ζ_e², `h = κ g T_b² / 8` within 1 %.
6. Slack/snap (§3.5): hook at 1.5 g down → FREE; after it stops → snap; e = 0 → `v_r = σ r ω`; with the arm, equal and opposite impulses.
7. Throw (§4.1): holds at rest; releases iff the upward acceleration exceeds `F_mag0/m − g` (≈ 10.2 m/s²).
8. Catch (§4.2, §4.3): aligned and slow → STOWED; fast → rebound `e_cup·v`; gate `β ≥ alpha_cap` → FAIL; magnet only while inside; geometry: wall guides an off-axis yo-yo to the seat, outside the mouth → FAIL.
9. Arm statics and Jacobian (§5.1, §5.2): gravity vs. hand calculation, Jac vs. finite differences, closed form vs. sympy.
10. Relative-angle torque mapping (§5.2): `τ_m2 q̇2 = Q_drive·(q̇1, φ̇2)`; q1 locked → only φ2 accelerates, link 1 receives −τ_m2.
11. Arm energy (§5.2): no friction, no input, no yo-yo → conserved.
12. Zlajpah pulse baseline: ≥ 20 bounces without decay; plus the full throw → 3 bottoms → catch sequence (`tests/test_baseline.py`).

## 10. Implementation order
| Version | Content | Done when | Status |
|---|---|---|---|
| v0 | 1-D, kinematic hook, constant r; FREE/TAUT and bottom | tests 1–2, 4–6 | done |
| v1 | funnel: STOWED, throw, catch; Zlajpah baseline | tests 7, 8, 12; throw → 3 bounces → catch animation | done (`control/baseline.py`) |
| v2 | 2-D swing θ | test 3 | done |
| v3 | arm dynamics, motors, sensors, coupling; env interface | tests 9–11 | done (numpy); JAX + vmap in stage 5 |
| v4 | r(ℓ), back-EMF, belt compliance, cogging | sim vs. real within the stage-4 gates | r(ℓ), back-EMF, cogging available as flags; belt pending |

## 11. Amendments made during implementation
| # | Section | Change | Why |
|---|---|---|---|
| A1 | §3.6 | yo-yo equations exclude the hook acceleration caused by the bottom pulse F_b | the instant reversal already used that impulse; otherwise spurious slack 3 ms after every bottom |
| A2 | §3.5, §4.2 | magnet also acts in FREE while inside; seat event also in FREE | when the magnet out-pulls the string (F_mag > m g near the seat) the string goes slack; without A2 the yo-yo never reached the seat |
| A3 | §3.5 | snap rebound only for Δ ≥ `v_snap_min` (0.05 m/s instead of 1e-3) | wall contact produced endless micro-rebounds |
| A4 | §4.3 | funnel geometry with wall contact, now the default capture model | the "loose seat + trumpet mouth" design needs real geometry to evaluate |
| A5 | §5.4 | T and Ḧ solved simultaneously from the start | exact, no lag oscillation |
| A6 | §2, §4.2 | `inside` is set/cleared by `|p − H|` crossing `ell_mouth`, in TAUT and FREE | needed by A2 |
| A7 | §3.1, §3.5 | in FREE the slack length winds with the spin, `ℓ̇_s = σ r ω` | physical (string tied to the axle); brief slack after a bottom no longer destroys the climb |
| A8 | §4.2 | `inside` is set only when the mouth plane is crossed from outside | right after a release the yo-yo is inside and may move inwards for a moment; it was re-captured by the magnet |

### Known open issues
- Rope–funnel contact is ignored (only the yo-yo rim touches the wall); out-of-plane (3-D) tilt is not modelled.
- The stowed yo-yo is a point mass (its own spin inertia about the axle is not added to link 2).
- Belt efficiency η is applied in both power directions.
