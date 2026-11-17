# proprio-yoyo
Actuator-as-sensor dynamic manipulation: a 2-DOF arm throws, bounces, and stows a yo-yo using only motor feedback, trained in a custom JAX simulator and transferred zero-shot.

**Status:** roadmap stage 1 (sim v1) is done. The numpy simulator covers spec versions v0–v3: 1-D/2-D yo-yo, funnel, arm, motors, sensors and coupling, plus an env interface and visualization. A true-state pulse baseline runs the full "throw → 3 bounces → catch" sequence with nominal parameters (step 9).
Next is stage 2: design sweeps and CMA-ES. See "Baseline and margins" below.

Physics reference: [PHYSICS_SPEC.md](PHYSICS_SPEC.md). Every formula in the code cites its section, e.g. `# §3.2`.

**Working on hardware or firmware?** Start with [docs/EMBEDDED_HANDBOOK.md](docs/EMBEDDED_HANDBOOK.md). It also
holds the project's progress log: update its §1 with every commit.

## Setup
```bash
conda activate yoyo            # project env (Python 3.11)
pip install -r requirements.txt
python -m pytest               # 86 tests (§9), ~40 s
```
Run everything from the repo root (the folders are not installed as a package). `ffmpeg` is optional; without it animations are written as GIFs.

## Quick start
```bash
python -m scripts.run_demo full_flow               # throw -> 3 bounces -> catch (step 9)
python -m scripts.run_demo full_flow --n-bounces 0 --duration 30   # keep bouncing
python -m scripts.run_demo full_flow --hand flip   # flip hand: turn-down throw -> 4 bounces -> catch -> flip stow
python -m scripts.run_demo throw_bounce            # stowed -> throw -> passive bounces
python -m scripts.run_demo funnel_catch            # climbing yo-yo meets the funnel
python -m scripts.run_demo funnel_catch --theta0 9 --v0 0.9 --set mu_wall=0.1
python -m scripts.run_demo flip_stow               # flip hand: yo-yo in the seat, turn 180 deg
python -m scripts.run_demo flip_stow --direction wrong   # turning the other way drops it
python -m scripts.run_demo flip_catch --theta0 0   # flip hand: climb into the cup, then flip
python -m scripts.funnel_sweep                     # capture map: swing angle x climb speed
python -m scripts.mech_sweep run --n 1000          # stage 2: 1000 randomized full flows per hand (~15 min, 12 cores)
python -m viz.plot_funnel                          # funnel cross-sections -> runs/funnel_profile.png
python -m viz.animate runs/<run_dir> --speed 0.25  # re-render a saved run (or --show)
```
Browse every run in one page: `python -m scripts.gallery --open` writes `runs/index.html`. Each run gets a card with its outcome, events, the parameters that differ from `config/params.yaml`, `summary.png` and `anim.gif`.

Each demo writes `runs/<time>_<scenario>/` with `log.npz` (one row per 0.2 ms physics step), `events.csv/json`, `params.json`, `summary.png` and `anim.gif`.

## Layout
```
config/params.yaml     every numeric parameter + randomization ranges (TODO(sysid) = placeholder)
config/hand_magnet.yaml  hand overlay: magnet catch (= the defaults)
config/hand_flip.yaml    hand overlay: flip catch (§4.4) + controller settings (`controller`, `baseline` blocks)
PHYSICS_SPEC.md        physics spec (English) + implementation amendments (§11)
sim/
  params.py            Params dataclass, derived quantities (kappa, zeta, ...), sample_params()
  simulator.py         hybrid stepper: RK4, event location, arm-yoyo coupling, KinematicHook
  env.py               YoyoEnv.reset / step  (obs = proprioception, priv = true state)
  log.py               per-step log fields, save_run / load_run
yoyo/
  modes.py             STOWED / FREE / TAUT / FAIL / IN_CUP
  catch.py             dispatch on catch_mode, shared seat rebound
  catch_magnet.py      magnet catch: stow, release (§4.1-§4.2)
  catch_flip.py        flip catch: IN_CUP slide model with the arm (§4.4, A9)
  dynamics.py          string equations (§3.2), bottom (§3.6), snap impulse (§3.5)
  funnel.py            funnel profile, wall contact, magnet (§4)
arm/
  kinematics.py        H, Jacobians (§5.1)
  dynamics.py          M, C, G, drive and friction forces (§5.2)
  derive_sympy.py      symbolic Lagrangian used to check dynamics.py
  motor.py             current loop, firmware PD, limits, back-EMF, cogging (§6)
  sensors.py           encoder quantization, delays, velocity filter, current noise (§7)
control/
  trajectories.py      open-loop Throw, ThrowFlip and Hold reference generators
  baseline.py          true-state Zlajpah-style pulse baseline (throw, bounce, catch)
  stow_flip.py         flip hand: detect the seated yo-yo, 180 deg cosine flip
  full_flow.py         chains the controllers for either hand; catch_report() metrics
viz/
  animate.py           rig + funnel close-up + l/T/i1/i2 plots, GIF/MP4
  plot_funnel.py       funnel design figure
scripts/
  run_demo.py          demo scenarios
  funnel_sweep.py      capture-outcome table for funnel design
  mech_sweep.py        stage 2: randomized full flows in parallel, staged score, load distributions
  gallery.py           runs/index.html overview of all runs
  golden.py            record / check bit-identical magnet full-flow logs (refactor guard)
tests/                 test_yoyo (1-6), test_funnel (7-8), test_arm (9-11), test_baseline (12), test_env,
                       test_catch_flip (§4.4 F1-F4)
```

## Using the simulator
```python
from sim.params import Params
from sim.simulator import Simulator
from control.trajectories import Throw

p = Params.load(mu_wall=0.2)          # YAML values + overrides
sim = Simulator(p)
sim.reset_stowed()                    # or sim.reset_taut(ell=0.4, elld=-0.8, theta=0.1)
sim.run(3.0, Throw(p.l1))             # controller(sim) -> (q1_ref, q2_ref), called at 200 Hz
log = sim.log.arrays()                # dict of numpy arrays, fields in sim/log.py
sim.events                            # [{'t':..., 'type': 'release'|'snap'|'bottom'|...}, ...]
```
The env uses the interface the JAX version will keep:
```python
from sim.env import YoyoEnv
env = YoyoEnv(randomize=True)                 # domain randomization from params.yaml
state = env.reset(seed=0)                     # or start="after_bottom", k_bottom=2
state, obs, priv, reward, done, info = env.step(state, action)   # action = (dq1_ref, dq2_ref)
```
For unit tests or quick studies, `Simulator(p, hook=KinematicHook(...))` prescribes the hook motion instead of simulating the arm.

## Funnel model
`capture_model` in `params.yaml` selects how catching is modelled:
- `geometry` (default): seat with clearance plus a trumpet flare (`flare_exponent`, 1 = cone). The yo-yo rim gets penalty contact with friction against the wall (§4.3).
- `gate`: the spec's threshold model (§4.2), with `alpha_cap`, `v_cap_max` and `e_cup`.
- `none`: no funnel.

## Hands: magnet or flip
`catch_mode` picks how the caught yo-yo is held. Load a hand with `Params.load(hand="magnet" | "flip")`;
the overlay files in `config/` change only the parameters in the §4.4 table.
- `magnet` (default): a magnet in the seat holds the yo-yo; the throw pulls it out (§4.1–§4.2).
- `flip`: the yo-yo sits on the wrist axis (`l2 < 0`), no magnet. After the catch, `control/stow_flip.py`
  turns the hand 180° so the mouth faces up. In the cup the yo-yo slides along the funnel axis (`IN_CUP`).
  The flip direction must follow the winding sign σ. The other direction unwinds the string and the
  yo-yo falls out (amendment A9).

Both hands run the same full flow through `control/full_flow.py`:
```python
from control.full_flow import make_full_flow, reset_full_flow, catch_report
p = Params.load(hand="flip", i_peak_ratio=2.0)
sim = Simulator(p); ctrl, base = make_full_flow(p); reset_full_flow(sim)
sim.run(13.0, ctrl); catch_report(sim)     # arrival speed, swing angle, wrist current peak
```
The flip hand has no magnet, so its throw is "turn the mouth down" (`ThrowFlip`) and the yo-yo has to reach
the seat on its own. Its baseline settings (`apex_target`, `a_up`, `a_down`, throw timing) are the `baseline`
block of `config/hand_flip.yaml`; the magnet hand uses the `BaselineConfig` defaults. The flip hand bounces
4 times, not 3: the stow flip has to turn with the winding sign σ (A9), σ alternates at every bottom, and an
even count turns the hand back the way the throw came, so the cup never passes link 1 and the wrist makes no
net turn per cycle. Nominal flip result: arrival 0.32 m/s with −0.9° swing, wrist current at its 2× limit
during the flip.

## Mechanics sweep (stage 2)
`scripts/mech_sweep.py` draws parameter sets from the `randomization` block of `config/params.yaml`, runs the
full flow of both hands on the same draws (`ProcessPoolExecutor`, one row per case in
`runs/sweeps/<name>/results.csv`) and plots load distributions, a stage funnel and score-vs-parameter curves.
`--scale` shrinks the ranges towards nominal. Besides stowed / not stowed every run gets a staged score, 0.25
per stage passed in order: `throw` (released, first bottom reached), `bounce1` (first apex in range, second
bottom), `bounces` (the rest, up to the last bottom), `catch` (STOWED).

Results with the nominal controller, current limit 2× continuous (1000 runs per hand at scale 1 and 0.1, 200
at the others; flip hand = 4 bottoms, magnet hand = 3):

| scale | magnet stowed | magnet score | flip stowed | flip score |
|---|---|---|---|---|
| 0.05 | 78.5 % | 0.95 | 92.0 % | 0.96 |
| 0.1 | 30.5 % | 0.83 | 56.1 % | 0.85 |
| 0.2 | 6.0 % | 0.76 | 23.5 % | 0.77 |
| 0.35 | 2.5 % | 0.68 | 6.5 % | 0.64 |
| 1 (full ranges) | 0.3 % | 0.37 | 0.3 % | 0.38 |

- The throw never fails. The catch is the first stage to go (already at 5–10 % of the ranges); the bounces
  hold until about 20 %. At the full ranges only 35–45 % of the runs get past the first bounce.
- The flip settings were re-tuned on 60 of these draws (scale 0.1) and the magnet settings were not, so the
  table flatters the flip hand somewhat.
- Most influential parameters at every scale: string damping `c_l` and command latency `cmd_delay`. Both
  ranges are one-sided around the nominal value (`c_l` nominal 0, range 0–0.5 1/s), so every draw is "worse
  than nominal" for a controller that assumes no damping. The others are within noise at this sample size.
- Loads at the full ranges, all runs (median / p95 / max): string tension 3.7 / 6.4 / 20 N; joint-1 torque
  at the joint 1.5–1.6 / 1.6–1.7 / 1.7 N m; joint-1 motor speed 660–730 / 700–790 / 830 rpm; wrist motor
  speed about 500 rpm during the flip; hook speed lost at a bottom 0.65–0.7 / 1.1–1.15 / 1.4 m/s; hook pulled
  below coasting 10 / 20–22 / 28 mm.
- Joint 1 sits at the current limit: the motor torque peak is ≥ 1.5× continuous in 99.9 % of the runs
  (≥ 1.95× in 41 % magnet, 99.5 % flip), for 0.30–0.35 s per run at p95. The 1.5× torque margin the roadmap
  asks for is not there with N1 = 6; the hook drag passes the 2 cm gate in 5 % (magnet) and 10 % (flip) of
  the runs. Runs that fail early never reach the catch, so catch and flip loads are under-represented at
  scale 1 (flip hand at scale 0.1: tension p95 12 N, wrist at its 2× limit in every run).

## Baseline and margins (stage 1 step 9)
`control/baseline.py` reads the true yo-yo state, which is the "cheating" stage-1 baseline. In the hook frame,
`dE/dt = m·a_hook·ℓ̇`, so each pulse speeds the hook up before the bottom (`a_up`) and
slows it after the bottom (`a_down`, which must stay below g or the string goes slack). The size of each pulse comes
from an energy plan whose efficiency `eta` is adapted after every bounce. Because the firmware PD has no
velocity reference, the controller adds `(Kd·q̇_des + J·q̈_des)/Kp` to the reference itself. It also plans the
known drag of the bottom impulse (`J·λ_H`) into the reference, so the PD does not fight it.

Results with nominal parameters (`python -m scripts.run_demo full_flow`, `tests/test_baseline.py`):
- throw → 3 bottoms → catch succeeds for every sensor-noise seed tried (12/12);
- the bounce apexes land within 5 cm of the target, and the baseline holds ≥ 20 bounces without decay;
- this needs about 2× the continuous joint-1 current for about 0.2 s in total, hook pulses up to about 1.85 m/s, and about 15 cm of hook stroke.

The margin is thin. With ±10 % parameter randomization only about 2/12 runs catch. The bounces still work, but the
catch pulse needs more hook speed than the ±10 cm stroke allows. Two things drive this: the bottom impulse
drags the hook down by about 0.8 m/s (the arm's effective mass at H is only about 0.11 kg in that posture), and the
yo-yo loses 1 − ζ_e² ≈ 26 % of its energy at every bottom. These are stage-2 design questions: arm length and stroke,
joint-1 ratio and inertia, the r0/ζ of the yo-yo, `v_arrive`, and the peak current.
