# proprio-yoyo
Actuator-as-sensor dynamic manipulation: a 2-DOF arm throws, bounces, and stows a yo-yo using only motor feedback, trained in a custom JAX simulator and transferred zero-shot.

**Status:** roadmap stage 1 (sim v1) is done. The numpy simulator covers spec versions v0–v3: 1-D/2-D yo-yo, funnel, arm, motors, sensors and coupling, plus an env interface and visualization. A true-state pulse baseline runs the full "throw → 3 bounces → catch" sequence with nominal parameters (step 9).
Next is stage 2: design sweeps and CMA-ES. See "Baseline and margins" below.

Physics reference: [PHYSICS_SPEC.md](PHYSICS_SPEC.md). Every formula in the code cites its section, e.g. `# §3.2`.

## Setup
```bash
conda activate yoyo            # project env (Python 3.11)
pip install -r requirements.txt
python -m pytest               # 51 tests (§9), ~25 s
```
Run everything from the repo root (the folders are not installed as a package). `ffmpeg` is optional; without it animations are written as GIFs.

## Quick start
```bash
python -m scripts.run_demo full_flow               # throw -> 3 bounces -> catch (step 9)
python -m scripts.run_demo full_flow --n-bounces 0 --duration 30   # keep bouncing
python -m scripts.run_demo throw_bounce            # stowed -> throw -> passive bounces
python -m scripts.run_demo funnel_catch            # climbing yo-yo meets the funnel
python -m scripts.run_demo funnel_catch --theta0 9 --v0 0.9 --set mu_wall=0.1
python -m scripts.funnel_sweep                     # capture map: swing angle x climb speed
python -m viz.plot_funnel                          # funnel cross-sections -> runs/funnel_profile.png
python -m viz.animate runs/<run_dir> --speed 0.25  # re-render a saved run (or --show)
```
Browse every run in one page: `python -m scripts.gallery --open` writes `runs/index.html`. Each run gets a card with its outcome, events, the parameters that differ from `config/params.yaml`, `summary.png` and `anim.gif`.

Each demo writes `runs/<time>_<scenario>/` with `log.npz` (one row per 0.2 ms physics step), `events.csv/json`, `params.json`, `summary.png` and `anim.gif`.

## Layout
```
config/params.yaml     every numeric parameter + randomization ranges (TODO(sysid) = placeholder)
PHYSICS_SPEC.md        physics spec (English) + implementation amendments (§11)
sim/
  params.py            Params dataclass, derived quantities (kappa, zeta, ...), sample_params()
  simulator.py         hybrid stepper: RK4, event location, arm-yoyo coupling, KinematicHook
  env.py               YoyoEnv.reset / step  (obs = proprioception, priv = true state)
  log.py               per-step log fields, save_run / load_run
yoyo/
  modes.py             STOWED / FREE / TAUT / FAIL
  dynamics.py          string equations (§3.2), bottom (§3.6), snap impulse (§3.5)
  funnel.py            funnel profile, wall contact, magnet (§4)
arm/
  kinematics.py        H, Jacobians (§5.1)
  dynamics.py          M, C, G, drive and friction forces (§5.2)
  derive_sympy.py      symbolic Lagrangian used to check dynamics.py
  motor.py             current loop, firmware PD, limits, back-EMF, cogging (§6)
  sensors.py           encoder quantization, delays, velocity filter, current noise (§7)
control/
  trajectories.py      open-loop Throw and Hold reference generators
  baseline.py          true-state Zlajpah-style pulse baseline (throw, bounce, catch)
viz/
  animate.py           rig + funnel close-up + l/T/i1/i2 plots, GIF/MP4
  plot_funnel.py       funnel design figure
scripts/
  run_demo.py          demo scenarios
  funnel_sweep.py      capture-outcome table for funnel design
  gallery.py           runs/index.html overview of all runs
tests/                 test_yoyo (1-6), test_funnel (7-8), test_arm (9-11), test_baseline (12), test_env
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
