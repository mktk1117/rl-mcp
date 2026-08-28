# Tuning a run (for AI agents)

How to drive a training run from first launch to a closed record: what to
verify before spending GPU time, which numbers to watch while it trains, how to
turn a symptom into a parameter change, and what counts as evidence.

Everything here was paid for. Across two long campaigns — a humanoid
locomotion roster of ~90 recorded runs and an in-hand manipulation notebook —
**no blocking failure was ever a PPO hyperparameter.** Every one was a broken
task, a broken measurement, or a broken interface. Tuning is mostly
verification and diagnosis; the parameter edit is the last and smallest step.

The loop, and where each step's tools are documented:

| step | tools |
| --- | --- |
| 1. verify the task before training | [`check`](tools.md#check), [`shot`](tools.md#shot) |
| 2. open the record | [`record new`](records.md) |
| 3. watch the right numbers | [`status`](tools.md#status), [`metrics`](tools.md#metrics), [`plot`](tools.md#plot), [`curriculum`](tools.md#curriculum) |
| 4. look at the robot | [`shot`](tools.md#shot), [`video`](tools.md#video), [`play`](tools.md#play) |
| 5. measure smoothness | [`diagnose`](tools.md#diagnose), [`trace`](tools.md#trace), [`plot-trace`](tools.md#plot-trace) |
| 6. diagnose the mechanism | reward breakdown, termination mix, decomposition |
| 7. change one thing, safely | [`checkpoint`](tools.md#checkpoint), [`set --why`](tools.md#set), [`load`](tools.md#load) |
| 8. close the record | [`record close`](records.md), `record compare` |

## 1. Verify the task before training on it

A wrong task trains fine. It produces curves, checkpoints and plausible
metrics, and tells you nothing. Each check below is cheap and each caught a
real defect.

**`rlmcp check --task <id>` does the first three of these**, and its exit code
is the verdict: it builds the task with no policy, rolls it, and reports
imports / constructs / steps / rewards finite / terminations sane / trains,
followed by what each reward term paid. What follows is what it checks and why
each one earned its place.

**Roll zero actions first.** ~150 passive steps: the robot holds its keyframe
pose, held objects stay put, nothing terminates. This catches reset-ordering
bugs, actuator gains sized for the wrong limb (stock arm gains under a hand
twice the design mass drooped 0.49 rad — every lobbed ball landed where the
palm used to be), and an action offset centred on servo mid-range so that
"do nothing" meant "open the hand".

**Roll random actions too.** Limbs move freely, nothing self-penetrates.
Identical random-action rollouts are also the cheapest A/B for an interface
change: an action filter was settled before any training run — 130 drops
unfiltered vs 2 filtered, peak joint speed 70 → 43 rad/s.

**Take one training iteration before you buy thirty thousand.** A zero-action
rollout never constructs a policy, so it never reads the task's `rl_cfg` at
all — and half of what kills a run at iteration 0 lives there. One task passed
every environment check and then died on its first act: its actor had no
`distribution_cfg`, so it was built deterministic and PPO had no distribution
to take a log-prob from. `rlmcp check` builds the runner and takes one
iteration for this reason (the `trains` gate, on by default, under a second on
CPU). Nothing cheaper can see it, because nothing cheaper opens the agent
config.

**Screenshot the running scene, not the compiled spec.** A standalone
`spec.compile()` passes while the live scene is wrong; one full run trained on
a hand mounted as a vertical claw. Take `rlmcp shot` at iteration 0 and look.
Renderer traps to know: neighbouring envs composite into close-ups unless the
viewer's extra-envs is 0, and fixed-base entities stack at the world origin
unless a reset event places them — physics fine, every frame misleading.

**Ask the do-nothing question of every number.** *If the policy learned to do
nothing, which logged number would go down?* If the answer is "none", add that
number before launching. Then roll the zero-action policy through the full
eval pipeline and confirm every metric you intend to cite calls it bad.
Reward and episode length routinely endorse a broken run:

- A curriculum promoting on displacement-from-spawn met a command direction
  that resampled every few seconds; every env was demoted to flat ground while
  **reward climbed the whole time** — it was reward earned on flat.
- A task reward `0.5·vel + 0.5·facing` paid half its value for standing still
  with correct heading. Reward climbed and episode length climbed; only the
  curriculum, which measured ground actually covered, caught it.

Ask the question of every gate and assist signal, not only rewards. A
promotion gate a passive policy passes measures nothing — set gates from a
measured passive baseline (a do-nothing hand already caught 52% of slightly
jittered lobs; the gate went above the passive score). Metric traps that
recur: "fraction that ever failed" grows with rollout length (report rates per
sim-minute); a holding metric read on the current step reads ~0.9 on a policy
that drops constantly, because freshly reset envs all count as holding — gate
on survival as well.

**Measure a reward term before wiring it in.** Log its raw per-step magnitude
against the task reward over a short rollout. mjlab's additive
`action_rate_l2` at its default weight would have been 45× one task's entire
signal — which is why style terms default to multiplicative `exp(−w·cost)`
gates that cannot swamp the task at any scale. Then assert every configured
term is non-zero at least once in the first 50 iterations: a term reading
exactly 0.0000 forever is disconnected — a touch sensor left at its default
0.005 m site volume, a limit penalty computed on the hard range the solver
already enforces — and its presence reads as coverage. Ship optional
regularizers at weight 0.0: skipped per step, but live-tunable later without a
restart.

**Check the ask is physically feasible.** Commands must be feasible on the
geometry they are issued on — a strafe command on a narrow beam is an
instruction to step into a hole (measured: commands pointed ~90° off the
walkable surface before the fix). A reference trajectory's torque demand comes
from inverse dynamics, not forward simulation: a saturated actuator clips, so
a forward-sim figure can never exceed 100% and once hid a 13× overdemand
behind "76–99% of limit". And check the velocity at the moment that matters: a
positions-only reference arrived at its release point at 0.95 m/s where the
throw needed 2.36, capping any faithful tracker. For imported motion data,
replay the raw frames through the robot's real collision model first — one
corpus self-collided 31% of the time before curation.

**Histogram the observations.** ≥512 envs, hardest condition, read
p1/median/p99. Height-scan miss rays saturating at 5 m put 2.4% of the input
at an outlier while genuine terrain variation occupied ~7% of the range — the
network spends its input range on noise. If geometry generation is procedural,
audit it offline by importing the actual generator and reading back what it
built; nine such audits found unwalkable slopes, front-loaded gaps and
infeasible spawn poses that no training curve would ever have named.

## 2. Open the record before you launch

The full lifecycle is in [records.md](records.md): hypothesis, prediction,
falsifier, one conceptual change per run. Two rules that repeatedly bit:

**Write falsifiers in absolute iteration numbers.** A warm start resumes the
counter — one run's first iteration was numbered 18001 — so "check after 300"
never fires.

**The config snapshot is the resolved config, not the source file.** A
regularizer introduced mid-campaign and absent from the baseline silently rode
along in every subsequent run, making each "one-change" comparison secretly
two changes; one PLAN even asserted the gate was off while the launch config
said 0.004. `rlmcp` snapshots resolved parameters at launch for exactly this
reason.

A multi-change run is legal only as a declared baseline reset. Its result is
directional; attribution comes from separate audits, not from the run.

## 3. Watch the run — in order of trustworthiness

1. **Curriculum / stage level.** Task-external, and it keeps absorbing
   improvement after reward and episode length saturate. Stop at *its*
   plateau, not the reward's.
2. **Termination breakdown.** A stalled run has a reason and the mix names it:
   self-collision terminations at 1.83/env/min vs 0.75 for real falls said
   "the terminator, not the policy". A healthy locomotion mix has `time_out`
   dominant.
3. **Normalised per-step tracking errors.**
4. **Reward and episode length** — necessary, never sufficient. See the two
   cases above where both endorsed a broken run.

Poll `rlmcp curriculum` and compare **stage index, not name** — a restart
re-enters at rung 0 and must not read as a promotion. Read each promotion
condition's current value and streak; a gate whose signal has high variance
can be statistically unreachable (one gate's signal had stdev 0.376 because it
was sampled once per iteration mid-swing; with 25 consecutive holds required,
promotion probability was zero for any policy — the fix was averaging over
every step, not lowering the bar).

The run films itself while it trains -- [progress
clips](tools.md#progress-clips-the-ones-you-do-not-have-to-ask-for) at
iteration 0, 50, 100, 200, 400 ... each one attached to the record -- so the
trajectory exists without a watcher script. Still verify the first clip
actually landed (`rlmcp video --schedule`, or the `progress_clip` events): one
hand-rolled wall-clock-keyed watcher produced zero clips across two long runs,
and every clip that existed had been captured by hand. A schedule that is
silently failing says so in `progress_clip_skipped` rather than in the absence
of files nobody checked for.

Three caveats on curriculum level as evidence: it is blind to falls (one
terrain family held the second-highest level and the worst fall rate — grade
on falls/min at pinned difficulty); its mean is compressed by ceiling
recycling (a mastered family cannot read near max); and a mean across families
cannot distinguish "competent everywhere" from "saturated on the easy half,
pinned on the hard half" — split it per family.

## 4. Look at the robot

**A result that has not been watched has not been verified.** Frames catch
what every scalar endorses:

- An open-stairs "PASS" (0.148 tracking MAE, 0.000 falls/min) was the policy
  walking the *ground under the floating stair treads*, torso threading the
  slab gaps. Root height 0.244 m with 42.7° knee flex flagged it; one rendered
  frame confirmed it; it voided every stairs number from three prior runs.
- A curriculum level ratcheting 0.67 → 4.8 was flat-lane walking beside the
  obstacle walls — zero successful climbs. The level was a lie, visible in one
  screenshot.

**Compare like with like.** A training capture (pushes on, randomisation on,
envs sitting on unearned terrain rungs) is not an eval render (pushes off,
difficulty pinned). One reported "gait regression" dissolved into exactly this
render-condition mismatch. `rlmcp play` replays the run's curriculum and
parameter history before rolling out, and refuses to render misleading
evidence when it cannot — see [why play replays the run
first](tools.md#play). During training, render clips at the run's *current*
stage and assist settings, read live from `rlmcp status`: a clip of an
assisted rung rendered unassisted shows a policy failing at a task it is not
being asked to do yet.

**Ask counterfactuals with one knob.** Two renders of one checkpoint —
`--set command.assist=1.0` vs `0.0` — measured 75 vs 0 catches/min and
settled in minutes what the training curves could not.

**Attach what you rendered.** Progress clips file themselves; anything *you*
rendered -- a counterfactual, an eval render, the final `play` -- does not. A
clip left in `logs/` is not a deliverable; register it on the record so the
record views can show it.

## 5. Chatter and jerkiness are measured, not seen

**Chatter is invisible in video and absent from reward.** A policy can pay
`action_rate_l2` and still buzz; buzzing rides *on top of* a healthy 1–3 Hz
gait rather than replacing it, which is why [`diagnose`](tools.md#diagnose)
measures the share of joint-velocity power above the gait band, plus jerk RMS,
effort, posture and gait, and returns a verdict naming the lever. Read the
verdict's distinction seriously: one loud joint and whole-body HF are
different problems with different fixes (chase the joint vs raise
`action_rate_l2` / lower the action scale). Absence of a finding is reported
as not-measured, never as clean.

Reference bars from the locomotion campaign: HF share mean ≤ 0.15, worst
joint ≤ 0.40. Chatter in wrists is cosmetic; chatter in ankles and hips
predicts foothold failures. Aggregate scalars cannot see style — two runs
where `joint_acc_rms` said "marginally smoother" disagreed with both the
spectra and the human watching. When smoothness matters, pull joint
position/velocity/torque traces ([`trace`](tools.md#trace), then
`plot-trace`): they catch what neither reward nor video reliably shows.

**If everything is jerky, suspect the action interface before the reward.**
Four consecutive failed runs read as reward-design problems — a freeze
optimum, then drop-fests — and were all one flaw: policy output mapped to
absolute joint targets over the full range, re-sampled every control step. A
first-order filter on the *target* (`target ← (1−α)·target + α·(offset +
scale·action)`, α = 0.2; the policy keeps its full range) took goals/min
1.1 → 16.8 with nothing else changed, and HF share 0.70 → 0.13 — smooth
without a smoothness fight. The corollary cost 300 iterations: a smoothness
calibration is only valid in the regime it was measured. The old weights,
honest under raw actions, charged 10× task income under the filter (−2.79/s
against +0.27/s earned) and rebuilt the freeze.

## 6. Symptom → mechanism → lever

Diagnose before tuning; the measurement picks the lever. The recurring cases:

| symptom | measure first | usual mechanism | lever |
| --- | --- | --- | --- |
| reward climbs, behaviour wrong | a task-external metric; a video | a term pays for something free | gate reward by the task signal: `vel·(0.5 + 0.5·face)`, not `0.5·vel + 0.5·face` — no travel, no reward |
| freezes, stands, holds still | per-term reward breakdown | existence out-earns progress (one term paid +0.52/s for existing vs +0.0002/s for progress — 2600×; any risk was irrational) | rebalance so no term out-earns another ~2× at the error the policy actually reaches; add a potential-based progress term |
| drops / bails out instantly | price of failure vs task income | failure priced near-free | price a drop against a goal (~2:1), neither free nor fatal |
| joints thrash from iteration 0 | `action_rate_rms` vs the action range | the interface, not the reward | filter the target (§5); re-measure smoothness weights afterwards |
| a metric plateaus exactly at a threshold | the threshold | a hard gate teaches clearing the bar (throw apex climbed 0.085 → 0.150 and stopped *on* the 0.150 gate) | a dense signal for the behaviour, not a higher bar — a higher bar pins too |
| curriculum never promotes | each condition's value, streak, and variance; termination mix | gate statistically unreachable, or the bar sits one skill above the frontier, or difficulty 0 is not achievable | per-step averaging; put the bar at the frontier; add an onramp |
| level climbs and falls climb too | falls/min at pinned difficulty | promotion buys speed with falls (level measures ground covered) | grade on falls; OR the failure into demotion |
| one objective simply ignored | per-term reward shares | 15× spread between terms → rational neglect; a mean-over-joints term weighted like a sum paid 0.2% of budget | weight from measured per-step magnitude, not intent |
| "the robot can't physically do it" | decompose the error; probe the ceiling | phase, not magnitude (speed matched at 0.96 m/s while direction cosine read 0.24); a full-effort probe showed 3.5× headroom | make phase observable; fix the reference, not capacity |
| a metric flat at exactly 0.0000 | the term's sensor and wiring | disconnected instrument | fix the wiring; make missing sensors raise, not return zeros |

**Suspect the instrument before the policy.** One promotion gate averaged a
running maximum over balls still in flight and read 0.11 while completed
flights measured 0.255 — three runs were diagnosed against the wrong number
while a second metric contradicted it the whole time. When two numbers
disagree, reconcile them before tuning anything: recompute episode reward from
the term sums, and if it does not reproduce within a few percent, one of your
instruments is lying. And when the question is "what does this system actually
do", instrument the system rather than reimplementing it — three offline
reconstructions of one curriculum rule were all wrong in different ways; a
frozen-policy probe through the real env settled it in eight sim-minutes.

## 7. Change one thing, safely

- **Checkpoint before the experiment.** `rlmcp checkpoint before-experiment`,
  and treat rollback as the undo button it is — it restores parameters, stage
  and extension state along with the weights.
- **Every `set` carries `--why`.** It costs nothing and turns the event log
  into something you can re-read: what changed, when, and the reason.
- **Trust refusals, then verify the change landed.** A refused write is the
  harness telling the truth about a knob that cannot work. One run lost 500
  iterations to two weights a bound had refused — their terms showed exactly
  0.0000 in the reward breakdown the whole time. After any edit, confirm the
  next reward-breakdown read shows the term alive.
- **One conceptual change per run.** Two changes make an ambiguous result.
- **A warm-started result proves a change *preserves* a behaviour, never that
  it creates one.** Two configs credited as breakthroughs failed when
  retrained from scratch — the warm start was load-bearing, and the honest
  recipe was "generalist pretrain + specialist finetune". Warm starts also
  carry habits: from-scratch retrains beat their warm-started teachers twice.
  The record store enforces the epistemics: warm-started runs cap at
  `provisional`.
- **Size weights from measurements.** A posture gate at w=4 charged a factor
  the policy simply paid (×0.55); measuring the cost at the bad behaviour and
  pricing it to ×0.22 fixed posture *and* cut falls 5× — the crouch had been
  costing stability too. Same rule for potentials: measure the per-step delta
  of each error and set weights so the terms speak at comparable volume.

## 8. Findings that transfer

Condensed from both campaigns. Each line survived at least one falsification
attempt; treat them as strong priors, not laws.

**Reward shaping**

- Style and regularity terms default to multiplicative `exp(−w·cost)` gates —
  they cannot swamp the task signal at any scale and relax toward 1 as
  behaviour improves. Additive penalties only after measuring magnitude.
- Bounded kernels are flat far from the target (`exp(−err/0.10)` at 0.15 is
  0.22 and nearly flat): a parked policy feels no pull however big the weight.
  Pair the kernel (pull when close) with a potential-based progress term (pays
  only for closing distance, exactly 0 at a standstill, telescopes so it
  cannot distort the optimum).
- A term on `|v − v_ref|` scores a statue better than an out-of-phase mover —
  the nearest descent direction is to stop. Use speed-weighted cosine
  alignment; a statue scores exactly 0.
- Regularize posture toward the *data*, not the robot's zero pose — the
  reference walk genuinely carries ~5° of lean, and pulling to zero is its own
  unnaturalness.
- Shape reward on the body the behaviour comes from. A throw came from fingers
  and wrist while the ball left at +2.46 m/s and the palm moved at −0.45;
  three runs of reward on palm position never moved it.
- Once success also ends episodes, a blanket termination penalty charges the
  policy for winning — bill the specific failure terminal.
- Reward *relaxations* are global trades exactly like penalties: relaxing a
  style prior on the legs halved falls on the target family and worsened
  tracking on 14 of 17 families, flat included.
- Per-foot placement shaping was a global trade in both directions (two
  opposite mechanisms, identical collateral on families the term never
  charged). The leak-free home for such a term was a single-family specialist.

**Curriculum**

- A rung changes one thing, states what must be true to leave, and holds for
  consecutive iterations past a floor. "When metric X passes Y, change Z" is a
  `CurriculumStage`, not a note to yourself — see
  [curriculum.md](curriculum.md).
- Buy regularization with competence, never before it: nothing that punishes
  motion before motion exists; ramp smoothness weights only once the task is
  being solved.
- Start hard-exploration skills by *giving* the policy the thing it cannot
  stumble into (lob the ball onto the palm; ask for the throw once catching is
  reliable), then fade the assist on evidence — and a missing signal must make
  the fade controller *hold*, not read as failure.
- Precision is a difficulty knob, not a starting condition — and tightening
  can *raise* measured competence (43% → 50% at a tighter band: the loose band
  was hiding precision the policy already had). Stop at a principled floor;
  tighter than the literature's threshold is a worse task, not a better one.
- Promotion wants several simultaneous conditions — survives AND climbs AND
  tracks — because each alone is buyable: episode length promotes standing,
  level promotes ignoring the command.
- Fall-demotion is right for walking-reward artifacts and wrong for skill
  acquisition (every failed attempt demotion-punished starves the learning
  tier) — but removing it opens promotion exploits; the durable fix was
  commands that make covered ground *mean* the skill.
- Curriculum state is not in the checkpoint. A resumed run re-climbs from the
  bottom; level curves are not continuous across a resume, and the re-climb
  looks like a spectacular learning rate that is nothing of the kind.

**Exploration and action noise**

- The entropy coefficient was the one PPO knob that repeatedly mattered: at
  0.005 the policy std ballooned 1.2 → 2.0 while success fell 0.26 → 0.14 in a
  20-dim action space. Cut it (0.0015) and validate live.
- Action noise is physical. σ = 0.30 on joint targets is ~10 cm of foot
  scatter — invisible until the task feature (a 12 cm gap) is smaller than the
  noise. The low-noise finetune (σ init 0.12, std ceiling ~0.15, entropy
  0.0015) took the worst family's falls 2.19 → 0.10/min.
- On a warm start the std *ceiling* is the load-bearing part — the checkpoint
  restores its old std over your `init_std`; only a running clamp enforces the
  change.
- Discovery tasks need noise that walking specialists clamp away — one
  climbing task raised the ceiling back to 0.25, deliberately, for that task
  only.
- Adaptive-LR collapse (lr < 1e-4 in the first 50 iterations) means too few
  envs → noisy KL, not a schedule choice. Add envs or fix the schedule; do not
  raise the initial LR.

**Terminations**

- Costs, not terminations, for recoverable contact: a hard self-collision
  termination charged normal leg brushing as full episode failure and capped
  the curriculum; a force-thresholded cost broke the ceiling in 120
  iterations.
- A tight terminator silently caps the skill: a 0.12 m object-drop radius
  clipped legitimate fingertip manipulation at 6–9 cm — which is why three
  successive fall-penalty raises did nothing.
- Raise tilt limits where the skill requires them (a mantle pitches the trunk
  past a walking-gait tilt terminator).

**Domain randomization**

- Mask pushes where recovery is impossible. A lateral kick shifts the capture
  point ~0.14 m; on stepping stones that recovery step is a hole, so at stock
  cadence essentially no crossing ever completed un-kicked and the family
  never learned.
- Startup-mode friction inside a per-env curriculum is a permanent handicap:
  22% of envs drew μ < 0.5 *forever* and could never promote, dragging their
  family's level down. Prefer resampling at reset, or an explicit mixture.
- No friction randomization at all makes sliding a reliable strategy.

**PPO**

- The recipe that served every task: clip 0.2, 5 epochs, 4 minibatches,
  adaptive lr 1e-3 with desired KL 0.01, γ 0.99, λ 0.95, 24 steps/env, grad
  norm 1.0, elu, observation normalization. Width scales with the problem
  (512-256-128 for one hand; 1024-512-256 for 61 actions). Give the critic
  privileged state and keep the noise on the actor's observations only.
- In ~90 recorded runs these were never the culprit. When a run fails, look at
  the task, the measurement, and the interface first.

## 9. Cheap controls worth running

- **Reference control.** Run the stock configuration on your setup before
  concluding "my change is bad" — one short run separates "my change" from
  "my environment".
- **Degenerate-policy check.** Zero-action rollout through the full eval
  pipeline; every metric must call it bad (§1).
- **Identical-random-action A/B** for any interface change (§1).
- **Frozen-policy probe.** Run the real env with learning off so only the
  curriculum/assist state moves — the direct measurement that beat three wrong
  reimplementations.
- **≥3 seeds on procedural terrain,** or label the number one-seed: the same
  checkpoint measured 0.37 and 3.16 falls/min on two seeds.
- **Compare at matched iterations** (`rlmcp record compare --at-iteration`).
  "Better at the end" often means "trained longer".

## 10. Close the record

Outcome in the first line: prediction met, missed, or ambiguous. `falsified`
is a good outcome — a run that killed its hypothesis in 20 minutes is
information-dense; every failure mode above was found by a cheap control, not
a long run. Evidence is the metric table at matched iterations plus the video
and the plots, attached to the record. Diagnosis is a mechanism backed by a
measurement — "self-collision was the dominant termination at 1.83 vs 0.75
for falls" beats "it seemed unstable". Then the belief update, and the single
next change with its falsifier.

Write the report from the artifacts, not from memory of what you did — the
agent that drove the run is holding the hypothesis it wants confirmed. Read
the plots and the event log back before writing; better, let a second agent
write it from the run directory alone.
