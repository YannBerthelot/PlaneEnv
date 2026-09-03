# Measurement protocol for learned policies

The library exists to ask whether reinforcement learning can beat classical
control on tasks that look like industrial ones. That question is easy to answer
badly: almost every choice below can be made in a way that decides the result
before any training starts. This page fixes those choices in advance and says
why each one is what it is.

Nothing here presumes an answer. The protocol has to be one that would convince a
reader who wanted the opposite conclusion.

## 1. Two questions, not one

They need different experiments and they have different audiences.

**Q1 — tabula rasa.** *Can a learned policy, starting from nothing, beat a tuned
PID?* The academic framing, and the one that tests the environments as an RL
benchmark.

**Q2 — expert-based.** *Given a tuned PID, can a learned policy improve on it?*
The industrial framing. Nobody replaces a working loop with a random network;
they ask whether learning buys anything on top of what they have. A negative
answer to Q1 and a positive answer to Q2 is a coherent and interesting result,
and it is the outcome plant engineers would care about most.

Both are run. They are reported separately and never averaged together.

## 2. What every comparison holds fixed

A number is only comparable to the PID and MPC beside it if it was produced
under the same conditions:

- the same reward, unmodified — no auxiliary shaping, no reward scaling;
- the same episode length, the environment's own `test_params`;
- the same evaluation episodes, so differences can be paired per seed;
- a deterministic policy at evaluation (the distribution's mean action), because
  the PID and MPC are deterministic and a stochastic policy would be scored on a
  different object than it is compared against.

Any deviation makes the row incomparable, and `tests/rl/test_rl_results.py`
rejects the two that can be checked mechanically.

## 3. The information asymmetry, stated up front

These three controllers do not see the same thing:

| controller | sees | has a model |
| --- | --- | --- |
| PID | observations | no |
| **learned policy** | **observations** | no |
| MPC | **true environment state** | **yes, exact** |

The MPC calls `_extract_x0(state)` and reads state fields directly, and it plans
against the real dynamics. Several environments are deliberately partially
observed — the glass furnace hides 6 of 9 dynamic states, the reactor 7 of 11,
the kiln 64 behind 8 measurements — so on those the MPC is solving a materially
easier problem than the one the agent faces.

Therefore: **the PID is the learned policy's peer, and the MPC is an upper bound
with more information.** Beating the PID is the result. Approaching the MPC is
interesting. Beating the MPC would be a claim requiring explanation, not a
victory lap, and the first hypothesis should be a simulator exploit rather than
superior control.

## 4. Metrics

**Within an environment**, the statistic is the *paired per-seed difference*
against the PID on identical evaluation episodes. Pairing removes the variance
from initial conditions and targets, which on these tasks is large: the circle
task's radius alone moved the PID's return by a factor of two.

**Across environments**, raw returns are not comparable — episodes run from 100
steps (cstr) to 1200 (reactor), and the reward is bounded in `[0, 1]` per step,
so return scales with horizon. Use **mean reward per step**, which is in `[0, 1]`
with 1 meaning perfect tracking at zero cost, and is the same quantity in every
environment.

**Report the interquartile mean with a 95% stratified bootstrap confidence
interval**, plus the win rate against the PID. Not the mean alone. This is
standard practice after Agarwal et al. (2021), and this repository has its own
evidence for it: the battery MPC scores `+14.0` on the mean and `−4.1` on the
median, winning 1 seed in 10. A mean alone would have published the opposite of
what happened.

## 5. Algorithms

**Headline: SAC and PPO.** One off-policy and one on-policy, so that a poor
result cannot be pinned on the quirks of a single family. Both are standard,
widely reimplemented, and what a sceptical reader will ask for. SAC is the
default choice for continuous control with dense rewards; PPO is the robust
on-policy reference and benefits most from the massively parallel environments
JAX makes cheap.

**Secondary study: an average-reward agent (ASAC or APO).** This is not
exoticism for its own sake. These tasks are *continuing* — reach a setpoint and
hold it indefinitely — and are ended by a time limit, not by achieving anything.
Discounted RL imposes an effective horizon that corresponds to nothing in the
task, and average-reward formulations are the theoretically correct fit. If the
discounted agents underperform, this distinguishes "RL cannot do this" from "the
discounted formulation was the wrong tool", which is a distinction the headline
result needs.

Not used for headline numbers: REDQ, AVG, UDRL, TD3. They are fine algorithms
and each invites "why that one?", which is a question a benchmark should not have
to answer.

## 6. Discount factor, chosen from physics

A single `gamma` across this suite would be indefensible: `delta_t` ranges from
0.05 s (first-order lag) to 900 s (building thermal mass), so `gamma = 0.99`
means a 5-second horizon in one environment and a 25-hour one in another.

Set the discount from *time*, not from steps:

    gamma = exp(-delta_t / tau)

with `tau` a fixed multiple of the plant's dominant time constant — the same
quantity the PID tuning already reasons about, and documented per environment in
its `PHYSICS.md`. This makes the agent's effective horizon a physical statement
("about five settling times") that is the same claim in every environment,
rather than an arbitrary constant that means something different in each.

The multiple is fixed across the suite and published; it is not tuned per
environment, because that would be tuning the objective rather than the agent.

## 7. Hyperparameters, and why they must be tuned

**The PID and the MPC are tuned per environment. If the learned policy is not,
the comparison is rigged in classical control's favour and the result is
worthless.**

This project has hard evidence for how much tuning matters. Re-tuning the
aircraft PIDs was worth between +26% and +139% on held-out seeds. The glass
furnace's gains sat pinned at the edge of their search grid and widening it was
worth 3.5%. A comparison against an untuned opponent measures the tuning, not
the method.

So: a **per-environment, per-algorithm hyperparameter search**, from a single
published search space, with a declared and equal budget (TPE, 64 trials, 3
seeds per trial). The space is fixed across environments so that "tuned" means
the same mechanical procedure everywhere and not the experimenter's taste.

**Tune on seeds disjoint from the ones reported.** The winner of a search is the
maximum of many noisy draws and is biased upward; reporting it on the same seeds
publishes that bias. Search on agent seeds 0–2, report on 10–39. This is the
discipline the aircraft PID tuner already follows — searched on seeds 0–2, quoted
on held-out 3–9 — and the learned side is held to the same rule.

**Observation normalisation is on, always, and is not a tuned choice.**
Observations span 9e-3 to 8.4e3 across the suite and about four orders of
magnitude *within* single environments — an aircraft reports altitude in
thousands of metres beside angles in radians. Without normalisation the
experiment would partly measure a network's tolerance of unscaled inputs. Every
run uses a running normaliser, and the fact is stated rather than swept into a
hyperparameter table.

## 8. Sample budget, reported as a curve

"Can RL beat a PID" has no answer without "at what cost". A policy that wins
after 50 million environment steps has not answered the industrial question,
because nobody runs 50 million steps on a real furnace.

Train to a fixed cap and report the **learning curve**, with results tabulated at
three budgets: **1e5, 1e6 and 1e7** environment steps. The cheapest column is the
one an engineer reads; the most expensive is the one that says whether the
method can do it at all. A single final number would hide the most
decision-relevant fact in the experiment.

Wall-clock and total environment steps are recorded alongside every result.

## 9. Seeds

Two distinct axes, routinely conflated:

- the **agent seed** — network initialisation, exploration, batch order;
- the **episode seed** — initial condition and target, which is what the shipped
  baselines vary.

Variation is reported over **agent seeds**, at least 30 of them. Each trained
agent is then evaluated on the **same fixed set of episodes** the PID and MPC
were measured on, which is what makes the comparison paired.

Thirty is affordable and ten would not have been: Ajax's own seed-scaling
measurements on `Plane3DCircle` show 15.8 s for one seed against 18.4 s for a
hundred, because fixed compilation overhead amortises. This project has been
misled by two-seed measurements three separate times — the wind turbine at "98%
of the PID", the aircraft at "no crashes", and an original verdict of MPC ahead
on 14 of 16 — each overturned by widening the seed count. Cheap seeds remove the
excuse.

## 10. The expert-based arm

Three structures, answering progressively weaker versions of Q2. The first is
the headline.

**Residual policy (primary).**

    a = clip(PID(obs) + alpha * pi_theta(obs), -1, 1)

The agent learns a bounded correction to the shipped controller. This is the
right primary for four reasons: it answers Q2 literally; it is the form industry
would actually deploy, since deviation from a known-good controller is bounded by
construction; it degrades gracefully, because `pi_theta -> 0` recovers the PID
exactly; and its failure mode is visible rather than silent.

`alpha` is swept over `{0.1, 0.25, 1.0}` and always reported. The sweep is not a
detail — at `alpha = 1` the residual can overwrite the expert entirely and the
method degenerates toward tabula rasa with an unusual prior. Showing the sweep is
what separates "learning improved the PID" from "learning ignored the PID", and
without it the claim is unfalsifiable.

**Expert-guided exploration (secondary).** Expert actions mixed into exploration
on an annealed schedule; Ajax implements this directly. Answers a weaker
question — whether the expert helps the agent *find* a good policy — while
leaving the final policy unconstrained.

**Behavioural cloning then fine-tuning (tertiary).** Pretrain on expert
trajectories, then train normally. The weakest form, because the expert's
influence decays with no guarantee and the result after enough steps is
indistinguishable from tabula rasa.

## 11. Controls

Every environment's table carries four reference rows, so a result is bracketed
rather than floating:

| row | what it establishes |
| --- | --- |
| random policy | the floor |
| best constant action | the bar a controller must clear to be doing anything — already asserted for the PID in the conformance suite |
| **PID** | **the peer** |
| MPC | the model-based, full-state upper bound |

A learned policy that fails to beat the best constant action has not learned
control, whatever its return looks like.

## 12. How to read the result

**If RL loses**, the result is on trial rather than the environments, and it has
to survive "your agent was under-trained". Cross-check three or four
environments against stable-baselines3, already a dev dependency with a working
PPO smoke test. If the two agree within noise, the JAX numbers inherit that
credibility; if they do not, that is a finding worth having before publishing.

**If RL wins**, check what it beat. This library has shipped a PID pinned to the
edge of its search grid, and an expert that could not fly a third of its own
task's radius range until it was allowed to trade speed for turn radius — worth
3.5% and 31% respectively once fixed. A win against a defective baseline
measures the defect.

**If RL wins on a partially observed environment**, suspect a simulator exploit
before celebrating. The figure-8's reward once paid for tracking precision its
own error metric could not resolve — an aircraft flying the commanded curve
exactly was scored as 66 m off it. A learned policy is far better than a PID at
finding that kind of seam, and check 11 of the model review checklist exists
because of it.

## 13. What is recorded

Every run is written through `target_gym.rl_results.record_result`, which stamps
the environment fingerprint so a result cannot outlive the environment it
describes. See [docs/rl-baselines.md](rl-baselines.md).

Tags distinguish the arms: `{algorithm}/{tabula|residual-a0.25|guided}/{budget}`.
Learning curves are stored alongside the final returns, since section 8 makes
them part of the result rather than a diagnostic.
