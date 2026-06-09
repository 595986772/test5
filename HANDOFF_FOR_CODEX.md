# Codex Handoff: FDEdge / MOFD V8 Project

This repository is a lightweight handoff snapshot for another Codex session.

## Start Here

Recommended first prompt for the next Codex:

```text
Please read HANDOFF_FOR_CODEX.md, todo/V8_项目结构问题清单.md, and the key files under FDEdge-main/. Then continue the FDEdge/MOFD V8 project. Keep the terminology consistent: preference vector omega, Pareto frontier, hypervolume, clean vector critic, feedback diffusion, multi-objective edge offloading.
```

## Current Project Story

The project studies preference-conditioned multi-objective edge offloading with two objectives:

- delay
- energy

The borrowed backbone is feedback diffusion scheduling from the delay-only FDEdge work. The current paper/project contribution should not claim feedback diffusion itself as new.

The most stable current main line is:

```text
V8 = clean vector critic for preference-conditioned delay-energy Pareto edge offloading
```

The clean V8 critic changes V5 in two important ways:

1. Critic target no longer mixes SAC entropy into `Q_delay` and `Q_energy`.
2. Critic loss no longer multiplies each objective channel by omega, so extreme preferences do not zero out the other objective's critic gradient.

Actor-side preference scalarization remains:

```text
Q_eff = omega_delay * Q_delay + omega_energy * Q_energy
```

So omega still controls action selection; only critic learning is made cleaner.

## Important Current Judgments

Use these as the current working assumptions unless new experiments overturn them:

- Main model: `MOFD_SAC_V8` in `FDEdge-main/mofd_v8.py`.
- V8 is a clean critic learning fix, not a full new architecture.
- OACR is currently inconclusive/failed: initial single-batch result looked positive, but multi-batch evaluation showed late fusion wins only 2/5 batches and mean diff is negative.
- H-MCSS full3 should not be a positive contribution: V8 ablation shows full3 is worse than single-source variants.
- Buffer/drift recovery should not be the main paper story unless new evidence clearly supports it.
- PGW is optional/future: it can be used only if it beats both greedy and uninformative diffusion under fair information access.

## Key Files

```text
FDEdge-main/mofd_v8.py
    V8 clean vector critic. Small and important.

FDEdge-main/mofd_v5.py
    V5 vector-Q, COR, PopArt-lite, SAC-style update, feedback diffusion actor.

FDEdge-main/mofd_main.py
    Main training/evaluation flow.

FDEdge-main/mofd_environment.py
    Delay-energy edge offloading environment.

FDEdge-main/fixed_testset.py
    Fixed evaluation testset builder/evaluator.

FDEdge-main/greedy_warmstart.py
    PGW/greedy warm-start diagnostic foundation.

修改方案/Preference_Greedy_Warmstart_Diffusion_完整修改方案.md
    PGW design plan and review context.

todo/V8_项目结构问题清单.md
    Current structural issues and prioritized TODO list.
```

## Important Result Files Included

Only lightweight key result summaries are intended to be committed. Full timestamped run folders and checkpoints are ignored.

Useful included results:

```text
FDEdge-main/results/oacr_multiseed_eval.txt
FDEdge-main/results/oacr_ab_compare.txt
FDEdge-main/results/ablation_hmcss_v8_summary.txt
FDEdge-main/results/ablation_hmcss_v8_summary.csv
FDEdge-main/results/pmem_ab_compare.txt
FDEdge-main/results/prior_content_probe.log
FDEdge-main/results/eval_consistency_probe.log
FDEdge-main/results/fixed_testset_build.log
FDEdge-main/results/eval_testset_dashboard.png
FDEdge-main/results/eval_testset_per_omega.png
FDEdge-main/results/eval_testset.pkl
```

## Current Writing Plan

Target venues discussed:

- MSN
- HPCC

Suggested paper title:

```text
Preference-Conditioned Feedback Diffusion for Pareto-Optimal Edge Offloading
```

Core paper structure:

1. Introduction: delay-energy edge offloading needs controllable preference-conditioned trade-offs.
2. Related Work: edge offloading, RL for MEC, MORL, diffusion scheduling.
3. System Model: task, server, channel, queue, delay, energy, preference vector omega.
4. Method: preference-conditioned feedback diffusion + clean vector critic.
5. Experiments: fixed testset, Pareto comparison, preference response, V8 ablation, optional PGW if proven.

Avoid writing OACR, H-MCSS full3, or buffer drift recovery as positive contributions unless new experiments reverse the evidence.

## Fixed Testset Note

`FDEdge-main/results/eval_testset.pkl` is a fixed evaluation set:

- 21 preference points.
- 40 scenarios per preference.
- Stored reference point for HV.

Purpose:

```text
Avoid unstable random evaluation where the same model's HV can swing heavily across small random evaluation batches.
```

Use it for future fair comparisons.

## Main Missing Work

Before writing the final paper, the project still needs:

1. Unified model version switch instead of mixed `use_v5/use_v8`.
2. Fixed-testset evaluation for V5/V8/baselines under one shared reference point.
3. Multi-seed training results.
4. Clean V8 ablation:
   - V8 vs V5.
   - entropy-in-critic-target ablation.
   - omega-weighted critic-loss ablation.
5. Preference response curves:
   - omega-delay response.
   - omega-energy response.
   - seen/unseen preference interpolation.
6. PGW only if fair A/B/C tests support it.

## Honesty Boundary

Do not invent experimental numbers. All numbers must come from result files. The project has explicitly chosen a strict honesty line:

```text
measured results and hypothesized mechanisms must be separated.
```

If a mechanism fails, mark it failed and remove it from the paper story.
