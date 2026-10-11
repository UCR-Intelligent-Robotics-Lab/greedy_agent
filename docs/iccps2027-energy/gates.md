# Pre-registered gates: ICCPS 2027 energy panel

Committed before any wave-1 result file was opened (Yifan, 2026-10-10). The
analysis code implements these rules exactly; any deviation is reported as a
deviation, never applied silently.

## Statistics used by every gate

- Unit of replication: training seed (10 seeds, 12341-12350). The 100 final
  evaluation episodes of a run are averaged within that run and never treated
  as independent replicates (Patterson et al. 2024; Agarwal et al. 2021).
- Comparisons are paired by seed: d_s = metric(method, s) - metric(reference, s).
  Relative cost differences use d_s / metric(reference, s).
- Test: exact one-sided Wilcoxon signed-rank test on d_s (Wilcoxon 1945), all
  2^n sign assignments enumerated; zero differences dropped and counted; ties
  use average ranks. Alpha = 0.05.
- Primary tests: G1, G3a, G3b. Familywise error across these three is controlled
  with the Holm procedure (Holm 1979). G2 and G4 are secondary or descriptive.
- Reported with every comparison: median paired difference with a 95%
  percentile bootstrap interval over seeds (10,000 resamples, generator seed
  20261010); probability of improvement, the fraction of seeds whose d_s lies
  in the predicted direction (Agarwal et al. 2021); and that seed count. A seed
  count alone is descriptive only: 7/10 has one-sided sign-test p = 0.17 and
  8/10 has p = 0.055.
- Effect-size thresholds below are smallest effects of interest fixed before
  the data (Lakens 2017). A gate passes only if both the test and the effect
  size condition hold.
- With 10 seeds only consistent effects are detectable; a failed test is
  reported as "not detected", not as "no effect" (Colas et al. 2018).

## G1 (primary, wave 1): does the energy term reduce mission cost?

REFiNE uniform_v1 versus LIO uniform_v1, homogeneous hardware, scored under
the uniform_v1 table. Pass iff all hold:
(a) Holm-adjusted one-sided Wilcoxon p < 0.05 for lower mean agent mission cost
    under REFiNE;
(b) median relative paired difference <= -5% (half of the -11% energy-only
    versus LIO difference in the SenSys data, chosen before these runs);
(c) success non-inferiority: the lower bound of the 95% bootstrap interval of
    the median paired success-rate difference is >= -0.05.
Descriptive: the same comparison for REFiNE low_contrast_v1 and paper_ratio_v1,
each under its own training table.

## G2 (secondary, wave 1): beyond a step penalty under homogeneous hardware?

REFiNE paper_ratio_v1 versus REFiNE uniform_v1, both scored under the
paper_ratio_v1 table; predicted: lower support cost component. Report test,
median, interval and probability of improvement. A non-detection does not fail
the panel; it records that under homogeneous hardware the energy term behaves
like a step penalty.

## G3 (primary, wave 2): does action-dependent cost route support to cheaper robots?

Metric per seed: support share = lever actions by agents 0 and 1 (lambda 1.0)
divided by all lever actions, over the final-policy evaluation episodes, with
hetero_2x1p5_v1. Seeds with no lever action are excluded and reported; if more
than 2 of 10 seeds are excluded in an arm, G3 is INCONCLUSIVE.
G3a: REFiNE paper_ratio_v1 + hetero_2x1p5_v1: one-sample Wilcoxon on
     (share - 0.5), one-sided > 0, Holm-adjusted p < 0.05, and median share >= 0.65.
G3b: paired REFiNE paper_ratio_v1 + hetero versus REFiNE uniform_v1 + hetero:
     higher share, Holm-adjusted p < 0.05.
Sanity reference (not a gate): LIO uniform_v1 and REFiNE uniform_v1 from wave 1
(homogeneous training), with support share computed over the same agent groups;
median expected within [0.35, 0.65]. If outside, report an agent-index
asymmetry before interpreting G3. Descriptive: lambda-weighted worst-agent cost.

## G4 (descriptive, wave 3): energy in the tradable reward

LIO + energy_in_reward (paper_ratio_v1; homogeneous and hetero_2x1p5_v1) versus
(i) LIO env_reward (accounting-only matching rule, P5) and (ii) REFiNE
paper_ratio_v1 with the same hardware. Pre-declared metrics: mean agent mission
cost, worst-agent cost, success rate, and support share under hetero. Report
test, median, interval and probability of improvement; no pass line.

## Decision rule (October 18)

- G1 passes and G3a and G3b pass: the main line stands.
- G1 passes, G3 fails: narrow the claim to cost reduction; no role-routing claim.
- G1 fails: discuss re-scoping or venue with the advisor.
- G3 INCONCLUSIVE: report and discuss; no claim.

## References

- Henderson, Islam, Bachman, Pineau, Precup, Meger. Deep Reinforcement Learning
  that Matters. AAAI 2018.
- Colas, Sigaud, Oudeyer. How Many Random Seeds? Statistical Power Analysis in
  Deep Reinforcement Learning Experiments. arXiv:1806.08295, 2018.
- Agarwal, Schwarzer, Castro, Courville, Bellemare. Deep Reinforcement Learning
  at the Edge of the Statistical Precipice. NeurIPS 2021.
- Patterson, Neumann, White, White. Empirical Design in Reinforcement Learning.
  JMLR 25(318), 2024.
- Lakens. Equivalence Tests: A Practical Primer for t Tests, Correlations, and
  Meta-Analyses. Social Psychological and Personality Science, 2017.
- Holm. A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian
  Journal of Statistics, 1979.
- Wilcoxon. Individual Comparisons by Ranking Methods. Biometrics Bulletin, 1945.
- Nosek, Ebersole, DeHaven, Mellor. The Preregistration Revolution. PNAS 2018.
