# Pilot study protocol

## Questions
- Can Jev choose safe actions from text UI candidate sets?
- Is its top-1 probability empirically predictive of correctness?
- Can it sustain 2–5 steps with fewer LLM calls *without* reducing task success?
- Does training improve hybrid-policy success/compute at fixed GPU-hour + API budget?

## Experiments
1. Candidate coverage and action correctness on recorded AndroidWorld UI states.
2. Reliability / risk-coverage plots on validation (pmax, entropy, abstention).
3. Full LLM, Jev-only, random-routing, fixed-confidence, trainable-delegation policy.
4. Track template-disjoint test in addition to unseen seeds; no test-time threshold tuning.
5. Reward ablations: success only vs success + compute penalty.
6. Report train time, H100 GPU-hours, task attempts, successful tasks, cost per success, Jev requests, error recovery, task steps, latency.

## Cost accounting
The YAML `cost` constants are placeholder normalized units for mock tests. Replace with actual token/GPU time/API usage before any scientific claim.

## Safety / leakage
Only process test tasks for final evaluation. Do not build prompts from ground-truth task metadata unavailable to agents. Separate oracle evaluation from observable UI state.