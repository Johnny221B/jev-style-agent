# verl GRPO integration contract (not yet implemented)

Official docs: https://verl.readthedocs.io/en/latest/advance/agent_loop.html

**IMPORTANT:** The included `OpenAICompatiblePolicy` is for inference testing only and cannot furnish policy log-prob/token trajectories for valid GRPO.

1. Pin `verl` commit and base model. Verify `AgentLoopBase`, `AgentLoopOutput`, and `LLMServerClient.generate` signatures on that exact commit.
2. Implement a subclass of `AgentLoopBase` with `async def run(self, sampling_params, **kwargs)`.
3. Obtain `prompt_ids` from the dataset prompt, call `self.server_manager`/`llm_client.generate` according to installed version with token IDs, and append **exact generated token IDs**, not retokenized text.
4. For LLM-generated actions or `CALL_JEV`, mark `response_mask=1`. Append Jev decisions, environment observations, and tool messages as context with `response_mask=0`. Ensure chat-role delimiters and causal context align with verl's chat template.
5. Compute terminal reward from environment success plus an explicit cost ledger (LLM prefill/decode tokens, Jev requests/tokens/latency, env actions). No external LLM judge needed.
6. Check policy updates affect only generated LLM tokens. Test a 2-step synthetic training episode, verify loss is finite and `response_mask` count matches the LLM output token count.
7. SFT is optional: train from existing task data and a tool call description first, then assess whether RL starts exploring Jev calls. Do not claim teacher-free superiority before measuring convergence.
8. Begin LoRA GRPO on 4B. Scale to 8B only after stable rollout and reward propagation.

Known limitation: Default rollout metrics in the pure Python runner are **not** token-accurate and not suitable as GRPO input. Avoid a shell command that advertises RL training before these conditions are met.