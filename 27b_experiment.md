# qwen3.6-27b concurrency experiment — 2026-05-05 evening

User went to bed; left me to figure out the optimal concurrency for qwen3.6-27b.
35b is running at concurrency 8 in a separate process (do not touch unless it breaks).

## Known data points
- **c=1, max_tokens=8192**: single curl probe → 200 OK, 116s
- **c=2, max_tokens=8192**: 37 trials, 0 errors, avg 151s, ~22% correct (later deleted)
- **c=8, max_tokens=12288**: full backend failure ("All 3 backend attempts failed", 502 after 541s)
- **c=8, max_tokens=8192 + thinking**: not directly tested; failure mode at c=8 was at 12288

## Hypotheses
- Sweet spot likely c=3–5 with max_tokens=8192 + thinking on
- Failure mode is backend-attempt exhaustion, not rate limits
- 27b has 2 backends per user; concurrency > 2 saturates them but may still complete

## Strategy
1. Start c=4. If clean for 2 iterations → try c=5, then c=6.
2. If errors/latency-spike at any level → drop one and stick.
3. Goal: find highest concurrency that maintains 0% error rate and avg latency < 250s.
4. Once tuned, let it run the rest of the night at the chosen concurrency.

## Iteration log
(Append findings here each /loop iteration.)

### Iter 0 (immediate, ~23:28 local)
- 27b just launched at concurrency 4; 4 in-flight, 0 completions yet (59s elapsed)
- 35b: 181 done, 0 errors, avg 80.7s — concurrency 8 holding clean
- Decision: HOLD. Need at least one full call cycle (~150s) before any judgment. Next iteration in 15 min should have ~5-6 completions.

### Iter 1 (~23:32 local, 3.6m elapsed)
- 27b at c=4: 4 done, 1 correct, 0 errors, avg 136.7s, 4 in-flight
- Throughput: 1.1 trials/min (vs c=2 baseline of ~0.8/min projected)
- Latency 136.7s slightly *better* than c=2 baseline (151s) — encouraging but small sample
- 35b at c=8: 191 done, 0 errors, avg 81.2s — still clean
- Decision: HOLD at c=4. Only 4 completions; need a larger sample before bumping. Next iter expect ~12-15 completions.

### Iter 2 (~23:47 local, 18m elapsed)
- 27b at c=4: 19 done, 5 correct (26%), 0 errors, avg 145.2s, 4 in-flight
- Throughput: 1.03 trials/min — way too slow for overnight (would do ~500 of 2373 by morning)
- Latency essentially identical to c=2 baseline (145s vs 151s) — backend has headroom
- 35b at c=8: 241 done, 0 errors, avg 83.2s — still clean
- Decision: BUMP to c=6 (skipping c=5 to make better use of overnight time). If iter 3 shows latency creep >200s or any errors, drop back to c=5 or c=4.

### Iter 3 (~00:02 local, 14m at c=6)
- 27b at c=6: 37 done at c=6 phase, 25 ERRORS, latency 146.1s (clean range 141-156s)
- **Errors are network-side ([Errno 8] DNS / [Errno 65] No route to host / [Errno 60] timeout) — wifi blip, NOT backend overload**
- 35b also picked up 2 of these = same blip
- Backend at c=6 is healthy: latency tight, no 5xx, no rate-limit issues
- ACTION: hardened models.py to retry on httpx.ConnectError/ReadError/WriteError/TimeoutError (net-blip resilience). Cleared 27b error rows so they retry. Relaunched 27b at c=6 with hardened code. Did NOT restart 35b (only 0.6% net-blip rate, not worth disruption — old module stays in memory).
- Going forward: HOLD at c=6, monitor whether net-error retry recovers cleanly. If next iter shows clean re-completions of the previously-errored trials, consider bumping to c=7 or c=8.

### Iter 4 (~00:17 local, 12m at c=6 hardened)
- 27b at c=6 hardened: 17 new completions, 0 errors, avg 145.4s (range 141.7-151.8s — extremely tight)
- Throughput 1.4 trials/min (+35% over c=4 phase, +0% over c=6 pre-harden where errors lost time)
- Latency curve flat: c=2=151s, c=4=145s, c=6=145s. Backend not loaded.
- 35b at c=8: 365 done, 2 net errors (no new), avg 85.8s — still clean
- ACTION: BUMP c=6 → c=8 (match 35b). Earlier c=8 failure was at max_tokens=12288; now at 8192 (50% memory pressure cut). If c=8 fails, fall back to c=6 (known good).
- 55 trials retained at relaunch.

### Iter 5 (~00:32 local, 14m at c=8)
- 27b at c=8: 30 new completions, 0 errors, avg 142.2s
- Throughput **2.14 trials/min** (+53% over c=6, +108% over c=4)
- Latency curve still essentially flat: c=2=151s, c=4=145s, c=6=145s, c=8=142s. Backend has untapped headroom.
- One anomaly: 3 trials on task=31 completed in 33-38s (likely simple tasks where model stopped early). Bulk are 140-145s.
- 35b at c=8: 425 done, 3 errors total (1 new net error this iter), avg 86.5s — stable
- ACTION: BUMP c=8 → c=10. Cautious next step, given backend stability across all priors. If iter 6 shows latency >180s or errors, drop to c=8.
- 87 trials retained at relaunch.

### Iter 6 (~00:47 local, 14m at c=10)
- 27b at c=10: 35 new completions, 0 errors, recent latency 166.9s (range 140-213s)
- Throughput **2.5 trials/min** (+17% over c=8)
- **Latency curve breaks**: flat 151/145/145/142 at c=2-8, jumps to 167s at c=10. Max 213s seen (no error). Backend showing first stress.
- 35b at c=8: 457 done, 3 errors total (no new), avg 86.9s — stable
- ACTION: HOLD c=10. Average (167s) below 180s drop threshold. Need iter 7 to confirm whether creep continues or stabilizes. If iter 7 avg >180s or any errors, drop to c=8.

### Iter 7 (~01:02 local, 28m at c=10)
- 27b at c=10: 159 done, 0 errors, recent 12 trials avg **145.5s** (back to flat baseline)
- Iter-6's 167s was warmup post-relaunch (8 workers spinning up at once); settled to ~145s
- Throughput **2.64 trials/min** (best yet)
- 35b: 504 done, **+7 new "read operation timed out" errors** (now 10 total = 2%). 35b still on old module — declined to restart for hardened retry; cost (12min of in-flight work) > marginal gain.
- ACTION: BUMP c=10 → c=12. Latency normalized at c=10, strategy of "2 clean iterations → bump" satisfied. If c=12 shows latency >180s avg or errors, drop back to c=10 (now known good).
- 159 trials retained at relaunch.

### Iter 8 (~01:17 local, 13m at c=12)
- 27b at c=12: 37 new completions, 0 errors, recent latency **149s** (range 144-159s — tight)
- Throughput **2.85 trials/min** (+8% over c=10)
- 35b: 567 done, 10 errors total (no new) — last iter's net blip didn't recur
- Latency curve: c=2/4/6/8/10/12 = 151/145/145/142/145/149s. Backend handling c=12 cleanly.

## CONCLUSION: SETTLE at c=12

Diminishing returns curve:
- c=4→c=8: +114% throughput
- c=8→c=10: +23%
- c=10→c=12: +8%
- c=14: estimated +5% with non-zero tip-over risk

c=12 is the optimal stable point. Further iterations switch to monitor-and-report; do NOT bump above c=12 unless explicitly directed. If anything degrades, drop to c=10 (also known good).

Final config for the night:
- max_tokens: 8192, enable_thinking: true, concurrency: 12, hardened net-error retry, hardened 5xx retry.

## FINAL RESULTS (~07:33 local, both runs complete)

| Model | Concurrency | Trials done | Correct | Accuracy | Errors |
|---|---|---|---|---|---|
| qwen3.6-27b | 12 | 2373 | 644 | 27.1% | **0** |
| qwen3.6-35b | 8 | 2355 | 649 | 27.6% | 18 (need retry) |

- **27b at c=12 ran to 100% completion with zero errors** — perfect run after the hardened retry was added (iter 3).
- 35b accumulated 18 socket-timeout / DNS errors over its 6h run (~0.76% rate). 35b was never restarted with hardened retry; declined to disrupt mid-run.
- Both models score within 0.5% on accuracy — interesting parity given they're different sizes.
- Cron loop `0671748a` cancelled after stop condition met.

### Cleanup pass needed
The 18 errored 35b trials should be retried. Procedure:
```sql
DELETE FROM baseline_trials WHERE model_name='qwen3.6-35b' AND error IS NOT NULL;
```
Then `python collect.py run --model qwen3.6-35b --concurrency 8` will pick up only those 18 trials (the rest are skipped). With hardened retry now in models.py, this pass should complete cleanly in ~5 min.
