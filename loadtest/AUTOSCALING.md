# Autoscaling results

Written by hand, unlike [RESULTS.md](RESULTS.md), which `loadtest/report.py` regenerates from `results/*.json` and would overwrite. Method: [README](README.md#autoscaling).

## Autoscaling

`python loadtest/autoscale.py` queues a burst of uploads, then samples queue depth, what the worker HPA reads, and the replica count every 5 s until the backlog clears. Raw data: `results/autoscale*.json`.

| Run | Burst | Workers | Peak queue | Scaled out at | Backlog cleared |
|---|---|---:|---:|---:|---:|
| one worker (control) | 150 × 160 KB | 1, pinned | 65 | — | **312 s** |
| autoscaled | 150 × 160 KB | 1 → 4 | 67 | 63 s | **324 s** |
| one worker (control) | 300 × 24 KB | 1, pinned | 58 | — | 108 s |
| autoscaled | 300 × 24 KB | 1 → 3 | 58 | 77 s | 109 s |

**The control loop works.** Queue depth reached 51 jobs per worker against a target of 5, the HPA added pods, and the queue drained to zero. Scale-out takes about 63 s: a deliberate 60 s stabilisation window, plus pod start. Nothing scales on an empty queue any more.

**On one node it does not pay for itself, and the honest number says so:** 324 s with four workers against 312 s with one. A single ingest job already uses about 8 cores, so the extra pods competed for CPUs that were busy, and each one paid a model-loading cost on the way in. With 24 KB documents the two runs were also a dead heat (108 s vs 109 s), for a different reason: uploads arrived no faster than one worker cleared them, so there was never a real backlog.

That is a statement about a 12-vCPU laptop, not about the design. A worker's CPU request was raised from 250m to 1000m to stop the scheduler packing workers onto a node that one of them saturates, the kind overlay caps the worker HPA at 2 replicas, and the base keeps 4 for a cluster with somewhere to put them.

**The same test found two bugs** (details in [README](README.md#autoscaling)): a document large enough to OOM-kill the worker embedding it, and the poison pill it then became — redelivered after each kill, one job crash-looped four workers for 70 minutes. Both are fixed and covered by tests; the burst that caused them now completes with zero restarts.
