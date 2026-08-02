# E0 replay validation

Overall: **PASS**; replay level: **bitwise**.

| Config | Status | Replay | Mismatches | Max sham deviation | Checkpoint |
|---|---:|---:|---:|---:|---:|
| e0_crossmachine_reference | PASS | bitwise | 0 | 0 | API_GAP |

Checkpoint note: RNG snapshots restore equivalently, but suffix hashes cannot be checked until the public runner accepts a model checkpoint and start round.
