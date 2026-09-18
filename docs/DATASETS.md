# Datasets

The final study uses CWRU, JNU, HIT, and MaFaulDa. Raw datasets are not distributed in this repository.

CWRU uses only official native 12 kHz recordings, without 48 kHz inputs or resampling. Its final load partition is TRAIN = 0 hp + 1 hp, VALIDATION = 2 hp, and TEST = 3 hp (`cwru_native12_load_v1`). The global protocol is `global_v2_final_s0s1_v1`.

Frozen manifests describe identities and partitions. Users must obtain each dataset under its original terms and configure `PCSTE_DATA_ROOT` locally.

