# Artifact Policy

Normal Git history contains source, frozen text manifests, protocol metadata, tests, and small result summaries only.

| Artifact | Current handling | Planned archival route |
|---|---|---|
| Raw datasets and archives | Excluded | Original providers; local path configuration |
| Full-S1 and K1 checkpoints | SHA/reference only | Release assets or Zenodo after final selection |
| Teacher caches (~645 MB each) | SHA/reference only | Zenodo/external archival storage |
| Frozen normalizer NPZ files | Hash/reference only initially | Release asset or compact archival bundle |
| Final selected lightweight checkpoints | Pending | Decide after all nine cells and TEST freeze |
| Q8 artifact | Pending scientific decision | Release/Zenodo only if retained |

No binary artifact should be added without verifying its identity, scientific necessity, license, and storage destination.

