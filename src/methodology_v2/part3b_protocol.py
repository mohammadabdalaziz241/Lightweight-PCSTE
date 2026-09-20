"""Frozen Part-3B window-extraction protocol constants — methodology_v2.

FROZEN by explicit approval (2026-08-11), superseding the Part-3A
25 kHz common-rate candidate with NATIVE-RATE processing:

  - channels: one fixed accelerometer per dataset, never label- or
    condition-dependent;
  - sampling: native rates kept, NO resampling anywhere;
  - window: exactly 1.000 s (samples/window = native rate x 1.0 s,
    derived from the verified per-recording rate at build time);
  - stride: train 0.5 s (50% overlap, documented as training
    augmentation), validation/test 1.0 s (0% overlap);
  - JNU guards: G = 1.000 s instantiated around every frozen Part-2
    anchor b as [b - ceil(G/2), b + ceil(G/2));
  - HIT: ordered concatenation of source fragments ONLY within one
    audited session x speed-group x ch3 x label stream; every window
    records fragment provenance and boundary crossings.
"""
from __future__ import annotations

METHODOLOGY_VERSION = "methodology_v2.part3b.v1"

WINDOW_S = 1.0
STRIDE_S = {"train": 0.5, "validation": 1.0, "test": 1.0}

# frozen primary channel per dataset (see Part-3A channel census)
CHANNELS = {
    "CWRU": "DE",                      # drive-end accelerometer
    "JNU": "acc_vertical",             # the single channel
    "HIT": "ch3",                      # first casing accelerometer (idx 2)
    "MAFAULDA": "col3_underhang_radial",  # csv column index 2
}

# expected native rates (asserted against the frozen Part-2 manifests;
# never used to resample)
EXPECTED_NATIVE_RATE = {"CWRU": 48_000, "JNU": 50_000, "HIT": 25_000,
                        "MAFAULDA": 50_000}

# JNU guard instantiation (Part-2 symbolic rule, frozen width)
JNU_GUARD_S = 1.0

# HIT source-fragment geometry (audited: every stream = 18 contiguous
# fragments of 20,480 samples in preserved acquisition order)
HIT_FRAGMENT_SAMPLES = 20_480
HIT_FRAGMENTS_PER_STREAM = 18
HIT_CH3_ROW_INDEX = 2   # row 2 of the (series, 8, 20480) array = ch3

HIT_CONCAT_POLICY = (
    "ordered concatenation of source fragments permitted ONLY within one "
    "audited session x speed-group x ch3 x label stream, in preserved "
    "acquisition order; no smoothing/interpolation/cross-fade; never "
    "across session, speed group, label, or channel")
