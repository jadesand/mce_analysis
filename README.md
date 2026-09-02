# mce_analysis

Analysis tools for MCE/SQUID tuning data: noise, PSD fitting, timestream
(tset) processing, IV curves, and tune visualization, plus an `ac_crosstalk`
subpackage for AC / two-level-muxing crosstalk analysis.

## Install

```bash
pip install -e .
```

`moby2` is a required dependency for I/O (`tools.py`, `iv.py`, `tset.py`,
`noise.py`) but is not on PyPI — install it separately as an editable
package before importing those modules, e.g.:

```bash
pip install -e /path/to/moby2
```

## Layout

- `tools.py` — shared I/O helpers (loading SSA/SQ1/row-select/chip-select
  tune data, config files, raw MCE files).
- `noise.py` — raw MCE readout, PSD computation, f3dB estimation,
  calibration.
- `psd_fitting.py` — noise model fitting (superfast readout PSDs: white,
  1/f, red noise).
- `tset.py` — timestream segmenting, filtering, multi-exponential response
  fitting, settle-time estimation.
- `iv.py` — IV curve analysis.
- `data_process.py` — misc signal-processing helpers (spike finding,
  baseline correction).
- `tune_visualization.py` — plots SSA V-phi curves from raw MCE sa_ramp
  data; also runnable as a script.
- `tune_cfg/` — per-array `.cfg` files (DAC scales, mutual inductances,
  cable resistances, etc.) used throughout the package.

### `ac_crosstalk/`

Analyzes SQ1 critical-current (Ic) and AC crosstalk from MCE SQUID tuning
sweeps, generalized for both 1-level (row-select only) and 2-level
(chip-select + row-select) muxing.

- `main.py` — SSA/SQ1 flux-modulation fitting, Ic min/max sweep, crosstalk
  bias-limit calculation, and the CLI entry point.
- `plot.py` — all associated plotting (modulation curves, Ic min/max
  overlays, per-chip/per-column summaries, row x column heatmaps).

Run as:

```bash
python -m mce_analysis.ac_crosstalk.main \
    --cs-on-rs-on  /path/to/ctime_dir \
    --cs-on-rs-off /path/to/ctime_dir \
    [--cs-off-rs-on /path/to/ctime_dir] \
    [--cs-off-rs-off /path/to/ctime_dir] \
    --dev_cur
```

Only `--cs-on-rs-on` is required; passing just it and `--cs-on-rs-off` runs
1-level analysis (row-select on/off). Adding either `--cs-off-*` flag
switches to 2-level analysis, comparing up to 4 chip-select x row-select
conditions. Add `--units ua` for micro-amp units (defaults to DAC), or
`--units both` for both. Use `-r`/`--rsservo` (with `-t`/`-u`) for the
separate row-select servo diagnostic.
