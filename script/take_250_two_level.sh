#!/bin/bash

# Copied from /home/mce/brianna/take_250_2level.sh

row=0
col=12
col_rc=$(($col%8))
gain=-30

mas_param set sq1_ramp_tes_bias 0


cd ~/rshi/mce_scripts/python

python go_250kHz_two_level.py ${col} ${row} `date +%`

mce_cmd -x wb rc2 gaini${col_rc} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain} ${gain}

mce_run `date +%s`_nr2_dr256_dm4_250kHz_r${row}c${col} 25000 2

for rc in 2; do for c in `seq 0 7`; do mce_cmd -x wb rc${rc} gaini${c} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; done; done
for rc in 2; do mce_cmd -x wb rc${rc} data_mode 0; done
for rc in 2; do mce_cmd -x wb rc${rc} fb_const 440; done
mce_cmd -x wb sq1 fb_const 440
mce_run `date +%s`_nr2_dr256_dm0_250kHz_r${row}c${col}_fboff 25000 2

# mce_cmd -x wb rca flx_lp_init 1
