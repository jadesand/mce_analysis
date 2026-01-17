#!/bin/bash

#initialise environment
if [ ! -x ${MAS_VAR:=/usr/mce/bin/mas_var} ]; then
  echo "Cannot find mas_var.  Set MAS_VAR to the full path to the mas_var binary." >&2
  exit 1
else
  eval $(${MAS_VAR} -s)
fi

source $MAS_SCRIPT/mas_library.bash

if [ "$#" -lt "1" ]; then
    echo "Usage:   mce_raw_acq_1col_dev <rc> <col> [ <samples> <suffix> <dirname> <runfile_flag> ]"
    echo "  rc             is 1,2,3, or 4"
    echo "  col            is 0,...,7"
    echo "  samples        number of 50 MHz samples to take (default is 65536)"
    echo "  suffix         will be appended to the filename"
    echo "  dirname        folder name under \$MAS_DATA (default: raw_<ctime>, created if needed)"
    echo "  runfile_flag   suppresses runfile creation when equal to \"no_runfile\""
    echo
    echo "  When a default sample count is not specified, num_rows_reported is "
    echo "  automatically set to 32 so that the output will fit into 256 frames"
    echo "  exactly."
    exit 1
fi

rc=$1
rc_cmd=$rc
[ "$rc_cmd" == "s" ] && rc_cmd="a"

column=$2
n_samples=65536
change_num_rows=1

if [ "$3" != "" ]; then
    n_samples=$3
    change_num_rows=0
fi

if [ "$4" != "" ]; then
    suffix="_$4"
fi

if [ -n "$5" ]; then
    dirname="$5"
else
    dirname="raw_$(date +%s)"
fi
mkdir -p "$MAS_DATA/$dirname"

# Check firmware
fw_rev=`command_reply rb rc$rc_cmd fw_rev |cut -d' ' -f1`
case "$fw_rev" in
    0x400000d | 0x400000e | 0x5*)
	;;
    0x4010007 | 0x4020007 | 0x4030007)
	echo "RC firmware $fw_rev does not support single column raw mode!"
	exit 1
	;;
    *)
	echo "RC firmware $fw_rev not recognized!  Are you sure it supports raw mode?"
	;;
esac

ct=`print_ctime`
filename=$MAS_DATA/${dirname}/${ct}_raw${suffix}
runfilename=${filename}.run

if [ "$6" == "no_runfile" ]; then
    do_runfile=0
else
    do_runfile=1
fi

# Disable sync box
use_sync=`command_reply rb cc use_sync`
use_dv=`command_reply rb cc use_dv`
mce_cmd -qX "wb cc use_sync 0" -X "wb cc use_dv 0"

# Store column index and data mode...
orig_col_index=`command_reply rb rc$rc_cmd readout_col_index`
orig_data_mode=`command_reply rb rc$rc_cmd data_mode`

# Get/set frame structure
num_cols_rep=8
if [ "$change_num_rows" == "1" ]; then
    old_num_rows_rep=`command_reply rb cc num_rows_reported`
    num_rows_rep=32
    mce_cmd -qx wb cc num_rows_reported $num_rows_rep
else
    num_rows_rep=`command_reply rb cc num_rows_reported`
fi

# Round to nearest complete readout frame
n_frames=$(( ($n_samples + $num_cols_rep*$num_rows_rep - 1) / \
    ($num_rows_rep * $num_cols_rep) ))

[ "$n_frames" == 0 ] && echo "Error: frame count is $n_frames" && exit 1

# Select column and data mode, then delay
mce_cmd -q \
    -X "wb rc$rc_cmd readout_col_index $column" \
    -X "wb rc$rc_cmd data_mode 12" \
    -X "sleep 100000"

if [ "$do_runfile" == "1" ]; then
    mce_status >> $runfilename
    frameacq_stamp $rc $filename $n_frames >> $runfilename
fi

echo "Acquiring raw data to $filename ..."
mce_cmd -q \
    -X "acq_config $filename rc$rc" \
    -X "wb rc$rc_cmd captr_raw 1" \
    -X "sleep 10000" \
    -X "acq_go $n_frames"

# Restore sync box settings
mce_cmd -q -X "wb cc use_sync $use_sync" -X "wb cc use_dv $use_dv"
[ "$change_num_rows" == "1" ] && \
    mce_cmd -qx wb cc num_rows_reported $old_num_rows_rep

# Restore data_mode and col_index
mce_cmd -q \
    -X "wb rc$rc_cmd readout_col_index $orig_col_index" \
    -X "wb rc$rc_cmd data_mode $orig_data_mode"

