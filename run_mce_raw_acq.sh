#!/bin/bash

if [ "$#" -lt "1" ]; then
    echo "Usage:   run_mce_raw_acq_all <ndatasets> [ <columns> <rcs> <nsamples> ]"
    echo
    echo "  ndatasets      number of datasets to acquire"
    echo "  columns        comma delimited list of columns to acquire data on (default is 0-7)"
    echo "  rcs            comma delimited list of RCs to acquire data on (default is 1,2)"
    echo "  nsamples       number of 50 MHz samples to take (default is 65536)"
    exit 1
fi


# [$1==ndatasets]
# [$2==0, 1,2,3 ; columns to take data on - optional, otherwise set default_columns internally]
# [$3==1, 2,3 ; RCs to take data on - optional, otherwise set default_rcs internally]
# 20260116 copied from b3tower3:/home/bicep3/shawn/mce_scripts/go_raw_all.sh

CTIME_FOR_LOGFILE=`date +%s`
LOGFILE=$MAS_DATA/raw_${CTIME_FOR_LOGFILE}.txt
echo "OUTFILE=${LOGFILE}"

ndatasets=$1

# defaults
default_columns=(0 1 2 3 4 5 6 7)
default_rcs=(1 2)

# did user specify columns? comma delimited...
if [ -n "$2" ] 
then 
    IFS=', ' read -r -a columns <<< "$2"
else
    columns=("${default_columns[@]}")
fi

# did user specify rcs? comma delimited...
if [ -n "$3" ] 
then 
    IFS=', ' read -r -a rcs <<< "$3"
else
    rcs=("${default_rcs[@]}")
fi

echo "columns=(${columns[@]})"
echo "rcs=(${rcs[@]})"

nsamples=${4:-65536}

# log header
echo -e "tune\trc_fpga_temp\trc_card_temp\trc_card_id\trc_card_type\trc_slot_id\trc_fw_rev\trc\tcol\tdatedir\tdata">>${LOGFILE}
for rc in "${rcs[@]}";
do
    MCE_OUTPUT=$(mce_status -s)
    CARD_ID=`echo "$MCE_OUTPUT" | grep rc${rc} | grep card_id | awk '{print $4}'`
    CARD_TYPE=`echo "$MCE_OUTPUT" | grep rc${rc} | grep card_type | awk '{print $4}'`
    SLOT_ID=`echo "$MCE_OUTPUT" | grep rc${rc} | grep slot_id | awk '{print $4}'`
    FW_REV=`echo "$MCE_OUTPUT" | grep rc${rc} | grep fw_rev | awk '{print $4}'`
    
    for idx in `seq 1 ${ndatasets}`;
    do
        for ((col=0;col<${#columns[@]};col+=1)); do
            suffix="${idx}"
            "$(dirname "$0")/run_mce_raw_acq_1col.sh" ${rc} ${columns[$col]} ${suffix} ${nsamples}

            # RC info to log
            FPGA_TEMP=$(mce_status -s | grep "rc${rc}" | grep fpga_temp | awk '{print $4}')
            CARD_TEMP=$(mce_status -s | grep "rc${rc}" | grep card_temp | awk '{print $4}')

            RCINFO="$FPGA_TEMP\t$CARD_TEMP\t$CARD_ID\t$CARD_TYPE\t$SLOT_ID\t$FW_REV"
            
            # log
            TUNE=$(basename "$(readlink -f "$MAS_DATA_ROOT/last_squid_tune")")	
            TUNE=${TUNE%.sqtune}
            LASTRUNFILE=$(find "$MAS_DATA" -maxdepth 1 -name "*.run" -printf '%T@ %f\n' | sort -n | tail -1 | cut -d' ' -f2)
            DATEDIR=$(basename "$(readlink -f "$MAS_DATA")")
            x=${TUNE}"\t${RCINFO}\t${rc}\t${columns[$col]}\t${DATEDIR}\t${LASTRUNFILE%.run}"
            echo -e ${x} >> ${LOGFILE}
        done
    done
done