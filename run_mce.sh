#!/bin/bash                                                                     
# [$1==ndatasets]
# [$2==0, 1,2,3 ; columns to take data on - optional, otherwise set default_columns internally]
# [$3==1, 2,3 ; RCs to take data on - optional, otherwise set default_rcs internally]
# 20260116 copied from b3tower3:/home/bicep3/shawn/mce_scripts/go_raw_all.sh

CTIME_FOR_LOGFILE=`date +%s`
LOGFILE=/data/cryo/current_data/raw_${CTIME_FOR_LOGFILE}.txt
echo "OUTFILE=${LOGFILE}"

# defaults
#default_columns=(1 2 3 4 5 6 7)
default_columns=(0 5 6)
#default_columns=(0 1 4 5)
#default_columns=(2 3 6)
#default_columns=(0 1)
default_rcs=(1)
#default_columns=(0)
#default_rcs=(2)

# did user specify columns? comma delimited...
if [ ! -z $2 ] 
then 
    IFS=', ' read -r -a columns <<< "$2"
else
    columns=("${default_columns[@]}")
fi

# did user specify rcs? comma delimited...
if [ ! -z $3 ] 
then 
    IFS=', ' read -r -a rcs <<< "$3"
else
    rcs=("${default_rcs[@]}")
fi

echo "columns=(${columns[@]})"
echo "rcs=(${rcs[@]})"

#65536 samples is the default
nsamples=65536
ndatasets=$1

echo -e "tune\trc_fpga_temp\trc_card_temp\trc_card_id\trc_card_type\trc_slot_id\trc_fw_rev\trc\tcol\tdatedir\tdata">>${LOGFILE}
for rc in ${rcs[@]}; 
do
    CARD_ID=`mce_status -s | grep rc${rc} | grep card_id | awk '{print $4}'`
    CARD_TYPE=`mce_status -s | grep rc${rc} | grep card_type | awk '{print $4}'`
    SLOT_ID=`mce_status -s | grep rc${rc} | grep slot_id | awk '{print $4}'`
    FW_REV=`mce_status -s | grep rc${rc} | grep fw_rev | awk '{print $4}'`
    
    for idx in `seq 1 ${ndatasets}`;
    do
	for ((col=0;col<${#columns[@]};col+=1)); do
	    suffix="${idx}"
	    ./go_raw.sh ${rc} ${columns[$col]} ${suffix}

            # RC info to log
	    FPGA_TEMP=`mce_status -s | grep rc${rc} | grep fpga_temp | awk '{print $4}'`
	    CARD_TEMP=`mce_status -s | grep rc${rc} | grep card_temp | awk '{print $4}'`
	    RCINFO="$FPGA_TEMP\t$CARD_TEMP\t$CARD_ID\t$CARD_TYPE\t$SLOT_ID\t$FW_REV"
	    
	    # log
	    TUNE=$(basename $(readlink -f /data/cryo/last_squid_tune))	
	    TUNE=${TUNE%.sqtune}
	    LASTRUNFILE=`ls -rtl /data/cryo/current_data/ | grep .run | awk '{print $(NF)}' | tail -n 1`
	    DATEDIR=$(basename `readlink -f /data/cryo/current_data`)
	    x=${TUNE}"\t${RCINFO}\t${rc}\t${columns[$col]}\t${DATEDIR}\t${LASTRUNFILE%.run}"
	    echo -e ${x} >> ${LOGFILE}
	done
    done
done