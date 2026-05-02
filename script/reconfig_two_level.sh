source $MAS_SCRIPT/mas_library.bash
SCRIPT_NAME=$(basename "$0")

tune_ctime=""
cs="0"
date="current_data"

# Parse keyword arguments
opts=$(getopt -o t:c:d: \
    --long tune-ctime:,cs:,date: \
    -n "$SCRIPT_NAME" -- "$@")

if [ $? -ne 0 ]; then echo "Error parsing options"; exit 1; fi
eval set -- "$opts"

while true; do
    case "$1" in
        -t|--tune-ctime)    tune_ctime="$2"; shift 2 ;;
        -c|--cs)            cs="$2"; shift 2 ;;
        -d|--date)          date="$2"; shift 2 ;;
        --)                 shift; break ;;
        *)                  echo "Unknown option: $1"; exit 1 ;;
    esac
done

mce_zero_bias > /dev/null
sleep 1
mce_make_config -x -e $MAS_DATA_ROOT/${date}/two_level_${tune_ctime}/CS${cs}/experiment.cfg 
sleep 1