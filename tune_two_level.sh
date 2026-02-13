# array_id = two_level_20260210
PARENT_NAME="two_level_$(date +%s)"

# Resolve the symlink to get the real path
MAS_DATA_REAL=$(readlink -f "$MAS_DATA")

mkdir -p "$MAS_DATA_REAL/$PARENT_NAME"
mkdir -p "$MAS_DATA_REAL/analysis/$PARENT_NAME"

{
for CS in 10 11 12 13 14 15 16 17; do
    echo "CS=$CS"
    
    mce_zero_bias > /dev/null 2>&1
    mas_param set row_order 0 1 2 3 4 5 6 7 8 9 ${CS}
    
    auto_setup --rc=2
    
    # Find the most recently created directory
    LATEST=$(find "$MAS_DATA_REAL" -maxdepth 1 -type d \
             -not -path "$MAS_DATA_REAL" \
             -not -path "$MAS_DATA_REAL/$PARENT_NAME" \
             -printf '%T+ %p\n' | sort -r | head -1 | cut -d' ' -f2-)
    
    if [[ -n "$LATEST" ]]; then
        DIRNAME=$(basename "$LATEST")
        mv "$MAS_DATA_REAL/$DIRNAME" "$MAS_DATA_REAL/$PARENT_NAME/CS$((CS-10))"
        mv "$MAS_DATA_REAL/analysis/$DIRNAME" "$MAS_DATA_REAL/analysis/$PARENT_NAME/CS$((CS-10))"
    fi
done
} 2>&1 | tee "$MAS_DATA_REAL/$PARENT_NAME/log"