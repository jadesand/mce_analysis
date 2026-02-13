
default_columns=(4 5 6 7)
default_rcs=(2)

for CS in 10 11 12 13 14 15 16 17; do
    mce_zero_bias > /dev/null 2>&1
    mas_param set row_order 0 1 2 3 4 5 6 7 8 9 ${CS}
    auto_setup --rc=2
    
    run_mce_raw_acq 1 "${default_columns[@]}" "${default_rcs[@]}"
done