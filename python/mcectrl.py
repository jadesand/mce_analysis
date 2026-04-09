# Copied from /home/mce/brianna/mcectrl.py
import numpy as np

import os
import re
import sys
import time
import subprocess
from datetime import datetime

#print str(datetime.now())

def get_card_and_dacid(row):
    mux11d_row_select_cards=get_mas_param('mux11d_row_select_cards')
    mux11d_row_select_cards_row0=[int(r0) for r0 in get_mas_param('mux11d_row_select_cards_row0')]
    mux11d_mux_order_num=int(get_mas_param('mux11d_mux_order')[row])
    #print mux11d_row_select_cards,mux11d_row_select_cards_row0,mux11d_mux_order_num
    dacnum=None
    card=None
    for (c,r0) in zip(mux11d_row_select_cards,mux11d_row_select_cards_row0):
        c=c.strip('"')
        if c=='ac':
            if mux11d_mux_order_num in range(r0,r0+41):
                dacnum=(mux11d_mux_order_num-r0)
                card=c
        if 'bc' in c:
            if mux11d_mux_order_num in range(r0,r0+32):
                dacnum=(mux11d_mux_order_num-r0)
                card=c
    return (card,dacnum)

def put_one_rs_on_all_rows(rs=0):
    orig_row_order=get_mas_param('row_order')
    new_row_order=[str(rs)]*len(orig_row_order)

    orig_default_row_select=get_mas_param('default_row_select')
    orig_row_deselect=get_mas_param('row_deselect')

    new_default_row_select=[orig_default_row_select[rs]]*len(orig_row_order)
    new_row_deselect=[orig_row_deselect[rs]]*len(orig_row_order)


    cmd='mas_param set row_order '+' '.join(new_row_order)
    print 'cmd=',cmd
    os.system(cmd)
    cmd='mas_param set default_row_select '+' '.join(new_default_row_select)
    print 'cmd=',cmd
    os.system(cmd)
    cmd='mas_param set row_select '+' '.join(new_default_row_select)
    print 'cmd=',cmd
    os.system(cmd)
    cmd='mas_param set row_deselect '+' '.join(new_row_deselect)
    print 'cmd=',cmd
    os.system(cmd)

def put_one_rs_on_all_rows_2level(rs=0):
    orig_row_order=get_mas_param('row_order')
    cs_idx = orig_row_order[-1]
    new_row_order=[str(rs)]*len(orig_row_order)
    new_row_order[-1] = cs_idx

    orig_default_row_select=get_mas_param('default_row_select')
    orig_row_deselect=get_mas_param('row_deselect')

    new_default_row_select=[orig_default_row_select[rs]]*len(orig_row_order)
    new_default_row_select[-1] = orig_default_row_select[-1]
    new_row_deselect=[orig_row_deselect[rs]]*len(orig_row_order)
    new_row_deselect[-1] = orig_row_deselect[-1]

    cmd='mas_param set row_order '+' '.join(new_row_order)
    print 'cmd=',cmd
    os.system(cmd)
    cmd='mas_param set default_row_select '+' '.join(new_default_row_select)
    print 'cmd=',cmd
    os.system(cmd)
    cmd='mas_param set row_select '+' '.join(new_default_row_select)
    print 'cmd=',cmd
    os.system(cmd)
    cmd='mas_param set row_deselect '+' '.join(new_row_deselect)
    print 'cmd=',cmd
    os.system(cmd)

def change_hybrid_row_order(row_order='0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 40 57 56 55 54 60 59 58 53 52 51 50 39 39 39 39 39 39 39 39 39 39 39'):
    if len(row_order.split())==len(get_mas_param('mux11d_mux_order')):
        cmd='mas_param set mux11d_mux_order '+row_order
        print '* cmd=',cmd
        os.system(cmd)
    else:
        print '* requested mux11d_mux_order is the wrong size ... doing nothing'
        sys.exit(1)

#def change_mux11d_mux_order(mux11d_mux_order='40 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71'):
#    # assume user passed the RS they want, like in plotting, not the actual DAC #.
#    orig_mux11d_mux_order=get_mas_param('mux11d_mux_order')
#
#    if len(mux11d_mux_order.split())==len(get_mas_param('mux11d_mux_order')):
#        cmd='mas_param set mux11d_mux_order '+mux11d_mux_order
#        print '* cmd=',cmd
#        os.system('mas_param set mux11d_mux_order '+mux11d_mux_order)
#    else:
#        print '* mux11d_mux_order is the wrong size ... doing nothing'
#        sys.exit(1)

def Config_OneDetectorReadoutw_MuxAt250kHz_ReadoutAt1kHz(row,col,data_mode=4,adjust_row_order=False):
    # this will configure the MCE to MUX at 250 kHz and sample one detector at ~1 kHz (977 Hz)
    print '* configuring 250 kHz MUX/1 kHz readout on c%dr%d w/ data_mode=%d ...'%(col,row,data_mode)

    # only need to do this if actually taking data on a detector at 250 kHz.  If you just
    # want the MUX to go at 250 kHz (ie for 50 MHz data) you don't have to care about the 
    # row_order, which requires a retuning...
    ## DEPRECATED
    ##if adjust_row_order:
    ##    # change put row as every entry in row_order
    ##    new_row_order=((' %d'%(row))*44).lstrip(' ')
    ##    change_row_order(row_order=new_row_order)    
    ##    
    ##    # return after changing the row_order;
    ##    # for some reason tuning after the below changes
    ##    # reverts to a default num_rows etc...
    ##    print '* tuning after changing row_order for 250 kHz MUX/1 kHz readout on c%dr%d ...'%(col,row)
    ##    os.system('auto_setup -d')

    # pretty sure I don't need to retune after the below changes,
    # but should probably think about it a little more ...
    # Configure readout scheme
    os.system('mce_cmd -x wb rca num_rows_reported 1')
    os.system('mce_cmd -x wb rca num_cols_reported 1')
    os.system('mce_cmd -x wb cc num_rows_reported 32')
    os.system('mce_cmd -x wb cc data_rate 256')
    os.system('mce_cmd -x wb rca data_mode %d'%(data_mode))
    
    # Pick a row, column (note only rows 0 and 1 will work)
    os.system('mce_cmd -x wb rca readout_row_index 0')
    os.system('mce_cmd -x wb rca readout_col_index %d'%(col))
    
    # Configure fast muxing 
    os.system('mce_cmd -x wb sys num_rows 2')
    
    # Check configuration
    os.system('python $MAS_PYTHON/rect_check.py --mce')

def get_mas_param(mas_param):
    mas_param_ret=get_command_result(["mas_param","get",mas_param])[0]
    return mas_param_ret.split()

def get_command_result(COMMAND,shell=False):
    cmd=subprocess.Popen(COMMAND,
                         shell=shell,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    result = cmd.stdout.readlines()
    if result == []:
        error = cmd.stderr.readlines()
        print >>sys.stderr, "ERROR: %s" % error
        return None
    else:
        return result

def get_first_word_in_file(filename,filepath='./'):
    first_word=''
    with open(os.path.join(filepath,filename),'r') as f:
        first_word=f.readline()
    print first_word
    return first_word.strip()

def bias_teses(pctRn=0,take_new_iv=False):
    if take_new_iv:
        os.system("""auto_iv --no-bias --do-plots=0""")
    
    last_iv=get_first_word_in_file('last_iv_completed_name','/data/cryo/')
    print 'last_iv=',last_iv

    if pctRn==0:
        os.system("""bias_tess 0""")
    else:
        array_id=get_first_word_in_file('array_id','/data/cryo')
        print 'array_id=',array_id        

        # specify pct Rn to compute optimal bias        
        os.system("""sed -i "s/^per\_Rn\_bias = .*/per\_Rn\_bias = 0.%02d;/g" /data/mas/config/array_%s.cfg"""%(pctRn,array_id))
        # compute optimal bias for this pct Rn
        os.system("""auto_iv analyze --do-plots=0 `readlink -f /data/cryo/last_iv_completed`""")
        # bias at this pct Rn based on this analysis
        os.system("""auto_iv bias `readlink -f /data/cryo/last_iv_completed`.out""")

def bias_teses_binary(bias=0,take_new_iv=False):
    if take_new_iv:
        os.system("""auto_iv --no-bias --do-plots=0""")
    
    last_iv=get_first_word_in_file('last_iv_completed_name','/data/cryo/')
    print 'last_iv=',last_iv

    if bias==0:
        os.system("""bias_tess 0""")
    else:
        array_id=get_first_word_in_file('array_id','/data/cryo')
        print 'array_id=',array_id        

        # compute optimal bias for this pct Rn
        os.system("""auto_iv analyze --do-plots=0 `readlink -f /data/cryo/last_iv_completed`""")
        # bias at this pct Rn based on this analysis
        os.system("""auto_iv bias `readlink -f /data/cryo/last_iv_completed`.out""")
        
def bias_steps_versus_pctRn(pctRns=[],wait_after_changing_bias=15):
    bias_steps_taken=()
    for pctRn in pctRns:
        print 'biasing @ %d%% Rn ...'%pctRn
        bias_teses(pctRn)
        
        print 'waiting %d sec after changing tes bias...'%(wait_after_changing_bias)
        time.sleep(wait_after_changing_bias)
        
        # take bias step
        os.system("""mce_bias_step_acq --frames=2500 --dwell=0.2508 --depth=100 --data-mode=4 --bc bc3""")

        # get name of bias step data file just created for logging
        last_bias_step=os.path.basename((((get_command_result(r'ls -t /data/cryo/current_data/*_bc3_step',True)[0]).split())[0]))
        #print 'last_bias_step=',last_bias_step
        bias_steps_taken=bias_steps_taken+((pctRn,last_bias_step),)
    return bias_steps_taken

def stop_temperature_logging():
    COMMAND="/etc/init.d/drlog stop"
    response=command_over_ssh(COMMAND)
    print response

def start_temperature_logging():
    COMMAND="/etc/init.d/drlog start"
    response=command_over_ssh(COMMAND)
    print response

def get_temperature_over_ssh(npts=1,time_btw_pts=0,lksn='370A7E'): 
    COMMAND="python /home/act/bkoop/lakeshore_setup/check_temperature_over_ssh.py %s %d %d"%(lksn,npts,time_btw_pts)

    # parse return value; should look like 'T6:8.004e+01PM0.000e+00MK\n'
    REGEXP_TO_MATCH=r'T([0-9]+)\:([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)PM([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)MK'
    response=command_over_ssh(COMMAND)
    # first line of output from script is annoyingly from locking the temperature readout
    m=re.match(REGEXP_TO_MATCH,response[1])
    
    if m!=None:
        channel=int(m.group(1))
        average_temperature=float(m.group(2))
        rms_temperature=float(m.group(4))
        return (channel,average_temperature,rms_temperature)
    else:
        print '... did not get sensible results from penzias for the current MXC temperature, response was;'
        print response
        print '... quitting.'

        sys.exit(1)

    return '...'

def change_servo_temperature(new_temperature_to_servo_to):
    COMMAND="python /home/act/bkoop/lakeshore_setup/change_servo_set_temperature_over_ssh.py %d"%(new_temperature_to_servo_to)
    for line in command_over_ssh(COMMAND):
        print line

def command_over_ssh(COMMAND,HOST="act@mr36.classe.cornell.edu"):
    
    print ["ssh", "%s" % HOST, COMMAND]
    ssh = subprocess.Popen(["ssh", "%s" % HOST, COMMAND],
                           shell=False,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    result = ssh.stdout.readlines()
    if result == []:
        error = ssh.stderr.readlines()
        print >>sys.stderr, "ERROR: %s" % error
        return None
    else:
        return result
