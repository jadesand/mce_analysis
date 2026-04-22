# Copied from /home/mce/brianna/go_250kHz_2level.py
import re
import subprocess
import sys
import os
import datetime
import time
import numpy as np
import mcectrl
from mce_control import mce_control
from datetime import datetime
print str(datetime.now())

# only open once
mce=mce_control()
print mce

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

def flush_experiment_cfg():
    print '* flushing experiment.cfg and and re-tuning ...'    
    os.system('rm -fv /data/cryo/current_data/experiment.cfg')
    os.system('set_directory')

def main():

    # do everything in the current data directory
    os.chdir(os.path.realpath("/data/cryo/current_data"))

    data_mode=4 
    # the total # of samples will be data_rate * this, so 256x this for 250 kHz acquisition
    nsamples=25000 # 25.6 sec @ 250 kHz

    flush_before=False
    flush_after=False

    if flush_before:
        flush_experiment_cfg()

    print '!!! Disable dead mask...in this mode, only one channel so obviously user wants it servo\'d!'
    os.system('mas_param set config_dead_tes 1')

    tuned=False
    col=int(sys.argv[1])
    row=int(sys.argv[2])
    # log_file_ctime=int(sys.argv[3])

    extra_text_for_log_file=' '.join(sys.argv[4:])

    row_len=100
    sample_num=10
    sample_dly=row_len-sample_num

    rc=int(np.floor(col/8.))+1

    col_for_daq=col
    if rc>1:
        col_for_daq=col-8*(rc-1)
    print 'col_for_daq=',col_for_daq
    
    # should already be tuned when we get here...
    if not tuned or retune_at_each_bias:
        #
        mcectrl.put_one_rs_on_all_rows_2level(row)
        print '* tuning after changing row_order for 250 kHz MUX/1 kHz readout on c%dr%d ...'%(col,row)
        cmd='auto_setup --last-stage=rs_servo'
        os.system(cmd)
        
        # Why do I have to do this?
        mcectrl.put_one_rs_on_all_rows_2level(row)
        cmd='auto_setup --first-stage=sq1_servo_sa'
        os.system(cmd)
        
        # configure readout for fast muxing
        mcectrl.Config_OneDetectorReadoutw_MuxAt250kHz_ReadoutAt1kHz(row,col_for_daq,data_mode,adjust_row_order=False)
        tuned=True
        
    # set row_len
    os.system('mce_cmd -x wb sys row_len %d'%row_len)
    # set sample_dly
    os.system('mce_cmd -x wb rca sample_dly %d'%sample_dly)
    '''
    ctime=int(time.mktime(datetime.now().timetuple()))
    print '* taking data on c%dr%d ...'%(col,row)
    label='test'

    cmd="""mce_run %d_nr2_dr256_dm%d_250kHz_r%dc%d %d %d"""%(ctime,data_mode,row,col,nsamples,rc)
    print 'cmd=',cmd
    os.system(cmd)
    '''
    if flush_after:
        flush_experiment_cfg()
    
    print '* done'
    
if __name__ == "__main__":
    main()
