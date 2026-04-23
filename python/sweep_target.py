import numpy as np

import os
import sys
import time
from datetime import datetime
from subprocess import Popen,PIPE

from mce_control import mce_control
import auto_setup as ast

_SAB = {
    'range': range(-100,100,5),
    'nframes': 1000,
    'card': 'sa',
    'param': 'bias',
}
_SAFB= {
    'range': range(-1500,1501,15),
    'nframes': 30,
    'card': 'sa',
    'param': 'fb',
}
_SQ1B= {
    'range': range(-100,100,5),
    'nframes': 1000,
    'card': 'sq1',
    'param': 'bias',
}
_SQ1FB= {
    'range': range(-500,501,5),
    'nframes': 30,
    'card': 'sq1',
    'param': 'fb_const',
}

MAS_DATA = os.environ.get('MAS_DATA')


USAGE="""
%prog [options] <sweep_target> <row>

Sweep the specified parameter for the specified row, and save the data to a file
in the current data directory with a name like 
"<timestamp>_openloop_ramp_<sweep_target>_r<row>.dat".

The "sweep_target" argument indicates which parameter to sweep, can be one of:
 - sab: SA bias
 - safb: SA feedback
 - sq1b: SQ1 bias
 - sq1fb: SQ1 feedback

The "row" argument indicates which row to read out for the data.  
"""

from optparse import OptionParser
o = OptionParser(usage=USAGE)
o.add_option('--config', default=None, type=str,
             help="tune config file.  If not provided, will use the default config file for the current setup.")
opts, args = o.parse_args()

SWEEP_TARGET = ['sab', 'safb', 'sq1b', 'sq1fb']
if len(args) != 2:
    o.error("Provide the sweep type: (%s) and the row number." % ','.join(SWEEP_TARGET))
if args[0] not in SWEEP_TARGET:
    o.error("Provide the sweep type: (%s)." % ','.join(SWEEP_TARGET))


# Load config file
if opts.config is None:
    mas_path = ast.util.mas_path()
    exp_file = mas_path.experiment_file()
else:
    exp_file = opts.config
cfg = ast.config.configFile(exp_file)


target = args[0].lower()
row = int(args[1])
ctime = int(time.mktime(datetime.now().timetuple()))

mce = mce_control()


target_dict = eval('_'+args[0].upper())
nframes = target_dict['nframes']
sweep_range = target_dict['range']
card = target_dict['card']
param = target_dict['param']
print '{}={}'.format(args[0].lower(), sweep_range)

ofn = os.path.join(MAS_DATA, '%d_openloop_ramp_%s_r%d.dat'%(ctime, target, row))
of = open(ofn,'a+')

orig = Popen(["mce_cmd","-x","rb",card,param],stdout=PIPE).communicate()[0].strip()
orig = np.array([int(ob) for ob in Popen(["mce_cmd","-x","rb",card,param],stdout=PIPE).communicate()[0].strip().split('\n')[1].split(':')[2].split()],'int')
print 'orig_%s_%s='%(card, param), orig

columns_off = np.array(cfg['columns_off'][:len(orig)])
print 'columns_off=',columns_off

ncol = len(orig)
for swp in sweep_range:

    new = orig + np.array([swp]*ncol)

    # make sure we don't sweep sab on columns in columns off
    new[np.where(columns_off==1)]=0

    print 'new_%s_%s='%(card, param), new
    mce.write(card, param, new)
    time.sleep(0.1) # let settle

    # 0 in [0,:,:]=row, which shouldn't matter for this data
    data = mce.read_data(nframes, row_col=True).data[row,:,:]    
    errs, derrs = data.mean(axis=-1), data.std(axis=-1)
    err=['%.4e'%err for err in errs]
    of.write(str(swp)+'\t'+'\t'.join(err)+'\n')

mce.write(card, param, orig)    
of.close()
