import numpy as np

import matplotlib
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt

import os
from collections import OrderedDict


##############
# SSA plots  #
##############

def plot_ssa_debug(fb, adu):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fb, adu)
    plt.close()


def _mark_phi0_boundaries(ax, x, x0, phi0, y_min, y_max):
    """Draw dashed vertical lines at each phi0 boundary starting from x0."""
    for phi in np.arange(0, np.max(x), phi0) + x0:
        ax.plot([phi, phi], [y_min, y_max], ls='--', color='k', lw=1, alpha=0.5)


def _mark_slope(ax, x_pt, y_pt, m, b, y_min, y_max, color):
    """Scatter the extremum point and draw its linear-fit trend line."""
    ax.scatter(x_pt, y_pt, c=color)
    ax.plot([(y_min - b) / m, (y_max - b) / m], [y_min, y_max], color + '--')


def _ssa_label(cfg, col, sa_bias_ua, Vmod_mV, phi0, M_fb, sa_fb_dac_to_uA, up, down):
    """Build the SSA annotation text box. `up`/`down` are dicts with
    dV_dFB, dV_dPhi0, dI_dI (see fit_slope_pair in main.py)."""
    return (
        r"\textbf{Column " + str(col) + "}\n\n" +
        r'\underline{Measured}' + '\n\n' +
        r"$I^{SSA}_{bias}$ = " + f"{sa_bias_ua:.1f} $\\mu$A\n" +
        "$V^{SSA}_{mod}$ = " + f"{Vmod_mV:0.3f} mV\n" +
        r"$\Phi^{SSA\,FB}_{0}$ = " + f"{phi0:.1f} $\\mu$A\n" +
        r"$M^{SSA}_{FB}$ = " + f"{M_fb:0.1f} pH" + '\n\n' +
        r"$dV^{SSA}_{ADU}/dFB^{SSA}_{DAC}$ $\downarrow$ = " + f"{down['dV_dFB']:.2f}" + '\n' +
        r"$dV^{SSA}_{nV}/dFB^{SSA}_{\mu\Phi_0}$ $\downarrow$ = " + f"{down['dV_dPhi0']:.2f}" + '\n' +
        r"$|dI^{SSA}_{IN,pA}/dV^{SSA}_{nV}|$ $\downarrow$ = " + f"{np.abs(1. / down['dI_dI']):.2f}" + '\n\n' +
        r"$dV^{SSA}_{ADU}/dFB^{SSA}_{DAC}$ $\uparrow$ = " + f"{up['dV_dFB']:.2f}" + '\n' +
        r"$dV^{SSA}_{nV}/dFB^{SSA}_{\mu\Phi_0}$ $\uparrow$ = " + f"{up['dV_dPhi0']:.2f}" + '\n' +
        r"$|dI^{SSA}_{IN,pA}/dV^{SSA}_{nV}|$ $\uparrow$ = " + f"{np.abs(1. / up['dI_dI']):.2f}" + '\n\n' +
        r'\underline{Assumed}' + '\n\n' +
        f"nV/ADU = {1.e9 * float(cfg['PREAMPADC']['ADU_TO_VOLTS_AT_PREAMP_INPUT']):.1f}" + '\n' +
        r"$M^{SSA}_{IN}$ = " + f"{cfg['SSA']['SSA_M_IN_PICOHENRY']} pH" + '\n' +
        f'SSAFB nA/DAC = {1000. * sa_fb_dac_to_uA:.3f}' + '\n' +
        r"$R^{cryo\,cable}_{roundtrip}$ = " +
        f"{cfg['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS']} $\\Omega$"
    )


def plot_ssa(cfg, fb, adu, phi0, max_idx, fb_mid, adu_mid,
             up_idx, down_idx, sa_fb_dac_to_uA, up, down,
             col, sa_bias_ua, Vmod_mV, M_fb, show_plot=True):
    """
    Plot the SSA flux-modulation curve with phi0 boundaries and up/downslope fits.

    up, down : dict with keys m, b, dV_dFB, dV_dPhi0, dI_dI (see main.fit_slope_pair)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fb, adu)

    y_min, y_max = ax.get_ylim()
    _mark_phi0_boundaries(ax, fb, fb[max_idx], phi0, y_min, y_max)
    ax.set_ylim([y_min, y_max])
    ax.set_xlim([np.min(fb), np.max(fb)])

    # Upslope is red, downslope is green
    _mark_slope(ax, fb_mid[up_idx], adu_mid[up_idx], up['m'], up['b'], y_min, y_max, 'r')
    _mark_slope(ax, fb_mid[down_idx], adu_mid[down_idx], down['m'], down['b'], y_min, y_max, 'g')

    ax.text(1.05, 0.5, _ssa_label(cfg, col, sa_bias_ua, Vmod_mV, phi0, M_fb, sa_fb_dac_to_uA, up, down),
            size=10, weight='bold', linespacing=1.5,
            bbox=dict(edgecolor='k', facecolor='none', pad=10, linewidth=1),
            ha='left', va='center', transform=ax.transAxes)

    plt.subplots_adjust(right=0.75)
    ax.set_xlabel('SSA Feedback Current ($\\mu$A)', fontsize=18)
    ax.set_ylabel('SSA Voltage (ADU)', fontsize=18)
    if show_plot:
        plt.show()
    plt.close()
    return ax


##############
# SQ1 plots  #
##############

def _fmt_or_none(value, fmt):
    return format(value, fmt) if value is not None else 'None'


def _sq1_label(cfg, col, row, phi0, M_fb, sa_fb_dac_to_uA, up, down):
    """Build the SQ1 annotation text box. `up`/`down` are dicts with
    m (dI_SSA_FB/dI_SQ1_FB) and dI_dI (dI_SSA_IN/dI_SQ1_IN); down's may be None."""
    return (
        r"\textbf{SQ1 on " + f'c{col:02}r{row:02}' + "}\n\n" +
        r'\underline{Measured}' + '\n\n' +
        r"$\Phi^{SQ1\,FB}_{0}$ = " + f"{phi0:.1f} $\\mu$A\n\n" +
        r"$M^{SQ1}_{FB}$ = " + f"{M_fb:0.1f} pH" + '\n\n' +
        r"$dI^{SSA}_{FB,\mu A}/dI^{SQ1}_{FB,\mu A}$ $\downarrow$ = " + _fmt_or_none(down['m'], '.2f') + '\n' +
        r"$dI^{SSA}_{IN,\mu A}/dI^{SQ1}_{IN,\mu A}$ $\downarrow$ = " + _fmt_or_none(down['dI_dI'], '.2f') + '\n\n' +
        r"$dI^{SSA}_{FB,\mu A}/dI^{SQ1}_{FB,\mu A}$ $\uparrow$ = " + f"{up['m']:.2f}" + '\n' +
        r"$dI^{SSA}_{IN,\mu A}/dI^{SQ1}_{IN,\mu A}$ $\uparrow$ = " + f"{up['dI_dI']:.2f}" + '\n\n\n' +
        r'\underline{Assumed}' + '\n\n' +
        f'SSAFB nA/DAC = {1000. * sa_fb_dac_to_uA:.3f}' + '\n' +
        r"$R^{cryo\,cable}_{roundtrip}$ = " +
        f"{cfg['CRYOCABLE']['CRYOCABLE_ROUNDTRIP_RESISTANCE_OHMS']} $\\Omega$" + '\n' +
        r"$M^{SQ1}_{IN}$ = " + f"{cfg['SQ1']['SQ1_M_IN_PICOHENRY']} pH\n"
    )


def plot_sq1(col, row, fb, servo, filter_sq1, servo_unfilt,
             max_idx, fb_mid, servo_mid, down_idx, down_fit_x, down_fit_y,
             up_idx, up_fit_x, up_fit_y, phi0, M_fb,
             sa_fb_dac_to_uA, sa_ax, cfg, up, down, show_plot=False):
    """
    Plot the SQ1 servo curve (max SSA FB vs SQ1 FB) with phi0 boundaries and
    up/downslope fits.

    up, down : dict with keys m, b, dI_dI (see main.fit_slope_pair); down's
    fields may be None if the downslope fit failed.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fb, servo)
    ax.scatter(fb, servo, s=10)
    if filter_sq1:
        ax.plot(fb, servo_unfilt, linestyle='--', color='gray')

    y_min, y_max = ax.get_ylim()
    _mark_phi0_boundaries(ax, fb, fb[max_idx], phi0, y_min, y_max)
    ax.set_ylim([y_min, y_max])
    ax.set_xlim([np.min(fb), np.max(fb)])

    # Downslope
    _mark_slope(ax, fb_mid[down_idx], servo_mid[down_idx], down['m'], down['b'], y_min, y_max, 'g')
    ax.scatter(down_fit_x, down_fit_y, s=50, facecolors='None', edgecolors='g')

    # Upslope
    _mark_slope(ax, fb_mid[up_idx], servo_mid[up_idx], up['m'], up['b'], y_min, y_max, 'r')
    ax.scatter(up_fit_x, up_fit_y, s=50, facecolors='None', edgecolors='r')

    ax.set_ylabel('SSA Feedback Current ($\\mu$A)', fontsize=18)
    ax.set_xlabel('SQ1 Feedback Current ($\\mu$A)', fontsize=18)

    sa_ax.text(1.05, 0.5, _sq1_label(cfg, col, row, phi0, M_fb, sa_fb_dac_to_uA, up, down),
               size=10, weight='bold', linespacing=1.5,
               bbox=dict(edgecolor='k', facecolor='none', pad=10, linewidth=1),
               ha='left', va='center', transform=sa_ax.transAxes)

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    if show_plot:
        plt.show()
    plt.close()


####################
# Ic min/max plots #
####################

# Fixed min/max color pair per (cs_state, rs_state). cs_state=None means
# 1-level muxing (no chip-select axis); it reuses the cs=True colors.
_ICMINMAX_COLORS = {
    (True, True):   ('b', 'deepskyblue'),
    (True, False):  ('#E4572E', '#FFB521'),
    (False, True):  ('#12B800', '#80FF72'),
    (False, False): ('#E55381', '#EFA9AE'),
    (None, True):   ('b', 'deepskyblue'),
    (None, False):  ('brown', 'lightcoral'),
}


def _icminmax_color(cs_state, rs_state):
    """(min_color, max_color) for one (cs_state, rs_state) condition. Edit
    _ICMINMAX_COLORS above to change the palette."""
    return _ICMINMAX_COLORS[(cs_state, rs_state)]


def condition_label(cs_state, rs_state):
    """Display label for one (cs_state, rs_state) condition, e.g. 'rs on' for
    1-level muxing (cs_state is None) or 'cs on rs off' for two-level."""
    rs = 'rs on' if rs_state else 'rs off'
    if cs_state is None:
        return rs
    cs = 'cs on' if cs_state else 'cs off'
    return f'{cs} {rs}'


def _label_icminmax_axes(ax, convert_units):
    """Set y/x labels and limits for an Ic min/max axis; returns the unit-name suffix."""
    if convert_units:
        ax.set_ylabel('SSA Input Current ($\\mu$A)', fontsize=10)
        ax.set_xlabel('SQ1 Total Bias Current ($\\mu$A)', fontsize=10)
        ax.set_ylim(0, 25)
        return 'ua'
    ax.set_ylabel('SSA Output $[\mathrm{ADU}]$', fontsize=10)
    ax.set_xlabel('$I_\mathrm{SQ1B}~[\mathrm{DAC}]$', fontsize=10)
    ax.set_ylim(0, 6000)
    return 'dac'


def plot_icminmax(col, row, conditions, ctime=None, convert_units=False,
                   savedir='../output_data', s1b_minmax_fig=None, s1b_minmax_ax=None,
                   manual_bias_idx=None, show_plot=False):
    """
    Plot SQ1 min/max vs SQ1 bias for one row, overlaying every condition.

    conditions : list of (cond, ic_params) pairs, cond = {'cs_state', 'rs_state'}.
    The first (primary) condition's max-modulation point and manual-bias
    marker are drawn; every condition after it also gets a dashed crosstalk
    "Bias Limit" line.
    """
    alpha = 1
    if s1b_minmax_fig is None:
        s1b_minmax_fig, s1b_minmax_ax = plt.subplots(figsize=(5, 4), dpi=150, layout='constrained')

    for i, (cond, ic_params) in enumerate(conditions):
        bias, ic_min, ic_max, max_idx, max_mod, start_idx, start_mod = ic_params
        min_c, max_c = _icminmax_color(cond['cs_state'], cond['rs_state'])
        label = condition_label(cond['cs_state'], cond['rs_state'])

        s1b_minmax_ax.plot(bias, ic_min, lw=2, label=f'SQ1 min, {label}', color=min_c, alpha=alpha)
        s1b_minmax_ax.plot(bias, ic_max, lw=2, label=f'SQ1 max, {label}', color=max_c, alpha=alpha)

        if i == 0:
            s1b_minmax_ax.plot([bias[max_idx], bias[max_idx]], [ic_min[max_idx], ic_max[max_idx]],
                                lw=3, color='purple', alpha=alpha,
                                label='$I^{SQ1}_{mod}$ = ' + f'{max_mod:.3f} $\\mu$A @ ' +
                                      '$I_{SQ1B,total} = $' + f'{bias[max_idx]:.1f} $\\mu$A')
            # if manual_bias_idx is not None:
            #     s1b_minmax_ax.plot([bias[manual_bias_idx], bias[manual_bias_idx]], [0, ic_max[manual_bias_idx]],
            #                         lw=2, color='orange', alpha=1, label='Manually Chosen Bias', linestyle='dotted')
            #     s1b_minmax_ax.plot([0, bias[manual_bias_idx]], [ic_max[manual_bias_idx], ic_max[manual_bias_idx]],
            #                         lw=2, color='orange', alpha=1, linestyle='dotted')
        else:
            bias_limit = bias[start_idx]
            # s1b_minmax_ax.plot([bias_limit, bias_limit], [0, bias[-1]],
            #                     label='Bias Limit' if i == 1 else None, color='deeppink', lw=3, linestyle="dotted")
            # s1b_minmax_ax.plot([0, bias[-1]], [start_mod, start_mod], color='deeppink', lw=3, linestyle="dotted")

    leg = s1b_minmax_ax.legend(loc='upper left', fontsize=10)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    
    uname = _label_icminmax_axes(s1b_minmax_ax, convert_units)
    s1b_minmax_ax.set_xlim(left=0)
    s1b_minmax_fig.suptitle('Ic Check Column ' + str(col) + ' Row ' + str(row))
    s1b_minmax_fig.tight_layout()
    savename = str(ctime) + '_icminmax_units' + uname + '_row' + str(row) + '_col' + str(col) + '.png'
    print('saving to: ' + os.path.join(savedir, savename))
    s1b_minmax_fig.set_facecolor('white')
    s1b_minmax_fig.savefig(os.path.join(savedir, savename), dpi=300)
    if show_plot:
        s1b_minmax_fig.show()
    s1b_minmax_ax.clear()
    print("Figures open: " + str(plt.get_fignums()))
    return s1b_minmax_fig, s1b_minmax_ax


def plot_icminmax_col(last_fig, col, conditions, ctime=None,
                       s1b_minmax_ax=None, s1b_minmax_fig=None, manual_bias_idx=None,
                       convert_units=False, show_plot=False, savedir='../output_data',
                       chip_num=None):
    """
    Overlay one row's Ic min/max/mod curve (every condition) onto a running
    per-chip summary plot. On the last row of the chip (last_fig=True),
    finalize, save, and close the figure.

    conditions : list of (cond, ic_params) pairs, cond = {'cs_state', 'rs_state'}.
    chip_num : if given, included in the plot title and filename so each
        chip's summary is a separate file.
    """
    alpha = 1 if last_fig else 0.1
    if s1b_minmax_ax is None:
        s1b_minmax_fig, s1b_minmax_ax = plt.subplots(figsize=(4, 4), dpi=150, layout='constrained')

    for i, (cond, ic_params) in enumerate(conditions):
        bias, ic_min, ic_max, max_idx, max_mod, start_idx, start_mod = ic_params
        min_c, max_c = _icminmax_color(cond['cs_state'], cond['rs_state'])
        label = condition_label(cond['cs_state'], cond['rs_state'])

        s1b_minmax_ax.plot(bias, ic_min, lw=2, color=min_c, alpha=alpha,
                            label=f'SQ1 min, {label}' if last_fig else None)
        s1b_minmax_ax.plot(bias, ic_max, lw=2, color=max_c, alpha=alpha,
                            label=f'SQ1 max, {label}' if last_fig else None)

        if i == 0:
            pass
            # s1b_minmax_ax.plot([bias[max_idx], bias[max_idx]], [ic_min[max_idx], ic_max[max_idx]],
            #                     lw=3, color='purple', alpha=alpha,
            #                     label='Maximum modulation' if last_fig else None)
            # if manual_bias_idx is not None:
            #     manual_alpha = 1 if last_fig else alpha
            #     s1b_minmax_ax.plot([bias[manual_bias_idx], bias[manual_bias_idx]], [0, ic_max[manual_bias_idx]],
            #                         lw=2, color='orange', alpha=manual_alpha,
            #                         label='Manually Chosen Bias' if last_fig else None, linestyle='dotted')
            #     s1b_minmax_ax.plot([0, bias[manual_bias_idx]], [ic_max[manual_bias_idx], ic_max[manual_bias_idx]],
            #                         lw=2, color='orange', alpha=manual_alpha, linestyle='dotted')
        elif last_fig:
            bias_limit = bias[start_idx]
            # s1b_minmax_ax.plot([bias_limit, bias_limit], [0, bias[-1]],
            #                     label='Bias Limit' if i == 1 else None, color='deeppink', lw=3, linestyle="dotted")
            # s1b_minmax_ax.plot([0, bias[-1]], [start_mod, start_mod], color='deeppink', lw=3, linestyle="dotted")

    if not last_fig:
        return s1b_minmax_fig, s1b_minmax_ax

    # leg = s1b_minmax_ax.legend(loc='upper left', fontsize=10)
    # for lh in leg.legend_handles:
    #     lh.set_alpha(1)
    s1b_minmax_ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    s1b_minmax_ax.axvline(16000, c='gray', lw=1.)

    def i2dac(x):
        return x / (3.7e-3)

    def dac2i(x):
        return x * (3.7e-3)
    ax2 = s1b_minmax_ax.secondary_xaxis('top', functions=(dac2i, i2dac))
    ax2.set_xlabel(r'$I_\mathrm{SQ1B}~[\mathrm{\mu A}]$')

    uname = _label_icminmax_axes(s1b_minmax_ax, convert_units)
    s1b_minmax_ax.set_xlim(left=0)
    chip_suffix = '' if chip_num is None else f' Chip {chip_num}'
    # s1b_minmax_fig.suptitle('Ic Check Column ' + str(col) + chip_suffix)
    # s1b_minmax_fig.tight_layout()
    chip_filesuffix = '' if chip_num is None else f'_chip{chip_num}'
    savename = str(ctime) + '_icminmax_units' + uname + '_summary_col' + str(col) + chip_filesuffix + '.png'
    print('saving to: ' + os.path.join(savedir, savename))
    s1b_minmax_fig.set_facecolor('white')
    s1b_minmax_fig.savefig(os.path.join(savedir, savename), dpi=300)
    if show_plot:
        s1b_minmax_fig.show()
    print("Figures open: " + str(plt.get_fignums()))
    plt.close('all')
    return None, None


###################
# Row-select plot #
###################

def plot_rsservo_col(last_fig, col, chip_num, sq1_params, sq1_params2=None, ctime=None,
                      s1b_minmax_ax=None, s1b_minmax_fig=None, show_plot=False,
                      savedir='../output_data/'):
    """Plot the row-select servo curve at the max-modulation bias point for one chip."""
    try:
        os.mkdir(savedir)
    except FileExistsError:
        pass
    colors = ['deepskyblue', 'red', 'green', 'purple']
    color = colors[chip_num]
    alpha = 0.2
    curves, biases, max_span_bias, max_span, servo = sq1_params

    if s1b_minmax_ax is None:
        s1b_minmax_fig, s1b_minmax_ax = plt.subplots(figsize=(8, 6))

    s1b_minmax_ax.plot(servo, curves[0], alpha=alpha, color=color, label='Chip Number: ' + str(chip_num))
    if sq1_params2 is not None:
        curves2, biases2, max_span_bias2, max_span2, servo2 = sq1_params2
        s1b_minmax_ax.plot(servo2, curves2[0], alpha=alpha, color='aqua')

    if not last_fig:
        return s1b_minmax_fig, s1b_minmax_ax

    handles, labels = s1b_minmax_ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    leg = s1b_minmax_ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    s1b_minmax_ax.set_ylabel('SSA Input Current (DAC units)', fontsize=18)
    s1b_minmax_ax.set_xlabel('SQ1 Total Bias Current (DAC units)', fontsize=18)
    s1b_minmax_fig.suptitle('RS Check Column ' + str(col))
    s1b_minmax_fig.tight_layout()
    savename = str(ctime) + '_rs_summary_col' + str(col) + '.png'
    print('saving to: ' + os.path.join(savedir, savename))
    s1b_minmax_fig.set_facecolor('white')
    s1b_minmax_fig.savefig(os.path.join(savedir, savename), dpi=150)
    if show_plot:
        plt.show()
    plt.close()
    return s1b_minmax_fig, s1b_minmax_ax


##############
# Tile plot  #
##############

def tile_plot(rows, cols, data, label, title, vmin=0, vmax=20, cmap='bwr',
              savedir='../output_data', show_plot=False, display_title=None):
    """
    Draw a row-vs-column heatmap. `data` is indexed as data[row][col].
    `title` names the saved file (and is the on-plot title if display_title
    is not given); pass display_title separately when `title` needs to stay
    long/unique for the filename but the on-plot text should be shorter.
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 6), layout='constrained', dpi=150)
    im = ax.imshow(data, interpolation='none', aspect='equal', cmap=cmap, vmin=vmin, vmax=vmax)

    col_min, col_max = np.min(cols), np.max(cols)
    row_min, row_max = np.min(rows), np.max(rows)
    ax.set_xticks(np.arange(col_min, col_max, 4))
    ax.set_yticks(np.arange(row_min, row_max, 3))
    ax.set_xticks(np.arange(col_min - .5, col_max, 1), minor=True)
    ax.set_yticks(np.arange(row_min - .5, row_max, 1), minor=True)

    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax.set_xlim(col_min - 0.5, col_max + 0.5)
    ax.set_ylim(row_min - 0.5, row_max + 0.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_title(display_title if display_title is not None else title, fontsize=10, wrap=True)

    cbar = fig.colorbar(im)
    cbar.set_label(label)

    savename = os.path.join(savedir, title + '.png')
    fig.set_facecolor('white')
    print('saving: ' + savename)
    plt.savefig(savename, dpi=150)
    if show_plot:
        plt.show()
    plt.close('all')
