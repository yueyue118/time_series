import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import floating_axes
from mpl_toolkits.axisartist import grid_finder

plt.rcParams["font.family"] = "Times New Roman"


def test_set_plot(y_test_pre, y_test):
    plot_size = 0.33333
    length = int(len(y_test_pre) * plot_size)
    plt.plot(figsize=(12, 8))
    plt.ylim([3, 22])
    plt.plot(y_test_pre[len(y_test_pre) - length:], "r", label='Predicted', linewidth=4, linestyle='--')
    plt.plot(y_test[len(y_test_pre) - length:], "k", label='Measured', linewidth=4, alpha=0.8)
    plt.legend()


def save_csv(y_test_pre, y_test, file_name):
    result = pd.DataFrame({'y_test_pre': y_test_pre, 'y_test': y_test})
    result.to_csv(file_name, index=False, sep=',')


def cal_cc(y_test_pre, y_test):
    CC = np.corrcoef(y_test_pre, y_test)
    return CC


def set_taylor_axes(fig, location):
    trans = PolarAxes.PolarTransform()
    r1_locs = np.hstack((np.arange(1, 10) / 10.0, [0.95, 0.99]))
    t1_locs = np.arccos(r1_locs)
    gl1 = grid_finder.FixedLocator(t1_locs)
    tf1 = grid_finder.DictFormatter(dict(zip(t1_locs, map(str, r1_locs))))
    r2_locs = np.arange(0, 2, 0.25)
    r2_labels = ['0 ', '0.25 ', '0.50 ', '0.75 ', 'REF ', '1.25 ', '1.50 ', '1.75 ']
    gl2 = grid_finder.FixedLocator(r2_locs)
    tf2 = grid_finder.DictFormatter(dict(zip(r2_locs, map(str, r2_labels))))
    ghelper = floating_axes.GridHelperCurveLinear(trans, extremes=(0, np.pi / 2, 0, 1.75),
                                                  grid_locator1=gl1, tick_formatter1=tf1,
                                                  grid_locator2=gl2, tick_formatter2=tf2)
    ax = floating_axes.FloatingSubplot(fig, location, grid_helper=ghelper)
    fig.add_subplot(ax)

    ax.axis["top"].set_axis_direction("bottom")
    ax.axis["top"].toggle(ticklabels=True, label=True)
    ax.axis["top"].major_ticklabels.set_axis_direction("top")
    ax.axis["top"].label.set_axis_direction("top")
    ax.axis["top"].label.set_text("Correlation")
    ax.axis["top"].label.set_fontsize(14)
    ax.axis["left"].set_axis_direction("bottom")
    ax.axis["left"].label.set_text("Standard deviation")
    ax.axis["left"].label.set_fontsize(14)
    ax.axis["right"].set_axis_direction("top")
    ax.axis["right"].toggle(ticklabels=True)
    ax.axis["right"].major_ticklabels.set_axis_direction("left")
    ax.axis["bottom"].set_visible(False)
    ax.grid(True)
    polar_ax = ax.get_aux_axes(trans)

    rs, ts = np.meshgrid(np.linspace(0, 1.75, 100),
                         np.linspace(0, np.pi / 2, 100))
    rms = np.sqrt(1 + rs ** 2 - 2 * rs * np.cos(ts))
    CS = polar_ax.contour(ts, rs, rms, colors='gray', linestyles='--')
    plt.clabel(CS, inline=1, fontsize=10)
    t = np.linspace(0, np.pi / 2)
    r = np.zeros_like(t) + 1
    polar_ax.plot(t, r, 'k--')
    polar_ax.text(np.pi / 2 + 0.032, 1.02, " 1.00", size=10.3, ha="right", va="top",
                  bbox=dict(boxstyle="square", ec='w', fc='w'))

    return polar_ax


def plot_taylor(axes, ref_sample, sample, *args, **kwargs):
    std = np.std(ref_sample) / np.std(sample)
    corr = np.corrcoef(ref_sample, sample)
    theta = np.arccos(corr[0, 1])
    t, r = theta, std
    d = axes.plot(t, r, *args, **kwargs)
    return d
