"""Plots candlestick charts for a single day of data.

Always plots candlesticks and volume bars by default. Includes functions to plot technical
indicators such as the MACD, RSI, and EMA. Additionally, entry and exit points if trade
information is passed by user.
"""

__author__ = "Dante St. Prix"

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import pandas as pd
import numpy as np
import string
import bisect
from typing import Dict, List, Union


class CandleGraph:
    """"Creates candlestick plots for 1 day of data for a given symbol.

    Creates candlestick, volume and indicator (only if function called to plot RSI or MACD)
    plots. After calling CandleGraph.show(), a new object should be created for additional plots.
    """

    def __init__(self, symbol: str, data: Dict[str, np.ndarray], timeframe: int, i,
                 trades: Dict[str, Union[float, List]] = None, open_hours: bool = False,
                 strategy: str = None):
        """Initializes initial state of plot.

        A plot is created for the entire day corresponding to a bar at index i. The first and
        last indices of the day are calculated automatically. Creates candlestick and volume
        plots. Call instance functions to add indicators to the plot.

        Args:
            symbol: Symbol name.
            data: Dictionary containing all data for this symbol.
            timeframe: Bar length in minutes.
            i: Any index corresponding to a specified day.
            trades: Trade information (optional)
            open_hours: Specifies whether bars corresponding to times during which the market is
            closed are included. Set to True to only include bars during open-hours.
            strategy: Name of trading strategy (optional)
        """

        def find_day_indices(integer_time, i):
            """Finds indices corresponding to the start and end of the day."""
            day = (integer_time[i] // 10000) * 10000
            next_day = (1 + integer_time[i] // 10000) * 10000
            max_candle = int(
                390 * 2 / timeframe)  # 390 minute bars in a day -> assumes less morning bars
            # than open-hours bars.
            start_index = bisect.bisect_left(integer_time, day, lo=max(0, i - max_candle), hi=i)
            end_index = bisect.bisect_right(integer_time, next_day, lo=i,
                                            hi=min(i + max_candle, len(integer_time) - 1))
            return start_index, end_index

        def find_open_close_indices(integer_time, i):
            """Finds indices corresponding to when the market opens and closes on this day."""
            day = (integer_time[i] // 10000) * 10000
            day_open = day + 930
            day_close = day + 1560
            max_candle = int(
                390 * 2 / timeframe)  # 390 minute bars in a day -> assumes less morning bars
            # than open-hours bars.
            open_index = bisect.bisect_left(integer_time, day_open, lo=max(0, i - max_candle),
                                            hi=min(i + max_candle, len(integer_time) - 1))
            close_index = bisect.bisect_right(integer_time, day_close, lo=i,
                                              hi=min(i + max_candle, len(integer_time) - 1))
            return open_index, close_index

        open_index, close_index = find_open_close_indices(data['integer time'], i)
        if open_hours:
            i = max(0,
                    open_index - 3)  # i, j = first and last indices when the market is open (+- 3)
            j = min(close_index + 3, len(data['close']) - 1)
        else:
            i, j = find_day_indices(data['integer time'],
                                    i)  # i, j = first and last indices of the day, respectively.

        self.fig = plt.figure(constrained_layout=False, figsize=(18, 10), num=4, clear=True)
        self.ax = self.fig.add_gridspec(14, 6)
        self._main_ax = self.fig.add_subplot(self.ax[0:6, 0:6])
        self._vol_ax = self.fig.add_subplot(self.ax[6:8, 0:6])
        self._ind_ax = None
        self._set_title(symbol, strategy)
        # self.day_ax = self.fig.add_subplot(ax[10:12, 0:6])  # To be added later
        # self.dayvol_ax = self.fig.add_subplot(ax[12:14, 0:6])

        self.c = data['close'][i:j]
        self.o = data['open'][i:j]
        self.l = data['low'][i:j]
        self.h = data['high'][i: j]
        self.v = data['volume'][i:j]
        self.indicators = {key: data[key][i:j] for key in list(data.keys())[6:]}
        self.green_indices, self.red_indices, self.doji_indices = self._get_bar_color()
        self.timestamps = np.array(
            [dt.datetime.strptime(timestamp, '%Y-%m-%d %H:%M') for timestamp in data['time'][i:j]])
        self._plot_candlesticks()
        self._plot_volume_bars()
        self._lock_ylim()
        self._plot_trades(trades)
        if open_hours:
            self._add_open_close_lines(3, -3)
        else:
            self._add_open_close_lines(open_index - i, close_index - i)
        self._manage_x_axis(self._main_ax, self._vol_ax)
        self.displayed = False

    def _set_title(self, symbol, strategy):
        """Sets title in form of 'symbol: mm/DD/YYYY (strategy name)'"""
        if strategy is not None:
            self._main_ax.set_title(
                '{}: {} ({})'.format(symbol, timestamps[i].strftime('%m/%d/%Y'), strategy),
                fontsize=16)
        else:
            self._main_ax.set_title('{}: {}'.format(symbol, timestamps[i].strftime('%m/%d/%Y')),
                                    fontsize=16)

    def _get_bar_color(self):
        """Determines whether a bar is bullish (green) or bearish (red)."""
        return np.where(self.c > self.o), np.where(self.c < self.o), np.where(self.c == self.o)

    def _plot_candlesticks(self):
        linewidth = 5.5

        green_body_min = self.o[self.green_indices]
        green_body_high = self.c[self.green_indices]
        red_body_min = self.c[self.red_indices]
        red_body_high = self.o[self.red_indices]

        self._main_ax.vlines(self.timestamps, self.l, self.h, linewidth=0.5, color='k')
        self._main_ax.vlines(self.timestamps[self.green_indices], green_body_min, green_body_high,
                             linewidth=linewidth, color='g')
        self._main_ax.vlines(self.timestamps[self.red_indices], red_body_min, red_body_high,
                             linewidth=linewidth, color='r')

        plt.setp(self._main_ax.get_xticklabels(), color='w')

    def _plot_volume_bars(self):
        """Plots volume bars. Doji bars are gray."""
        linewidth = 4.5
        green_volume = self.v[self.green_indices]
        red_volume = self.v[self.red_indices]
        doji_volume = self.v[self.doji_indices]
        self._vol_ax.vlines(self.timestamps[self.green_indices], 0, green_volume,
                            linewidth=linewidth, color='g', alpha=0.8)
        self._vol_ax.vlines(self.timestamps[self.red_indices], 0, red_volume, linewidth=linewidth,
                            color='r', alpha=0.8)
        self._vol_ax.vlines(self.timestamps[self.doji_indices], 0, doji_volume,
                            linewidth=linewidth, color='tab:gray', alpha=0.2)
        self._vol_ax.set_ylim(0, (np.median(self.v) + np.amax(
            self.v)) / 2)  # Set to crop highest bars to make average bars visible.
        plt.setp(self._vol_ax.get_xticklabels(), rotation=45, ha='right')

    def _plot_trades(self, trades):
        """Plots trade information if provided.

        Vertical lines created at entry (yellow), stoploss (red) and target (blue) prices
        between the entry and exit times. Trailing stoplosses are represented as lines starting
        when the new stoploss is set and extends until the exit time. Lines representing
        stoplosses below breakeven are solid red, at breakeven are dotted red and above are
        solid green.
        """
        if trades is None:
            return
        self._main_ax.hlines(0, 0, 0, color='y',
                             label='Entry')  # Invisible line to get proper legend label.
        self._main_ax.hlines(0, 0, 0, color='b', label='Target')
        self._main_ax.hlines(0, 0, 0, color='r', label='Stoploss')
        for trade in trades:
            entry_time = trade['entry time']
            exit_time = trade['exit time']
            self._main_ax.hlines(trade['entry'], entry_time, exit_time, color='y')
            self._main_ax.plot(entry_time, trade['entry'], marker=6, color='y',
                               markeredgecolor='k')
            self._main_ax.hlines(trade['target'], entry_time, exit_time, color='b')
            self._main_ax.plot(exit_time, trade['exit'], marker=7, color='y', markeredgecolor='k')
            for stoploss in trade['stoploss']:
                color = 'r' if stoploss[0] <= trade['entry'] else 'g'
                linestyle = '-' if stoploss[0] != trade['entry'] else '--'
                self._main_ax.hlines(stoploss[0], stoploss[1], exit_time, color=color,
                                     linestyle=linestyle)

    def _lock_ylim(self):
        """Prevents the y-axis of the candlestick plot from changing."""
        y_lim = self._main_ax.get_ylim()
        self._main_ax.set_ylim(y_lim[0], y_lim[1])

    def _add_open_close_lines(self, i, j):
        """Adds faint vertical lines when the market opens and closes."""
        y_lim = self._main_ax.get_ylim()
        self._main_ax.vlines([self.timestamps[i], self.timestamps[j]], y_lim[0], y_lim[1],
                             color='k', alpha=0.35, linewidth=1, linestyle='-.')

    def _manage_x_axis(self, *args):
        """Formats the plot so the time properly shows on the x-axis."""
        for ax in args:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45], interval=1))
            formatter = mdates.DateFormatter('%H:%M')
            ax.xaxis.set_major_formatter(formatter)
            ax.set_xlim(left=self.timestamps[0], right=self.timestamps[-1])
            ax.grid(axis='x', alpha=0.4)

    def plot_EMA(self, n):
        """Plots the n-period EMA."""
        self._main_ax.plot(self.timestamps, self.indicators['{} EMA'.format(n)],
                           label='{} EMA'.format(n))

    def _set_indicator_plot(self):
        """Creates indicator plot below volume plot (currently used only for plotting MACD and
        RSI)."""
        if self._ind_ax is not None:
            del self._ind_ax.lines[:]
        else:
            self._ind_ax = self.fig.add_subplot(self.ax[8:10, 0:6])
            plt.setp(self._ind_ax.get_xticklabels(), rotation=45, ha='right')
            plt.setp(self._vol_ax.get_xticklabels(), color='w')

    def plot_RSI(self):
        """Deletes MACD plot if previously plotted."""
        self._set_indicator_plot()
        self._ind_ax.plot(self.timestamps, self.indicators['RSI'], label='RSI')
        self._ind_ax.axhline(30, color='k', linestyle='-.', alpha=0.35, linewidth=1)
        self._ind_ax.axhline(70, color='k', linestyle='-.', alpha=0.35, linewidth=1)
        self._ind_ax.set_ylim([0, 100])
        self._manage_x_axis(self._ind_ax)

    def plot_MACD(self):
        """Deletes RSI plot if previously plotted."""
        self._set_indicator_plot()
        self._ind_ax.plot(self.timestamps, self.indicators['MACD'], label='MACD')
        self._ind_ax.plot(self.timestamps, self.indicators['signal'], label='signal')
        self._ind_ax.axhline(0, color='k', linestyle='-.', alpha=0.35, linewidth=1)
        y_limit = max(np.amax(np.absolute(self.indicators['MACD'])),
                      np.absolute(np.amax(self.indicators['signal']))) * 1.25
        self._ind_ax.set_ylim([-y_limit, y_limit])
        self._manage_x_axis(self._ind_ax)

    def plot_VWAP(self):
        self._main_ax.plot(self.timestamps, self.indicators['VWAP'], label='VWAP')

    def plot_bollinger(self):
        """Lightly fills the space between the upper and lower Bolling Bounds."""
        bb_plus = self.indicators['Bollinger (Lower)']
        bb_minus = self.indicators['Bollinger (Upper)']
        self._main_ax.plot(self.timestamps, bb_minus, label='Bollinger Bands', alpha=0.5,
                           color='b', linewidth=1, linestyle='-.')
        self._main_ax.plot(self.timestamps, bb_plus, alpha=0.5, color='b', linewidth=1,
                           linestyle='-.')
        self._main_ax.fill_between(self.timestamps, bb_minus, bb_plus, alpha=0.05, color='b')

    def plot_pattern(self, *args, all=False):
        """Plots candlestick patterns.

        Plots all patterns given in args. If all=True, all available candlestick patterns are
        shown. Each occurrence of a pattern is represented by a small marker.
        """
        patterns = args
        y_lim = self._main_ax.get_ylim()
        offset = (y_lim[1] - y_lim[0]) * 0.04
        if all:
            patterns = ['Hammer', 'Shooting Star', 'Green Marubozu', 'Red Marubozu',
                        'Bullish Engulfing', 'Bearish Engulfing', 'Evening Star', ' Morning Star',
                        'Tweezer Top', 'Tweezer Bottom']
        if not patterns:
            raise ValueError('No candlestick pattern specified')
        for pattern in patterns:
            pattern = string.capwords(pattern)
            indices = np.flatnonzero(self.indicators[pattern])
            if indices.size > 0:
                if pattern == 'Hammer':
                    self._main_ax.plot(self.timestamps[indices], self.l[indices] - offset,
                                       marker='1', color='r', linestyle='None', label='Hammer')
                elif pattern == 'Shooting Star':
                    self._main_ax.plot(self.timestamps[indices], self.h[indices] + offset,
                                       marker='2', color='y', linestyle='None',
                                       label='Shooting Star')
                elif pattern == 'Spinning Top':
                    self._main_ax.plot(self.timestamps[indices],
                                       (self.o[indices] + self.c[indices]) / 2, marker='P',
                                       color='b', linestyle='None', label='Spinning Top')
                elif pattern == 'Green Marubozu':
                    self._main_ax.plot(self.timestamps[indices], self.l[indices] - offset,
                                       marker='s', color='g', linestyle='None',
                                       label='Green Marubozu')
                elif pattern == 'Red Marubozu':
                    self._main_ax.plot(self.timestamps[indices], self.h[indices] + offset,
                                       marker='s', color='r', linestyle='None',
                                       label='Red Marubozu')
                elif pattern == 'Bullish Engulfing':
                    self._main_ax.plot(self.timestamps[indices], self.l[indices] - offset,
                                       marker='^', color='g', linestyle='None',
                                       label='Bullish Engulfing')
                elif pattern == 'Bearish Engulfing':
                    self._main_ax.plot(self.timestamps[indices], self.h[indices] + offset,
                                       marker='v', color='r', linestyle='None',
                                       label='Bearish Engulfing')
                elif pattern == 'Evening Star':
                    self._main_ax.plot(self.timestamps[indices], self.h[indices] + offset,
                                       marker='*', color='b', linestyle='None',
                                       label='Evening Star')
                elif pattern == 'Morning Star':
                    self._main_ax.plot(self.timestamps[indices], self.l[indices] - offset,
                                       marker='*', color='r', linestyle='None',
                                       label='Morning Star')
                elif pattern == 'Tweezer Top':
                    self._main_ax.plot(self.timestamps[indices], self.h[indices] + offset,
                                       marker='d', color='r', linestyle='None',
                                       label='Tweezer Top')
                elif pattern == 'Tweezer Bottom':
                    self._main_ax.plot(self.timestamps[indices], self.l[indices] - offset,
                                       marker='d', color='g', linestyle='None',
                                       label='Tweezer Bottom')

    def show(self):
        """Displays plot.

        Throws error if user attempts to plot using same object twice.
        """
        if self.displayed:
            raise RuntimeError('Cannot plot same object twice.')
        self.displayed = True
        plt.subplots_adjust(hspace=0.5)
        self._main_ax.legend()
        if self._ind_ax is not None:
            self._ind_ax.legend()
        plt.show()
