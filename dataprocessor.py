"""Imports data from Alpaca, calculates indicators and saves to file.

All data for a list of specified symbols are imported and processed for a given time period.
Data for all symbols are combined to one Pandas DataFrame and saved to a .Feather file. A file
is created for each month. Another file specifying the start and end indices of each symbol is
created. The user can select the timeframe of the bars.
"""

__author__ = 'Dante St. Prix'

import threading
import multiprocessing
from pyrfc3339 import generate
import datetime as dt
from dateutil.relativedelta import relativedelta
import pytz
from dateutil import tz
import requests
import json
import numpy as np
import pandas as pd
import time
from numba import njit
import math
import csv
from typing import Tuple, Union, Type, Dict, List, Set

API_KEY = < Insert API key here >
SECRET_KEY = < Insert secret key here >
FILEPATH = < Insert location to save files to here >
HEADERS = {'APCA-API-KEY-ID': API_KEY, 'APCA-API-SECRET-KEY': SECRET_KEY}
MARKET_DATA_BASE = 'https://data.alpaca.markets/'
UTC_ZONE = tz.gettz('UTC')
EST_ZONE = tz.gettz('America/New_York')


def import_data(month: Tuple[str, str], timeframe: str, symbols: List[str],
                data_dict: Dict[str, Dict[str, Union[List[str], np.ndarray]]]):
    """"Imports historical market data from Alpaca.markets

    Gets a single month of data for the specified symbols from alpaca.markets using the requests
    module. Saves data to data_dict as 64-bit floats (This is done for better accuracy when
    calculating indicators. Floats stored in file will be 32 bits).

    Args:
        month: Start and end times for the month in RFC3339-format.
        timeframe: Integer followed by min, e.g., '5min'.
        symbols: List of symbols.
        data_dict: Dictionary to which imported data is saved.
    """
    print('Importing {}'.format(month))
    start = time.perf_counter()
    leftover_symbols = set(symbols)
    lock = threading.Lock()
    time_dict = {}
    bars_url = '{}/v2/stocks/bars?timeframe={}'.format(MARKET_DATA_BASE, timeframe)

    def lists_to_np_float(size: int, *args: list) -> Tuple[np.ndarray, ...]:
        """Converts lists containing floats to Numpy arrays.

        Args:
            size: Size of array element (only supports 32 and 64 bits).
            *args: Lists to convert.

        Returns:
            Numpy array.
        """
        if size not in [32, 64]:
            raise ValueError('Invalid float size. Inputs should be one of 32 and 64')
        dtype = np.float32 if size == 32 else np.float64
        output = []
        for arg in args:
            output.append(np.asarray(arg, dtype=dtype))
        return tuple(output)

    def import_symbol():
        """Imports historical market data for one symbol.

        Used as a thread's target function. Adds imported data to data_dict.
        """
        s = requests.Session()
        while leftover_symbols:
            with lock:
                if not leftover_symbols:
                    break
                symbol = leftover_symbols.pop()
                if len(leftover_symbols) % 100 == 0:
                    print('{} symbols left'.format(len(leftover_symbols)))
            page_token = 1
            i = 0
            o, c, h, l, v, timestamp = [], [], [], [], [], []  # Arrays to keep track of open,
            # close, low, high, volume and bar time
            min, hour, int_date = [], [], []
            while page_token != None:
                params = {'symbols': symbol,
                          'timeframe': timeframe,
                          'start': month[0],
                          'end': month[1],
                          'limit': 10000,
                          'adjustment': 'split'}
                params['page_token'] = None if page_token == 1 else page_token
                while True:
                    try:
                        if not len(leftover_symbols):
                            r = s.get(bars_url, params=params, headers=HEADERS, timeout=2)
                        else:
                            r = s.get(bars_url, params=params, headers=HEADERS, timeout=(3.05, 20))
                        raw_data = json.loads(r.content)
                    except Timeout:
                        print('{} - Connection failed.. trying again'.format(
                            dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                        time.sleep(0.5)
                    else:
                        break
                if 'bars' not in raw_data.keys() or raw_data['bars'] == {}:
                    # print('{} - No data found for {}'.format(dt.datetime.now().strftime(
                    # '%Y-%m-%d %H:%M:%S'), symbol))
                    break
                for bar in raw_data['bars'][
                    symbol]:  # Raw data in the form of list of dictionaries
                    o.append(bar['o'])
                    c.append(bar['c'])
                    l.append(bar['l'])
                    h.append(bar['h'])
                    v.append(bar['v'])
                    time_string = f'{bar["t"][:10]} {bar["t"][11:16]}'
                    if time_string in time_dict.keys():
                        converted_time = time_dict[time_string]
                    else:  # If first time converting timezone from UTC to EST.
                        converted_time = str(
                            dt.datetime.strptime(time_string, '%Y-%m-%d %H:%M').replace(
                                tzinfo=UTC_ZONE).astimezone(EST_ZONE))[:16]
                        time_dict[time_string] = converted_time
                    int_date.append(
                        int(converted_time[2:4]) * 10000 + int(converted_time[5:7]) * 100 + int(
                            converted_time[8:10]))
                    min.append(int(converted_time[14:]))
                    hour.append(int(converted_time[11:13]))
                    timestamp.append(converted_time)
                page_token = raw_data['next_page_token']

            o, c, l, h, v, min, hour, int_date = lists_to_np_float(64, o, c, l, h, v, min, hour,
                                                                   int_date)  # converts lists
            # to 32 bit numpy arrays
            data_dict[symbol] = {'time': timestamp,
                                 'open': o,
                                 'close': c,
                                 'low': l,
                                 'high': h,
                                 'volume': v,
                                 'min': min,
                                 'hour': hour,
                                 'integer date': int_date}

    max_threads = 100  # Adjust to maximize performance.
    threads = []
    for i in range(max_threads):
        t = threading.Thread(target=import_symbol)
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
    end = time.perf_counter()
    print('Imported data in {}s'.format(end - start))


@njit(cache=True)
def njit_calc_nan(n: int, missing_five: np.ndarray, missing: np.ndarray, data_length: int) -> set:
    """Determines which indices should be set to NaN.

    Starting from every index in missing_five, counts up to n. If any indices are missing while
    counting, restart the count. Every index from the missing_five index to the end of the count
    is included in the output. The general idea for indicators like the EMA is to begin
    calculating only when the previous n timestamps are present. If 5 timestamps in a row are
    missing, then the calculation resets as if that index is the first index of the dataset.
    Placed outside of class to be accessible to instance methods.

    Args:
        n: Value to count up to.
        missing_five: Indices where at least 5 of the previous timestamps are missing.
        missing: Indices where the previous timestamp is missing.
        data_length: Length of dataset.

    Returns:
        Set of all NaN indices.
    """
    missing = set(missing)
    nan_indices = set()
    for index in missing_five:
        done = False
        while not done:
            for n_next in np.arange(index, index + n):
                if n_next == data_length:
                    done = True
                    break
                nan_indices.add(n_next)
                if n_next in missing:
                    index = n_next + 1
                    break
            if n_next not in missing:
                done = True
    return nan_indices


@njit(cache=True)
def get_missing(integer_time: np.ndarray, active_times: np.ndarray, open_boolean: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Determines missing indices.

    Args:
        active_times: Contains all timestamps present across all symbols (no duplicates).

    Returns:
        A tuple containing a Numpy array of booleans indicating whether this bar follows a
        missing bar (bars during which the market is closed are excluded), a similar array which
        isn't exclusive to market-open bars, and another array of booleans indicating whether
        this bar follows 5 missing bars in a row. For the last two arrays, values for bars
        during which the market is closed are always set to False. Placed outside of class to be
        accessible to instance methods.
    """
    symbol_indices = np.searchsorted(active_times, integer_time)
    shifted_symbol_indices = np.empty_like(symbol_indices)
    shifted_open_boolean = np.empty_like(open_boolean)
    shifted_symbol_indices[0] = symbol_indices[0]
    shifted_open_boolean[0] = open_boolean[0]
    shifted_symbol_indices[1:] = symbol_indices[:-1]
    shifted_open_boolean[1:] = open_boolean[:-1]
    missing = np.where((symbol_indices >= shifted_symbol_indices + 2) & (
            (open_boolean == True) & (shifted_open_boolean == True)), True, False)
    missing_five = np.where((symbol_indices >= shifted_symbol_indices + 5) & (
            (open_boolean == True) & (shifted_open_boolean == True)), True, False)
    return np.delete(missing, np.flatnonzero(np.invert(open_boolean))), np.flatnonzero(
        missing), np.flatnonzero(missing_five)


class IndicatorCalculator:
    """Contains functions for calculating indicators for one symbol.

    Attributes:
        symbol: A string of the symbol name.
        prev_month_indicators: A dictionary containing all important information from the
        previous month (to maintain indicator continuity).
        open_times: A dictionary containing all timestamps during which the market is open.
        timeframe: A string containing an integer followed by min, e.g., '5min'.
        o: A Numpy array containing bar open prices.
        c: A Numpy array containing bar close prices.
        h: A Numpy array containing bar high prices.
        l: A Numpy array containing bar low prices.
        v: A Numpy array containing bar volume.
        timestamp: A list containing bar timestamps.
        integer_time: A Numpy array that contains an integer representation of the time (
        YYYYMMDDHHmm)
        hour_min: A Numpy array containing only the hour and minute (HH:mm) of each timestamp.
        open_boolean: A Numpy array of booleans indicating whether an index corresponds to a bar
        when the market is open.
        open_o: A Numpy array containing bar open prices only for bars during which the market
        is open.
        open_c: A Numpy array containing bar close prices only for bars during which the market
        is open.
        open_indices: A Numpy array containing the array indices of bars during which the market
        is open.
        open_missing: A Numpy array of booleans indicating whether this bar follows a missing
        bar (bars during which the market is closed are excluded).
        missing: A Numpy array of booleans indicating whether this bar follows a missing bar (
        bars during which the market is closed are set to False).
        missing_five: A Numpy array of booleans indicating whether this bar follows 5 missing
        bars in a row (bars during which the market is closed are set to False).
    """

    def __init__(self, data: list):
        """Initializes IndicatorCalculator. Performs calculations to determine several instance
        variables."""
        self.symbol = data[0]
        month_data = data[1]
        self.prev_month_indicators = data[2]
        active_times = data[3]
        self.open_times = data[4]
        self.timeframe = data[5]
        self.o = month_data['open']
        self.c = month_data['close']
        self.h = month_data['high']
        self.l = month_data['low']
        self.v = month_data['volume']
        self.timestamp = month_data['time']
        self.integer_time = month_data['integer date'] * 10000 + month_data['hour'] * 100 + \
                            month_data['min']
        self.hour_min = (month_data['hour'] * 100 + month_data['min']).astype(np.int32)
        self.open_boolean = np.where((930 <= self.hour_min) & (self.hour_min < 1600), True, False)
        self.open_o = np.delete(self.o, np.invert(self.open_boolean))
        self.open_c = np.delete(self.c, np.invert(self.open_boolean))
        self.open_indices = np.flatnonzero(self.open_boolean)
        self.open_missing, self.missing, self.missing_five = get_missing(self.integer_time,
                                                                         active_times,
                                                                         self.open_boolean)

    def calc_ema(self, n: int) -> np.ndarray:
        """Calculates the n-period EMA.

        Calls the njit_ema function for better performances.

        Args:
            n: EMA period.

        Returns:
            An array containing the EMA.
        """
        return self.njit_ema(n, self.c, self.prev_month_indicators['{} EMA'.format(n)],
                             self.missing_five, self.missing).astype(np.float32)

    @staticmethod
    @njit(cache=True)
    def njit_ema(n: int, c: np.ndarray, prev_ema: np.float64, missing_five: np.ndarray,
                 missing: np.ndarray) -> np.ndarray:
        """Calculates the n-period EMA.

        For the first month, sets average of the first n values as the first EMA value. For
        proceeding months, converts the EMA using the last EMA of the previous month.
        If 5 or more bars in a row are missing, waits until n bars in a row are present and sets
        the next EMA as the average of these bars (the EMA of those bars are set to NaN).

        Args:
            n: EMA period.
            c: Close prices.
            prev_ema: Last EMA value of the previous month.
            missing_five: An array of booleans indicating whether this bar follows 5 missing
            bars in a row.
            missing: An array of booleans indicating whether this bar follows a missing bar.
            num_bits: Size of floats in output.

        Returns:
            An array containing the EMA.
        """
        k = 2 / (n + 1)
        ema = np.empty_like(c)
        ema.fill(np.nan)
        if c.size < n:
            return ema
        nan_indices = njit_calc_nan(n, missing_five, missing, c.size)
        if np.isnan(prev_ema):
            ema[n - 1] = c[:n].mean()
            start = n
        else:
            ema[0] = (c[0] - prev_ema) * k + prev_ema
            start = 1
        for i in range(start, c.size):
            if i not in nan_indices:
                if np.isnan(ema[i - 1]):
                    ema[i] = c[i - (n - 1): i + 1].mean()
                else:
                    ema[i] = (c[i] - ema[i - 1]) * k + ema[i - 1]
        return ema

    @staticmethod
    @njit(cache=True)
    def njit_sma(n: int, c: np.ndarray, prev_vals: np.ndarray, missing_five: np.ndarray,
                 missing: np.ndarray) -> np.ndarray:
        """Calculates the n-period SMA.

        Args:
            n: SMA period.
            c: Close prices.
            prev_vals: Array containing last n closes of previous month.
            missing_five: An array of booleans indicating whether this bar follows 5 missing
            bars in a row.
            missing: An array of booleans indicating whether this bar follows a missing bar.

        Returns:
            An array containing SMA.
        """
        if prev_vals.size > 0:  # When data from previous month is provided.
            combined = np.empty(prev_vals.size + c.size)
            combined[:prev_vals.size] = prev_vals
            combined[prev_vals.size:] = c
        else:
            combined = c
        combined = np.cumsum(combined)
        combined = (combined[n:] - combined[:-n]) / n
        if prev_vals.size > n:
            combined = combined[prev_vals.size - n:]
        elif prev_vals.size < n:
            start_nan = np.empty(n - prev_vals.size)
            start_nan.fill(np.nan)
            combined = np.append(start_nan, combined)
        nan_indices = njit_calc_nan(n, missing_five, missing, c.size)
        if len(nan_indices) != 0:
            nan_indices = np.array(list(nan_indices))
            combined[nan_indices] = np.nan
        return combined  # Returns as 64-bit floats (since only called by other functions)

    def calc_rsi(self) -> np.ndarray:
        """Calculates the RSI

        Returns:
            Array containing the RSI.
        """
        n = 14
        if self.prev_month_indicators[
            '14 SMA'].size != 0:  # When data from previous month is provided.
            c = np.empty(self.prev_month_indicators['14 SMA'].size + self.c.size)
            c[:self.prev_month_indicators['14 SMA'].size] = self.prev_month_indicators['14 SMA']
            c[self.prev_month_indicators['14 SMA'].size:] = self.c
        else:
            c = self.c
        right_shifted_c = np.empty_like(c)
        right_shifted_c[0] = c[0]
        right_shifted_c[1:] = c[:-1]
        gain = np.where(c - right_shifted_c > 0, c - right_shifted_c, 0)
        loss = np.where(c - right_shifted_c < 0, -c + right_shifted_c, 0)
        shifted_missing_five = self.missing_five + self.prev_month_indicators['14 SMA'].size
        shifted_missing = self.missing + self.prev_month_indicators['14 SMA'].size
        avg_loss = self.njit_sma(n, loss, np.array([]), shifted_missing_five, shifted_missing)[
                   self.prev_month_indicators['14 SMA'].size:]
        avg_gain = self.njit_sma(n, gain, np.array([]), shifted_missing_five, shifted_missing)[
                   self.prev_month_indicators['14 SMA'].size:]
        rs = avg_gain / np.maximum(avg_loss, 0.001)
        rsi = 100 - 100 / (1 + rs)
        return rsi.astype(np.float32)

    def calc_macd(self) -> Tuple[np.ndarray, np.ndarray, np.float32, np.float32]:
        """Calculates the MACD and signal.

        Also returns the last values of the 26 and 12 EMAs for MACD continuity when calculating
        the next month.

        Returns:
            Numpy arrays containing the MACD and signal, and the last values of the 26 and 12
            EMAs respectively.
        """
        ema_26 = self.njit_ema(26, self.c, self.prev_month_indicators['26 EMA'], self.missing_five,
                               self.missing)
        ema_12 = self.njit_ema(12, self.c, self.prev_month_indicators['26 EMA'], self.missing_five,
                               self.missing)
        macd = ema_12 - ema_26
        signal = self.njit_ema(9, macd, self.prev_month_indicators['signal'], self.missing_five,
                               self.missing)
        return macd.astype(np.float32), signal.astype(np.float32), ema_26[-1], ema_12[-1]

    def calc_vwap(self) -> np.ndarray:
        """Calculates the VWAP.

        Calls the njit_vwap function for better performance. The VWAP is calculated only for
        days where the first three bars of open-hours exist. Otherwise the values are set to NaN.
        The VWAP of bars corresponding to times when the market is closed are set to NaN.
        Assumes the timeframe is less than 15 minutes.

        Returns:
            Array containing the VWAP
        """
        return self.njit_vwap(int(self.timeframe[:-3]), self.hour_min, self.open_indices, self.c,
                              self.h, self.l, self.v)

    @staticmethod
    @njit(cache=True)
    def njit_vwap(timeframe: int, hour_min: np.ndarray, open_indices: np.ndarray, c: np.ndarray,
                  h: np.ndarray, l: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Calculates the VWAP.

        Args:
            open_boolean: An array of booleans indicating whether an index corresponds to a bar
            when the market is open.
            valid_open_candle: An array of booleans indicating whether an index corresponds to a
            bar at 9:30 and the following two bars are present.
            c: Bar close prices.
            h: Bar high prices.
            l: Bar low prices.
            v: Bar volume.

        Returns:
            Array containing the VWAP
        """
        shifted_hour_min = np.empty_like(hour_min)
        twice_shifted_hour_min = np.empty_like(hour_min)
        shifted_hour_min[-1] = hour_min[-1]
        twice_shifted_hour_min[-1] = shifted_hour_min[-1]
        shifted_hour_min[:-1] = hour_min[1:]
        twice_shifted_hour_min[:-1] = shifted_hour_min[1:]
        valid_open_candle = np.where((930 == hour_min) & (930 + timeframe == shifted_hour_min) & (
                930 + (2 * timeframe) == twice_shifted_hour_min), True, False)
        vwap = np.empty_like(c)
        cum_vol = np.empty_like(c)
        vwap.fill(np.nan)
        candle_avg = (h + l + c) / 3
        v = np.maximum(v, 1)  # To prevent division by zero with bad data.
        for i in open_indices:
            if valid_open_candle[i]:
                cum_vol[i] = v[i]
                vwap[i] = candle_avg[i]
            elif not np.isnan(vwap[i - 1]):
                cum_vol[i] = cum_vol[i - 1] + v[i]
                vwap[i] = (cum_vol[i - 1] * vwap[i - 1] + v[i] * candle_avg[i]) / (cum_vol[i])
        return vwap.astype(np.float32)

    def calc_atr(self):
        """Calculates the ATR.

        Calls the njit_atr functions for better performance.

        Returns:
            Array containing the ATR.
        """
        return self.njit_atr(self.c, self.h, self.l, self.prev_month_indicators['ATR'],
                             self.prev_month_indicators['Last Close'], self.missing_five,
                             self.missing)

    @staticmethod
    @njit(cache=True)
    def njit_atr(c: np.ndarray, h: np.ndarray, l: np.ndarray, prev_atr: np.float32,
                 prev_last_close: np.float32, missing_five: np.ndarray,
                 missing: np.ndarray) -> np.ndarray:
        """Calculates the ATR

        After calculating the ATR for all bars, the values for the ones following 5 missing bars
        in a row are set to NaN. These indices are found using the njit_calc_nan function.

        Args:
            c: Bar close prices.
            h: Bar high prices.
            l: Bar low prices.
            prev_atr: The last ATR value of the previous month.
            prev_last_close: The last close price of the previous month.
            missing_five: An array of booleans indicating whether this bar follows 5 missing
            bars in a row.
            missing: An array of booleans indicating whether this bar follows a missing bar.

        Returns:
            Array containing the ATR
        """
        n = 14
        right_shifted_c = np.empty_like(c)
        right_shifted_c[0] = prev_last_close
        right_shifted_c[1:] = c[:-1]
        high_minus_prev_close = np.absolute(h - right_shifted_c)
        low_minus_prev_close = np.absolute(l - right_shifted_c)
        atr = np.empty_like(c)
        atr.fill(np.nan)
        tr = np.maximum(h - l, high_minus_prev_close, low_minus_prev_close)
        if np.isnan(prev_atr) or np.isnan(prev_last_close):
            start = n + 1
            atr[n] = tr[1:n + 1].mean()
        else:
            start = 1
            atr[0] = (prev_atr + tr[0]) / n
        for i in range(start, c.size):
            atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
        nan_indices = njit_calc_nan(n, missing_five, missing, c.size)
        if len(nan_indices) != 0:
            nan_indices = np.array(list(nan_indices))
            atr[nan_indices] = np.nan
        return atr.astype(np.float32)

    def calc_bollinger(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the Bollinger Bands.

        Uses njit_bollinger_bounds for better performance. prev_month_indicators['20 SMA']
        contains the last 20 close prices of the previous month.

        Returns:
            Arrays containing the upper and lower Bollinger Bands.
        """
        n = 20
        if self.prev_month_indicators['20 SMA'].size == 0:
            sma_20 = self.njit_sma(n, self.c, self.prev_month_indicators['20 SMA'],
                                   self.missing_five, self.missing)
            return self.njit_bollinger(self.c, sma_20)
        else:
            extended_close = np.empty(n + self.c.size)
            extended_close[:n] = self.prev_month_indicators['20 SMA'][-n:]
            extended_close[n:] = self.c
            sma_20 = self.njit_sma(n, extended_close, self.prev_month_indicators['20 SMA'][-n:],
                                   self.missing_five + n,
                                   self.missing + n)  # extended 20 candles in the past
            return self.njit_bollinger_extended(extended_close, sma_20)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def njit_bollinger(c, sma_20):
        """ Calculate the Bollinger Bands when data from the previous months is not included.

        Args:
            sma_20: The 20 SMA.

        Returns:
            A tuple of arrays containing the upper and lower Bollinger Bands respectively.
        """
        n = 20
        rolling_window = np.lib.stride_tricks.as_strided(c, shape=(c.size - n + 1, n),
                                                         strides=(c.strides[0], c.strides[0]))
        stdev = np.empty(len(rolling_window))
        root = np.empty(n)
        for i in range(len(rolling_window)):
            mean = np.sum(rolling_window[i]) / len(rolling_window[i])
            for j in range(n):
                root[j] = (rolling_window[i][j] - mean) ** 2
            stdev[i] = np.sqrt(np.sum(root) / n)
        stdev = np.append(np.array([np.nan] * (n - 1)), stdev)
        BB_plus = sma_20 + stdev * 2
        BB_minus = sma_20 - stdev * 2
        return BB_plus.astype(np.float32), BB_minus.astype(np.float32)

    @staticmethod
    # @njit(cache=True, fastmath=True)
    def njit_bollinger_extended(c, sma_20):
        """ Calculate the Bollinger Bands when data from the previous months is not included.

        Args:
            sma_20: The 20 SMA.

        Returns:
            A tuple of arrays containing the upper and lower Bollinger Bands respectively.
        """
        n = 20
        rolling_window = np.lib.stride_tricks.as_strided(c, shape=(c.size - n + 1, n),
                                                         strides=(c.strides[0], c.strides[0]))
        stdev = np.empty(len(rolling_window))
        root = np.empty(n)
        for i in range(len(rolling_window)):
            mean = np.sum(rolling_window[i]) / len(rolling_window[i])
            for j in range(n):
                root[j] = (rolling_window[i][j] - mean) ** 2
            stdev[i] = np.sqrt(np.sum(root) / n)
        stdev = np.append(np.array([np.nan] * (n - 1)), stdev)
        BB_plus = sma_20 + stdev * 2
        BB_minus = sma_20 - stdev * 2
        return BB_plus.astype(np.float32)[n:], BB_minus.astype(np.float32)[n:]

    def calc_consecutive(self) -> np.ndarray:
        """Calculates the number of consecutive bullish/bearish candles at each index.

        Calls njit_consecutive for better performance. The output value is set to 1 at the index
        of the first bullish candle. Each proceeding bullish candle increments this value (at
        its appropriate index) until a bearish or doji is reached; which resets the counter.
        Consecutive bearish candles correspond to negative values. The value is set to zero when
        the candle is a doji. The value is set to zero for all overnight/premarket candles.
        When the bar follows a missing candle, the value is set to 0 regardless of the candle
        itself. For example, the output for this sample [bearish, bearish, doji, bullish,
        bearish, bullish, bullish, bullish] would be [-1, -2, 0, 1, -1, 1, 2, 3]

        Returns:
            Array containing consecutive-candle data (stored as 16-bit integers).
        """
        return self.njit_consecutive(self.open_c, self.open_o, self.open_missing,
                                     self.open_indices, self.c.size)

    @staticmethod
    @njit(cache=True)
    def njit_consecutive(open_c: np.ndarray, open_o: np.ndarray, open_missing: np.ndarray,
                         open_indices: np.ndarray, total_length: int) -> np.ndarray:
        """Calculates the number of consecutive bullish/bearish candles at each index.

        Args:
            open_c: Bar close prices only for bars during which the market is open.
            open_o: Bar open prices only for bars during which the market is open.
            open_missing: Boolean array indicating if a bar follows a missing bar; only for bars
            during which the market is open.
            open_indices: Indices during which the market is open.
            total_length: Length of the total dataset (includes bars corresponding to when the
            market is closed)

        Returns:
            Array containing consecutive-candle data (stored as 16-bit integers).
        """
        g_indices = np.flatnonzero(np.where((open_c >= open_o) | (open_missing == True), True,
                                            False))  # both are set to true on doji's and after
        # a missing candle; cancelling out and resulting in zero
        r_indices = np.flatnonzero(
            np.where((open_c <= open_o) | (open_missing == True), True, False))
        green_indices = np.zeros(open_c.size, dtype=np.int16)
        red_indices = np.zeros(open_c.size, dtype=np.int16)
        green_indices[g_indices] = g_indices
        red_indices[r_indices] = r_indices
        green_max = np.zeros(open_c.size, dtype=np.int16)
        red_max = np.zeros(open_c.size, dtype=np.int16)
        for i in range(1, open_c.size):
            green_max[i] = max(green_max[i - 1], green_indices[i])
            red_max[i] = max(red_max[i - 1], red_indices[i])
        open_consecutives = green_max - red_max
        all_consecutives = np.zeros(total_length, dtype=np.int16)
        all_consecutives[open_indices] = open_consecutives
        return all_consecutives

    def candlestick_analysis(self) -> Dict[str, np.ndarray]:
        """Determines whether certain candlestick patterns are found at each index.

        Better performance achieved without using Numba.

        Returns:
            Dictionary with each key being a candlestick pattern and the values as boolean arrays.
        """
        # candlestick_data = self.njit_candlestick(self.o, self.c, self.h, self.l)
        c = self.c.astype(np.float32)  # Works on 32-bit floats, faster than 64-bit ones
        o = self.o.astype(np.float32)
        h = self.h.astype(np.float32)
        l = self.l.astype(np.float32)
        body = np.abs(c - o)
        lower_wick = np.minimum(c, o) - l
        upper_wick = h - np.maximum(c, o)
        # Single candlestick patterns
        hammer = (lower_wick > body * 1.5) & (body > upper_wick * 2)
        shooting_star = (upper_wick > body * 1.5) & (body > lower_wick * 2)
        spinning_top = (upper_wick > body * 2) & (lower_wick > body * 2) & (body > 0)
        green_marubozu = (o == l) & (c == h) & (o != c)
        red_marubozu = (o == h) & (c == l) & (o != c)
        # Multi-candlestick patterns
        prev_o = np.empty_like(c)
        prev_c = np.empty_like(c)
        prev_h = np.empty_like(c)
        prev_l = np.empty_like(c)
        prev_upper_wick = np.empty_like(c)
        prev_lower_wick = np.empty_like(c)
        prev_body = np.empty_like(c)
        two_o = np.empty_like(c)
        two_c = np.empty_like(c)
        two_h = np.empty_like(c)
        two_l = np.empty_like(c)
        prev_o[0] = o[0]
        prev_c[0] = c[0]
        prev_h[0] = h[0]
        prev_l[0] = l[0]
        prev_upper_wick[0] = upper_wick[0]
        prev_lower_wick[0] = lower_wick[0]
        prev_body[0] = body[0]
        two_o[0] = o[0]
        two_c[0] = c[0]
        two_h[0] = h[0]
        two_l[0] = l[0]
        prev_o[1:] = o[:-1]
        prev_c[1:] = c[:-1]
        prev_h[1:] = h[:-1]
        prev_l[1:] = l[:-1]
        prev_upper_wick[1:] = upper_wick[:-1]
        prev_lower_wick[1:] = lower_wick[:-1]
        prev_body[1:] = body[:-1]
        two_o[1:] = prev_o[:-1]
        two_c[1:] = prev_c[:-1]
        two_h[1:] = prev_h[:-1]
        two_l[1:] = prev_l[:-1]
        bullish_engulfing = (prev_o > prev_c) & (c > o) & (o < prev_l) & (c > prev_h)
        bearish_engulfing = (prev_o < prev_c) & (c < o) & (o > prev_h) & (c < prev_l)
        evening_star = (two_c > two_o) & (prev_o > two_h) & (prev_l > two_o) & (
                prev_o > prev_c) & (o < prev_l) & (c < o) & (o < two_l)
        morning_star = (two_c < two_o) & (prev_o <= two_l) & (prev_h <= two_o) & (
                np.maximum(prev_upper_wick, prev_lower_wick) > prev_body) & (c > o) & (
                               l > prev_o) & (h == two_h)
        tweezer_top = (prev_l > two_l) & (prev_h == h) & (prev_o < prev_c) & (o > c)
        tweezer_bottom = (prev_h < two_h) & (prev_l == l) & (prev_c < prev_o) & (o < c)
        return {
            'Hammer': hammer,
            'Shooting Star': shooting_star,
            'Spinning Top': spinning_top,
            'Green Marubozu': green_marubozu,
            'Red Marubozu': red_marubozu,
            'Bullish Engulfing': bullish_engulfing,
            'Bearish Engulfing': bearish_engulfing,
            'Evening Star': evening_star,
            'Morning Star': morning_star,
            'Tweezer Top': tweezer_top,
            'Tweezer Bottom': tweezer_bottom
        }


def indicator_process(input: List) -> List[Union[str, Dict[str, Union[np.ndarray, np.float32]]]]:
    """Calculates all indicators for one symbol.

    Creates an instance of IndicatorCalculator class and calls the object's methods to calculate
    each indicator.

    Args:
        input: All required data regarding the symbol (passed in from the main process ).

    Returns:
        A list with the 1st element being the symbol name and 2nd element being a dictionary
        containing all data regarding each indicator, i.e. {indicator name: indicator data}.
    """

    def get_false_candlestick_arrays(num_candles: int) -> Dict[str, np.ndarray]:
        """Returns a dictionary with all candlestick patterns set to False

        Args:
            num_candles: Length of dataset.

        Returns:
            Dictionary with keys being the pattern name and all values being [np.nan] *
            num_candles.
        """
        false_array = np.array([False] * num_candles, dtype=np.bool)
        return {
            'Hammer': false_array,
            'Shooting Star': false_array,
            'Spinning Top': false_array,
            'Green Marubozu': false_array,
            'Red Marubozu': false_array,
            'Bullish Engulfing': false_array,
            'Bearish Engulfing': false_array,
            'Evening Star': false_array,
            'Morning Star': false_array,
            'Tweezer Top': false_array,
            'Tweezer Bottom': false_array
        }

    def empty_data_dict(symbol: str, num_candles: int, time_tuple: Tuple[int, int, int]) -> List[
        Union[str, Dict[str, Union[np.float32, np.ndarray]]]]:
        """Creates a dictionary with all indicator data set to NaN or an array of NaN's.

        Args:
            symbol: Name of symbol.
            num_candles: Length of dataset.

        Returns:
            List with the 1st element being the symbol name and 2nd element being a dictionary
            with the indicator name as keys and NaN or NaN arrays as values.
        """
        empty_32bit_array = np.array([np.nan] * num_candles, dtype=np.float32)
        return [symbol, {
            'integer time': time_tuple[0] * 10000 + time_tuple[1] * 100 + time_tuple[2],
            '14 SMA': np.array([]),
            '20 SMA': np.array([]),
            '9 EMA': np.copy(empty_32bit_array),
            '12 EMA': np.nan,
            '26 EMA': np.nan,
            '20 EMA': np.copy(empty_32bit_array),
            '50 EMA': np.copy(empty_32bit_array),
            '200 EMA': np.copy(empty_32bit_array),
            'RSI': np.copy(empty_32bit_array),
            'MACD': np.copy(empty_32bit_array),
            'signal': np.copy(empty_32bit_array),
            'VWAP': np.copy(empty_32bit_array),
            'ATR': np.copy(empty_32bit_array),
            'Bollinger (Upper)': np.copy(empty_32bit_array),
            'Bollinger (Lower)': np.copy(empty_32bit_array),
            'Last Close': np.nan,
            'Consecutives': np.array([0] * num_candles, dtype=np.uint16),
            **get_false_candlestick_arrays(num_candles)
        }]

    def time_function(n, func, arg=None):
        """For testing the execution time of each function.

        Performs 10 trials of each test. Prints time for each trial, average trial time and
        standard deviation.

        Args:
            n: Number of iterations.
            Func: Name of function.
            arg: Argument for function (applies only for calc_EMA)
        """
        run_times = []
        for run in range(10):
            start = time.perf_counter()
            if arg is not None:
                for i in range(n):
                    func(arg)
                end = time.perf_counter()
            else:
                for i in range(n):
                    func()
                end = time.perf_counter()
            run_times.append(end - start)
            print('{} = {}s for {} runs'.format(func.__name__, end - start, n))
        print(np.mean(run_times))
        print(np.std(run_times))

    symbol = input[0]
    num_candles = input[1]['close'].size
    if num_candles < 40:  # No data for this symbol
        return empty_data_dict(symbol, num_candles,
                               (input[1]['integer date'], input[1]['hour'], input[1]['min']))
    indicator_calculator = IndicatorCalculator(input)
    ema_9 = indicator_calculator.calc_ema(9)
    ema_20 = indicator_calculator.calc_ema(20)
    ema_50 = indicator_calculator.calc_ema(50)
    ema_200 = indicator_calculator.calc_ema(200)
    rsi = indicator_calculator.calc_rsi()
    macd, signal, last_ema_26, last_ema_12 = indicator_calculator.calc_macd()
    bb_plus, bb_minus = indicator_calculator.calc_bollinger()
    vwap = indicator_calculator.calc_vwap()
    atr = indicator_calculator.calc_atr()
    consecutives = indicator_calculator.calc_consecutive()
    candlestick_data = indicator_calculator.candlestick_analysis()
    c = input[1]['close']  # Used to calculate indicators for next month.
    h = input[1]['high']

    # Code to test execution times:
    # time_function(10000, indicator_calculator.calc_rsi)
    # time_function(10000, indicator_calculator.calc_ema, 9)
    # time_function(10000, indicator_calculator.calc_ema, 20)
    # time_function(10000, indicator_calculator.calc_ema, 50)
    # time_function(10000, indicator_calculator.calc_ema, 200)
    # time_function(10000, indicator_calculator.calc_macd)
    # time_function(10000, indicator_calculator.calc_vwap)
    # time_function(10000, indicator_calculator.calc_atr)
    # time_function(10000, indicator_calculator.calc_bollinger)
    # time_function(10000, indicator_calculator.calc_consecutive)
    # time_function(10000, indicator_calculator.candlestick_analysis)

    return [symbol, {
        'integer time': indicator_calculator.integer_time,
        '14 SMA': c[-14:],
        '20 SMA': c[-20:],  # 40 to be able to calculate bollinger bands
        '9 EMA': ema_9,
        '12 EMA': last_ema_12,
        '26 EMA': last_ema_26,
        '20 EMA': ema_20,
        '50 EMA': ema_50,
        '200 EMA': ema_200,
        'RSI': rsi,
        'MACD': macd,
        'signal': signal,
        'VWAP': vwap,
        'ATR': atr,
        'Last Close': c[-1],
        'Bollinger (Upper)': bb_plus,
        'Bollinger (Lower)': bb_minus,
        'Consecutives': consecutives,
        **candlestick_data
    }]


def calculate_indicators(month_data: Dict[str, Dict[str, Union[List[str], np.ndarray]]],
                         prev_month_indicators: Dict[
                             str, Dict[str, Union[np.float32, np.ndarray]]], timeframe: str,
                         p: Type[multiprocessing.Pool]) -> Dict[
    str, Dict[str, Union[np.float32, np.ndarray]]]:
    """Calculates indicators for every symbol.

    Coordinates the calculation of indicators using a pool of worker processes. Each task
    consists of calculating the indicators for one symbol; and is assigned using the Pool.map()
    function. Saves the indicator data in the month_data dictionary.

    Args:
        month_data: Contains timestamp, open, close, high, low, volume, for each bar for every
        symbol.
        prev_month_indicators: Contains some indicators from previous month that are used to
        maintain indicator continuity.
        timeframe: A string containing an integer followed by min, e.g., '5min'.
        p: A pool of processes.

    Returns:
        An updated version of prev_month_indicators.
    """

    def get_candle_times(
            month_data: Dict[str, Dict[str, Union[List[str], np.ndarray]]]) -> np.ndarray:
        """Gets all timestamps during the month and sorts in ascending order. Deletes duplicates.

        Args:
            month_data: Contains timestamp, open, close, high, low, volume, for each bar for
            every symbol.

        Returns:
            An array of all the timestamps.
        """
        times = {time for symbol in month_data.keys() for time in (
                month_data[symbol]['integer date'] * 10000 + month_data[symbol]['hour'] * 100 +
                month_data[symbol]['min'])}
        return np.asarray(sorted(list(times)))

    def get_open_times(timestamp: np.ndarray) -> Set[str]:
        """Returns a set of all timestamps during which the market is open.

        Args:
            timestamp: Array of all timestamps in order.

        Returns:
            Result as set.
        """
        timestamp = np.array(timestamp, dtype='|S16')
        hour_min = timestamp.view('|S1').reshape(-1, timestamp.dtype.itemsize)[:, 11:].reshape(
            -1).view('|S5').astype(str)
        closed_boolean = np.where((np.array('09:30') <= hour_min) & (hour_min < np.array('16:00')),
                                  False, True)
        open_times = np.delete(timestamp, closed_boolean)
        return set(open_times)

    def prepare_pool(month_data: Dict[str, Dict[str, Union[list, np.ndarray]]], timeframe: str) \
            -> Tuple[List[np.ndarray], List[Set[str]], List[str]]:
        """Prepares all inputs for Pool.map() function.

        Produces ordered list of all unique timestamps and set of timestamps corresponding to
        when the market is open. Multiples each of the aforementioned items and the timeframe
        by the number of symbols (using list multiplication).

        Args:
            month_data: Contains timestamp, open, close, high, low, volume, for each bar for
            every symbol.
            timeframe: A string containing an integer followed by min, e.g., '5min'.

        Returns:
            A tuple containing the three outputs.
        """
        active_times = get_candle_times(month_data)
        open_times = [get_open_times(active_times)] * len(month_data.keys())
        active_times = [active_times] * len(month_data.keys())
        timeframe = [timeframe] * len(month_data.keys())
        return active_times, open_times, timeframe

    def update_prev_month_indicators(
            prev_month_indicators: Dict[str, Dict[str, Union[np.float32, np.ndarray]]],
            indicator_data: List[Union[str, Dict[str, Union[np.ndarray, np.float32]]]]):
        """Updates prev_month_indicators using newly calculated indicator data.

        If the 9 EMA is an empty list, then set all indicators to NaN or a list of NaN's.

        Args:
            prev_month_indicators: Contains some indicators from previous month.
            indicator_data: A list with the 1st element being the symbol name and 2nd element
            being a dictionary containing all data regarding each indicator, i.e. {indicator
            name: indicator data}.
        """
        for entry in indicator_data:
            symbol = entry[0]
            values = entry[1]
            if not values['9 EMA'].size:  # If indicators were not calculated
                prev_month_indicators.update({
                    symbol: {
                        '14 SMA': np.array([]),
                        '20 SMA': np.array([]),
                        '9 EMA': np.nan,
                        '12 EMA': np.nan,
                        '26 EMA': np.nan,
                        '20 EMA': np.nan,
                        '50 EMA': np.nan,
                        '200 EMA': np.nan,
                        'RSI': np.nan,
                        'MACD': np.nan,
                        'signal': np.nan,
                        'ATR': np.nan,
                        'Last Close': np.nan
                    }
                })
            else:
                prev_month_indicators.update({
                    symbol: {
                        '14 SMA': values['14 SMA'],
                        '20 SMA': values['20 SMA'],
                        '9 EMA': values['9 EMA'][-1],
                        '12 EMA': values['12 EMA'],
                        '26 EMA': values['26 EMA'],
                        '20 EMA': values['20 EMA'][-1],
                        '50 EMA': values['50 EMA'][-1],
                        '200 EMA': values['200 EMA'][-1],
                        'RSI': values['RSI'][-1],
                        'MACD': values['MACD'][-1],
                        'signal': values['signal'][-1],
                        'ATR': values['ATR'][-1],
                        'Last Close': values['Last Close']
                    }
                })
            del values[
                '14 SMA']  # Deletes intermediate indicators that should not appear in dataset
            del values['20 SMA']
            del values['12 EMA']
            del values['26 EMA']
            del values['Last Close']

    def append_indicators(month_data: Dict[str, Dict[str, Union[List[str], np.ndarray]]],
                          indicator_data: List[
                              Union[str, Dict[str, Union[np.ndarray, np.float32]]]]):
        """Adds new indicator data to dictionary containing bar data.

        Args:
            month_data: Contains timestamp, open, close, high, low, volume, for each bar for
            every symbol.
            indicator_data: A list with the 1st element being the symbol name and 2nd element
            being a dictionary containing all data regarding each indicator, i.e. {indicator
            name: indicator data}.
        """
        for entry in indicator_data:
            symbol = entry[0]
            if entry[1]:
                month_data[symbol].update(entry[1])

    prev_month_indicators = {symbol: prev_month_indicators[symbol] for symbol in
                             month_data.keys()}  # Sort based on ordering of new monthly data
    active_times, open_times, timeframe = prepare_pool(month_data, timeframe)
    start = time.perf_counter()
    indicator_data = p.map(indicator_process, zip(month_data.keys(), month_data.values(),
                                                  prev_month_indicators.values(), active_times,
                                                  open_times, timeframe))
    end = time.perf_counter()
    print('Took {}s to calculate indicators for all symbols'.format(end - start))
    update_prev_month_indicators(prev_month_indicators, indicator_data)
    append_indicators(month_data, indicator_data)
    return prev_month_indicators


def save_month(month_data: Dict[str, Dict[str, Union[List[str], np.ndarray]]], month: str,
               timeframe: str):
    """Saves data to Feather file.

    Creates a file containing bar (timestamp, open, etc) and indicator data (9 EMA, RSI,
    etc). The header contains the data category (e.g. close, RSI) followed by rows; each of
    which corresponding to a single bar of a symbol. The symbol ordering is random and the
    symbol is not labelled. This information is found in the second file created, which consists
    of three columns: symbol, start, end. Start and end refer to the indices of the first file
    at which data for a particular symbol starts and ends (exclusively) respectively.

    Args:
        month_data: Contains timestamp, open, close, high, low, volume, and indicator data for
        each bar for every symbol.
        month: A string representation of the month (YYYY-mm-dd)
        timeframe: A string containing an integer followed by min, e.g., '5min'.
    """

    def calc_file_length(month_data_values: List[Dict[str, Union[List[str], np.ndarray]]]):
        """Calculates total number of bars across all symbols.

        Args:
            month_data_values: List with each element a dictionary containing data for one
            symbol, with key-value pairs of {data category: data}.

        Returns:
            Total number of bars.
        """
        file_length = 0
        for symbol_data in month_data_values:
            file_length += symbol_data['close'].size
        return file_length

    def convert_time_list_to_array(
            month_data_values: List[Dict[str, Union[List[str], np.ndarray]]]):
        """Replaces timestamp list with Numpy array in data dictionary.

        Args:
            month_data_values: List with each element a dictionary containing data for one
            symbol, with key-value pairs of {data category: data}.
        """
        for symbol_data in month_data_values:
            symbol_data.update({'time': np.array(symbol_data['time'], dtype='<U16')})

    def combine_data(df_headers: List[str], dtypes: List[np.dtype],
                     month_data_values: List[Dict[str, Union[List[str], np.ndarray]]]) -> Tuple[
        Dict[str, np.ndarray], Tuple[List[int], List[int]]]:
        """Concatenates like-arrays of all symbols into large array; computes start and end
        indices for all symbols.

        e.g. RSI arrays for all symbols are combined to one array. If 'MSFT' begins at index 100
        and ends at 200. The corresponding element in symbol_indices = ([100], [201])

        Args:
            df_headers: A list of the headers to be used in the Feather file.
            dtypes: The Numpy dtypes of each array (column) under df_headers.
            month_data_values: List with each element a dictionary containing data for one
            symbol, with key-value pairs of {data category: data}.

        Returns:
            A dictionary containing arrays with combined values and a tuple with the start and
            end indices of each symbol.
        """
        complete_data = {header: np.zeros(file_length, dtype=dtype) for header, dtype in
                         zip(df_headers, dtypes)}
        start, end = 0, 0
        symbol_indices = ([], [])
        for symbol_data in month_data.values():
            start = end
            end += symbol_data['close'].size
            symbol_indices[0].append(start)
            symbol_indices[1].append(end)
            for header, arr in symbol_data.items():
                if header not in ['hour', 'min', 'integer date']:
                    complete_data[header][start:end] = arr
        return complete_data, symbol_indices

    def convert_to_32_bits(complete_data):
        """Converts close, open, low, high, volume data from 64 to 32 bits for smaller Feather
        file.

        Args:
            complete_data: A dictionary containing data arrays for all symbols.
        """
        complete_data['close'] = complete_data['close'].astype(np.float32)
        complete_data['open'] = complete_data['open'].astype(np.float32)
        complete_data['low'] = complete_data['low'].astype(np.float32)
        complete_data['high'] = complete_data['high'].astype(np.float32)
        complete_data['volume'] = complete_data['volume'].astype(np.float32)

    def create_index_df(symbol_indices: Tuple[List[int], List[int]]) -> pd.DataFrame:
        """Creates Pandas DataFrame containing the symbol names, start and end indices.

        Args:
            symbol_indices: A list of the start and end indices.

        Returns:
            Output DataFrame.
        """
        index_df = pd.DataFrame()
        index_df['symbol'] = np.asarray(list(month_data))
        index_df['start'] = np.asarray(symbol_indices[0], dtype=np.uint32)
        index_df['end'] = np.asarray(symbol_indices[1], dtype=np.uint32)
        return index_df

    def create_month_df(complete_data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Creates Pandas DataFrame containing bar data.

        Args:
            complete_data: A dictionary containing data arrays for all symbols.

        Returns:
            Output DataFrame.
        """
        month_df = pd.DataFrame()
        for header, full_arr in complete_data.items():
            month_df[header] = full_arr
        return month_df

    symbols = list(month_data.keys())
    file_length = 0
    file_length = calc_file_length(month_data.values())
    convert_time_list_to_array(month_data.values())
    df_headers = [header for header in month_data[symbols[0]].keys() if
                  header not in ['hour', 'min', 'integer date']]
    dtypes = [arr.dtype for arr in month_data[symbols[0]].values()]
    complete_data, symbol_indices = combine_data(df_headers, dtypes, month_data.values())
    convert_to_32_bits(complete_data)
    index_df = create_index_df(symbol_indices)
    month_df = create_month_df(complete_data)
    main_filename = '{}_{}_{}.feather'.format(month[:4], month[5:7], timeframe)
    index_filename = '{}_{}_{}_indices.feather'.format(month[:4], month[5:7], timeframe)
    data_filepath = FILEPATH + '\\' + main_filename
    index_filepath = FILEPATH + '\\' + index_filename
    month_df.to_feather(data_filepath)
    index_df.to_feather(index_filepath)


def import_stock_csv() -> List[str]:
    """Reads CSV file containing 6102 symbols.

    Returns:
         List of symbols.
    """
    symbols = []
    with open('stocks.csv', 'r') as file:
        reader = csv.reader(file)
        symbols = [row[0] for row in reader]
    return symbols


def init_prev_month_indicators(symbols: List[str]) -> Dict[
    str, Dict[str, Union[np.float32, np.ndarray]]]:
    """Initializes prev_month_indicators to NaN values or an empty array. Called only when
    processing the first month.

    Args:
        symbols: List of symbols.

    Returns:
        Dictionary with all values for indicators set to NaN or an empty array.
    """
    return {
        symbol: {
            '14 SMA': np.array([]),
            '20 SMA': np.array([]),
            '9 EMA': np.nan,
            '12 EMA': np.nan,
            '26 EMA': np.nan,
            '20 EMA': np.nan,
            '50 EMA': np.nan,
            '200 EMA': np.nan,
            'RSI': np.nan,
            'MACD': np.nan,
            'signal': np.nan,
            'ATR': np.nan,
            'Last Close': np.nan
        }
        for symbol in symbols
    }


def est_tuple_to_utc(date: Tuple[int, int, int]) -> Type[dt.datetime]:
    """Converts tuple representing data in EST to UTC time.

    Args:
        date: EST date in the form of (YYYY, MM, dd)

    Returns:
        Same time in UTC timezone.
    """
    return dt.datetime(date[0], date[1], date[2]).replace(tzinfo=EST_ZONE).astimezone(UTC_ZONE)


def main():
    # Enter start and end dates, timeframe for each candlestick, and a list of symbols
    start_date = (2020, 1, 1)
    end_date = (2022, 12, 31)
    timeframe = '1min'  # Should not exceed 10min
    #  Manually select symbols with a list, e.g. ['AAPL', 'SPY', 'MSFT', 'TSLA']
    symbols = import_stock_csv()

    start_date = est_tuple_to_utc(start_date)
    end_date = est_tuple_to_utc(end_date)
    prev_month_indicators = init_prev_month_indicators(symbols)
    p = multiprocessing.Pool()  # For calculating indicators in parallel (idles while importing
    # data from Alpaca).
    while start_date < end_date:
        month_data = {}
        month_end = start_date + relativedelta(months=1)
        month = (generate(start_date), generate(month_end))
        import_data(month, timeframe, symbols,
                    month_data)  # Imports one month of data for all symbols (multithreaded) -
        # updates month_data.
        prev_month_indicators = calculate_indicators(month_data, prev_month_indicators, timeframe,
                                                     p)  # Calculates technical indicators for
        # this month (using multiprocessing) - updates month_data.
        save_month(month_data, str(start_date)[:7], timeframe)  # Saves data to Feather file.
        start_date = month_end
    p.close()


if __name__ == '__main__':
    main()
