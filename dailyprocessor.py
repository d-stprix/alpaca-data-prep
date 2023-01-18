"""Imports daily data from Alpaca, calculates indicators and saves to file.

All data for a list of specified symbols are imported and processed for a given time period.
Data for all symbols are combined to one Pandas DataFrame and saved to a .Feather file. One file
is created for the entire time period. Another file specifying the start and end indices of each symbol is
created.
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
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import pandas as pd
import time
from numba import njit
import math
import csv
from typing import Tuple, Union, Type, Dict, List, Set

API_KEY = < Insert your API key here >
SECRET_KEY = < Insert your secret key here >
FILEPATH = < Insert location to save files to here >
HEADERS = {'APCA-API-KEY-ID': API_KEY, 'APCA-API-SECRET-KEY': SECRET_KEY}
MARKET_DATA_BASE = 'https://data.alpaca.markets/'
UTC_ZONE = tz.gettz('UTC')
EST_ZONE = tz.gettz('America/New_York')
MAX_SR = 9


def import_data(date_range: Tuple[str, str], symbols: List[str],
                data_dict: Dict[str, Dict[str, Union[List[str], np.ndarray]]]):
    """"Imports historical market data from Alpaca.markets

    Gets all (unlike in the dataprocessor.py module which only imports 1 month of data) daily data
    for date range provided for the specified symbols from alpaca.markets using the requests
    module. Saves data to data_dict as 64-bit floats (This is done for better accuracy when
    calculating indicators. Floats stored in file will be 32 bits).

    Args:
        month: Start and end times for the month in RFC3339-format.
        symbols: List of symbols.
        data_dict: Dictionary to which imported data is saved.
    """
    print('Importing daily candles starting from: {} to {}'.format(date_range[0], date_range[1]))
    start = time.perf_counter()
    leftover_symbols = set(symbols)
    lock = threading.Lock()
    bars_url = '{}/v2/stocks/bars?timeframe=1Day'.format(MARKET_DATA_BASE)

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
            o, c, h, l, v, timestamp, int_day = [], [], [], [], [], [], []  # Arrays to keep track of open,
            # close, low, high, volume and bar time
            while page_token != None:
                params = {'symbols': symbol,
                          'timeframe': '1Day',
                          'start': date_range[0],
                          'end': date_range[1],
                          'limit': 10000,
                          'adjustment': 'split'
                          }
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
                    time_string = bar['t'][:10]
                    int_day.append(
                        int(time_string[:4]) * 10000 + int(time_string[5:7]) * 100 + int(
                            time_string[8:10]))
                    timestamp.append(time_string)
                page_token = raw_data['next_page_token']

            o, c, l, h, v, int_day = lists_to_np_float(64, o, c, l, h, v,
                                                       int_day)  # converts lists
            # to 64 bit numpy arrays
            data_dict[symbol] = {'time': timestamp,
                                 'open': o,
                                 'close': c,
                                 'low': l,
                                 'high': h,
                                 'volume': v,
                                 'integer day': int_day
                                 }

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
def get_missing(integer_day: np.ndarray, active_times: np.ndarray) -> \
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
    symbol_indices = np.searchsorted(active_times, integer_day)
    shifted_symbol_indices = np.empty_like(symbol_indices)
    shifted_symbol_indices[0] = symbol_indices[0]
    shifted_symbol_indices[1:] = symbol_indices[:-1]
    missing = np.where((symbol_indices >= shifted_symbol_indices + 2), True, False)
    missing_five = np.where((symbol_indices >= shifted_symbol_indices + 5), True, False)
    return np.flatnonzero(missing), np.flatnonzero(missing_five)


class IndicatorCalculator:
    """Contains functions for calculating indicators for one symbol.

    Attributes:
        symbol: A string of the symbol name.
        o: A Numpy array containing bar open prices.
        c: A Numpy array containing bar close prices.
        h: A Numpy array containing bar high prices.
        l: A Numpy array containing bar low prices.
        v: A Numpy array containing bar volume.
        timestamp: A list containing bar timestamps.
        integer_day: A Numpy array that contains an integer representation of the time
        (YYYYMMDD)
        missing: A Numpy array of booleans indicating whether this bar follows a missing bar
        (bars during which the market is closed are set to False).
        missing_five: A Numpy array of booleans indicating whether this bar follows 5 missing
        bars in a row (bars during which the market is closed are set to False).
    """

    def __init__(self, data: list):
        """Initializes IndicatorCalculator. Performs calculations to determine several instance
        variables."""
        self.symbol = data[0]
        data_dict = data[1]
        active_times = data[2]
        self.o = data_dict['open']
        self.c = data_dict['close']
        self.h = data_dict['high']
        self.l = data_dict['low']
        self.v = data_dict['volume']
        self.timestamp = data_dict['time']
        self.integer_day = data_dict['integer day']
        self.missing, self.missing_five = get_missing(self.integer_day, active_times)

    @staticmethod
    @njit(cache=True)
    def njit_sma(n: int, c: np.ndarray, missing_five: np.ndarray,
                 missing: np.ndarray) -> np.ndarray:
        """Calculates the n-period SMA.

        Args:
            n: SMA period.
            c: Close prices.
            missing_five: An array of booleans indicating whether this bar follows 5 missing
            bars in a row.
            missing: An array of booleans indicating whether this bar follows a missing bar.

        Returns:
            An array containing SMA.
        """
        sma = np.cumsum(c)
        sma[:n] = np.nan
        sma = (sma[n:] - sma[:-n]) / n
        nan_indices = njit_calc_nan(n, missing_five, missing, c.size)
        if len(nan_indices) != 0:
            nan_indices = np.array(list(nan_indices))
            sma[nan_indices] = np.nan
        return sma  # Returns as 64-bit floats (since only called by other functions)

    def calc_ema(self, n: int) -> np.ndarray:
        """Calculates the n-period EMA.

        Calls the njit_ema function for better performances.

        Args:
            n: EMA period.

        Returns:
            An array containing the EMA.
        """
        return self.njit_ema(n, self.c, self.missing_five, self.missing).astype(np.float32)

    @staticmethod
    @njit(cache=True)
    def njit_ema(n: int, c: np.ndarray, missing_five: np.ndarray,
                 missing: np.ndarray) -> np.ndarray:
        """Calculates the n-period EMA.

        For the first month, sets average of the first n values as the first EMA value. For
        proceeding months, converts the EMA using the last EMA of the previous month.
        If 5 or more bars in a row are missing, waits until n bars in a row are present and sets
        the next EMA as the average of these bars (the EMA of those bars are set to NaN).

        Args:
            n: EMA period.
            c: Close prices.
            missing_five: An array of booleans indicating whether this bar follows 5 missing
            bars in a row.
            missing: An array of booleans indicating whether this bar follows a missing bar.

        Returns:
            An array containing the EMA.
        """
        k = 2 / (n + 1)
        ema = np.empty_like(c)
        ema.fill(np.nan)
        if c.size < n:
            return ema
        nan_indices = njit_calc_nan(n, missing_five, missing, c.size)
        ema[n - 1] = c[:n].mean()
        start = n
        for i in range(start, c.size):
            if i not in nan_indices:
                if np.isnan(ema[i - 1]):
                    ema[i] = c[i - (n - 1): i + 1].mean()
                else:
                    ema[i] = (c[i] - ema[i - 1]) * k + ema[i - 1]
        return ema

    def calc_atr(self):
        """Calculates the ATR.

        Calls the njit_atr functions for better performance.

        Returns:
            Array containing the ATR.
        """
        return self.njit_atr(self.c, self.h, self.l, self.missing_five,
                             self.missing)

    @staticmethod
    @njit(cache=True)
    def njit_atr(c: np.ndarray, h: np.ndarray, l: np.ndarray, missing_five: np.ndarray,
                 missing: np.ndarray) -> np.ndarray:
        """Calculates the ATR

        After calculating the ATR for all bars, the values for the ones following 5 missing bars
        in a row are set to NaN. These indices are found using the njit_calc_nan function.

        Args:
            c: Bar close prices.
            h: Bar high prices.
            l: Bar low prices.
            missing_five: An array of booleans indicating whether this bar follows 5 missing
            bars in a row.
            missing: An array of booleans indicating whether this bar follows a missing bar.

        Returns:
            Array containing the ATR
        """
        n = 14
        right_shifted_c = np.empty_like(c)
        right_shifted_c[0] = np.nan
        right_shifted_c[1:] = c[:-1]
        high_minus_prev_close = np.absolute(h - right_shifted_c)
        low_minus_prev_close = np.absolute(l - right_shifted_c)
        atr = np.empty_like(c)
        atr.fill(np.nan)
        tr = np.maximum(h - l, high_minus_prev_close, low_minus_prev_close)
        start = n + 1
        atr[n] = tr[1:n + 1].mean()
        for i in range(start, c.size):
            atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
        nan_indices = njit_calc_nan(n, missing_five, missing, c.size)
        if len(nan_indices) != 0:
            nan_indices = np.array(list(nan_indices))
            atr[nan_indices] = np.nan
        return atr.astype(np.float32)

    def calc_gap_percent(self) -> np.ndarray:
        """Calculates the percentage change in price since the close of the previous trading day.

        Returns:
            Array containing the percentage.
         """
        right_shifted_c = np.empty_like(self.c)
        right_shifted_c[0] = np.nan
        right_shifted_c[1:] = self.c[:-1]
        gap_percent = 100 * (self.o - right_shifted_c) / right_shifted_c
        gap_percent[self.missing] = np.nan
        return gap_percent.astype(np.float32)

    def get_prev_color(self) -> np.ndarray:
        """Determines whether previous day is bullish (green) or bearish (red).

        Returns a boolean array where bullish days are True and bearish/doji ones are False.

        Returns:
            Array containing previous day's color.
        """
        prev_color = np.empty(self.c.size, dtype=np.bool)
        prev_color[0] = False
        prev_color[1:] = np.where(self.c > self.o, True, False)[:-1]
        return prev_color

    def calc_sr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns support/resistance values and start indices (day where there was a peak/trough).

        Each output is a 2D array. The array containing support/resistance values has MAX_SR
        preallocated slots. In the extremely unlikely case where more than MAX_SR
        support/resistance lines are detected, any extra values are not included. Unused slots are
        set as NaN. The start indices values are set to zero by default. There will always be an
        equal number of non-NaN values and non-zero index values.

        Returns:
            Arrays as described above.
        """

        n = 90  # Calculates support/resistance lines starting 90 days in the past.
        sr = np.empty((self.c.size, MAX_SR),
                      dtype=np.float32)  # Stores up to 7 support/resistance lines per data point.
        final_indices = np.empty((self.c.size, MAX_SR), dtype=np.uint16)
        sr.fill(np.nan)
        final_indices.fill(0)
        if self.c.size < n:
            return sr, final_indices
        body_high = np.maximum(self.c, self.o)
        body_low = np.minimum(self.c, self.o)
        body_length = body_high - body_low
        sma = np.empty_like(self.c)
        sma[:n] = np.nan
        sma[n:] = self.njit_sma(n, body_length, self.missing_five,
                                self.missing)  # Calculate average candle body length.
        for i in range(n, self.c.size):  # Iterating through each day.
            start = i - n + 1
            end = i + 1
            SR_list = []
            fitted_high = savgol_filter(body_high[start:end], 15, 7)  # Includes candle itself.
            fitted_low = savgol_filter(body_low[start:end], 15, 7)
            max_high = max(self.h[start:end])
            min_low = min(self.l[start:end])
            peaks_index, _ = find_peaks(fitted_high, prominence=((max_high - min_low) * 0.07))
            troughs_index, _ = find_peaks(-fitted_low, prominence=((max_high - min_low) * 0.07))
            long_candle = sma[i] * 1.5
            troughs, peaks, sr_indices = [], [], []
            #  Peaks and troughs used as support/resistance lines are calculated below. Uses
            #  open/close prices associated with peak (instead of high/low).
            for j in troughs_index:  # Append a trough (i.e. support). Based on lowest value of
                # open/close within one candle of the index found. Favors long shaved-top candles.
                if body_length[start + j - 1] > long_candle and body_high[start + j - 1] == self.h[
                    start + j - 1]:
                    troughs.append(body_low[start + j - 1])
                elif body_length[start + j] > long_candle and body_high[start + j] == self.h[
                    start + j]:
                    troughs.append(body_low[start + j])
                elif body_length[start + j + 1] > long_candle and body_high[start + j + 1] == \
                        self.h[start + j + 1]:
                    troughs.append(body_low[start + j + 1])
                else:
                    troughs.append(min(body_low[start + j - 1:start + j + 2]))
                sr_indices.append(start + j)
            for j in peaks_index:  # Append a peak (i.e. resistance). Same idea as troughs except
                # reverses high/low values.
                if body_length[start + j - 1] > long_candle and body_low[start + j - 1] == self.l[
                    start + j - 1]:
                    peaks.append(body_high[start + j - 1])
                elif body_length[start + j] > long_candle and body_low[start + j] == self.l[
                    start + j]:
                    peaks.append(body_high[start + j])
                elif body_length[start + j + 1] > long_candle and body_low[start + j + 1] == \
                        self.l[start + j + 1]:
                    peaks.append(body_high[start + j + 1])
                else:
                    peaks.append(max(body_high[start + j - 1:start + j + 2]))
                sr_indices.append(start + j)

            sr_list = troughs + peaks
            sr_index_dict = {value: index for value, index in zip(sr_list, sr_indices)}
            #  Removes support/resistance lines that are too close to each other. Favors old
            #  values.
            min_distance = (max(self.h[start:end]) - min(self.l[start:end])) * 0.1
            sorted_values = sorted(sr_list)
            prev_value, prev_index = -math.inf, math.inf
            sr_out, index_out = [], []
            for value in sorted_values:
                if prev_value + min_distance >= value and sr_index_dict[
                    value] < prev_index:  # Replace previous value.
                    prev_index = sr_index_dict[value]
                    prev_value = value
                    sr_out.pop()
                    index_out.pop()
                    sr_out.append(value)
                    index_out.append(prev_index)
                elif prev_value + min_distance < value:
                    prev_index = sr_index_dict[value]
                    prev_value = value
                    sr_out.append(value)
                    index_out.append(prev_index)
            if len(sr_out) > MAX_SR:  # Truncate support/resistance list if too many. Should not
                # happen.
                sr_out = sr_out[:MAX_SR]
            sr[i][:len(sr_out)] = sr_out
            final_indices[i][:len(sr_out)] = index_out
        return sr, final_indices


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

    def empty_data_dict(symbol: str, num_candles: int, int_day: int) -> List[
        Union[str, Dict[str, Union[np.float32, np.ndarray]]]]:
        """Creates a dictionary with all indicator data set to NaN or an array of NaN's.

        Args:
            symbol: Name of symbol.
            num_candles: Length of dataset.
            int_time: Integer representation of each day.
        Returns:
            List with the 1st element being the symbol name and 2nd element being a dictionary
            with the indicator name as keys and NaN or NaN arrays as values.
        """
        empty_32bit_array = np.array([np.nan] * num_candles, dtype=np.float32)

        return [symbol, {
            'integer day': int_time,
            '200 EMA': np.copy(empty_32bit_array),
            'ATR': np.copy(empty_32bit_array),
            'SR': np.full((num_candles * MAX_SR), np.nan, dtype=np.float32),
            'SR indices': np.zeros((num_candles * MAX_SR), dtype=np.uint16),
            'Gap Percent': np.copy(empty_32bit_array),
            'Previous Day': np.array([False] * num_candles, dtype=np.bool)
        }]

    symbol = input[0]
    num_candles = input[1]['close'].size
    if num_candles < 40:  # Basically no data for this symbol
        return empty_data_dict(symbol, num_candles, input[1]['integer day'])
    indicator_calculator = IndicatorCalculator(input)
    ema_200 = indicator_calculator.calc_ema(200)
    atr = indicator_calculator.calc_atr()
    gap_percent = indicator_calculator.calc_gap_percent()
    prev_color = indicator_calculator.get_prev_color()
    sr, sr_indices = indicator_calculator.calc_sr()  # Support/resistance lines

    return [symbol, {
        'integer day': indicator_calculator.integer_day,
        '200 EMA': ema_200,
        'ATR': atr,
        'gap percentage': gap_percent,
        'previous day': prev_color,
        'SR': sr,
        'SR indices': sr_indices
    }]


def calculate_indicators(data_dict: Dict[str, Dict[str, Union[List[str], np.ndarray]]]):
    """Calculates indicators for every symbol.

    Coordinates the calculation of indicators using a pool of worker processes. Each task
    consists of calculating the indicators for one symbol; and is assigned using the Pool.map()
    function. Saves the indicator data in the data_dict dictionary.

    Args:
        data_dict: Contains timestamp, open, close, high, low, volume, for each bar for every
        symbol.
    """

    def get_candle_times(
            data_dict: Dict[str, Dict[str, Union[List[str], np.ndarray]]]) -> np.ndarray:
        """Gets all trading days and sorts in ascending order. Deletes duplicates.

        Args:
            data_dict: Contains timestamp, open, close, high, low, volume, for each bar for
            every symbol.

        Returns:
            An array of all the timestamps.
        """
        times = {time for symbol in data_dict.keys() for time in (
            data_dict[symbol]['integer day'])}
        return np.array(sorted(list(times)))

    def append_indicators(data_dict: Dict[str, Dict[str, Union[List[str], np.ndarray]]],
                          indicator_data: List[
                              Union[str, Dict[str, Union[np.ndarray, np.float32]]]]):
        """Adds new indicator data to dictionary containing bar data.

        Args:
            data_dict: Contains timestamp, open, close, high, low, volume, for each bar for
            every symbol.
            indicator_data: A list with the 1st element being the symbol name and 2nd element
            being a dictionary containing all data regarding each indicator, i.e. {indicator
            name: indicator data}.
        """
        for entry in indicator_data:
            symbol = entry[0]
            if entry[1]:
                data_dict[symbol].update(entry[1])

    active_times = get_candle_times(data_dict)
    active_times = [active_times] * len(data_dict)
    start = time.perf_counter()
    with multiprocessing.Pool() as p:
        indicator_data = p.map(indicator_process,
                               zip(data_dict.keys(), data_dict.values(), active_times))
    end = time.perf_counter()
    print('Took {}s to calculate indicators for all symbols'.format(end - start))
    append_indicators(data_dict, indicator_data)


def save_month(data_dict: Dict[str, Dict[str, Union[List[str], np.ndarray]]]):
    """Saves data to Feather file.

    Creates a file containing bar (day, open, etc) and indicator data (200 EMA, support/resistance,
    etc). The header contains the data category (e.g. close, 200 EMA) followed by rows; each of
    which corresponding to a single daily bar of a symbol. The symbol ordering is random and the
    symbol is not labelled. This information is found in the second file created, which consists
    of three columns: symbol, start, end. Start and end refer to the indices of the first file
    at which data for a particular symbol starts and ends (exclusively) respectively.

    Args:
        data_dict: Contains timestamp, open, close, high, low, volume, and indicator data for
        each bar for every symbol.
    """

    def calc_file_length(data_dict_values: List[Dict[str, Union[List[str], np.ndarray]]]):
        """Calculates total number of bars across all symbols.

        Args:
            data_dict_values: List with each element a dictionary containing data for one
            symbol, with key-value pairs of {data category: data}.

        Returns:
            Total number of bars.
        """
        file_length = 0
        for symbol_data in data_dict_values:
            file_length += symbol_data['close'].size
        return file_length

    def convert_time_list_to_array(
            data_dict_values: List[Dict[str, Union[List[str], np.ndarray]]]):
        """Replaces timestamp list with Numpy array in data dictionary.

        Args:
            data_dict_values: List with each element a dictionary containing data for one
            symbol, with key-value pairs of {data category: data}.
        """
        for symbol_data in data_dict_values:
            symbol_data.update({'time': np.array(symbol_data['time'], dtype='<U10')})

    def combine_data(df_headers: List[str], dtypes: List[np.dtype]) -> Tuple[
        Dict[str, np.ndarray], Tuple[List[int], List[int]]]:
        """Concatenates like-arrays of all symbols into large array; computes start and end
        indices for all symbols.

        e.g. EMA arrays for all symbols are combined to one array. If 'MSFT' begins at index 100
        and ends at 200. The corresponding element in symbol_indices = ([100], [201])

        Args:
            df_headers: A list of the headers to be used in the Feather file.
            dtypes: The Numpy dtypes of each array (column) under df_headers.

        Returns:
            A dictionary containing arrays with combined values and a tuple with the start and
            end indices of each symbol.
        """
        complete_data = {header: np.zeros(file_length, dtype=dtype) for header, dtype in
                         zip(df_headers, dtypes) if header not in ['SR', 'SR indices']}
        complete_data['SR'] = np.full((file_length, MAX_SR), np.nan, dtype=np.float32)
        complete_data['SR indices'] = np.zeros((file_length, MAX_SR), dtype=np.uint16)
        start, end = 0, 0
        symbol_indices = ([], [])
        for symbol_data in data_dict.values():
            start = end
            end += symbol_data['close'].size
            symbol_indices[0].append(start)
            symbol_indices[1].append(end)
            for header, arr in symbol_data.items():
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
        index_df['symbol'] = np.asarray(list(data_dict))
        index_df['start'] = np.asarray(symbol_indices[0], dtype=np.uint32)
        index_df['end'] = np.asarray(symbol_indices[1], dtype=np.uint32)
        return index_df

    def create_data_df(complete_data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Creates Pandas DataFrame containing bar data.

        Args:
            complete_data: A dictionary containing data arrays for all symbols.

        Returns:
            Output DataFrame.
        """
        data_df = pd.DataFrame()
        for header, full_arr in complete_data.items():
            if header in ['SR', 'SR indices']:
                data_df[header] = [np.array(full_arr[i][:]) for i in range(full_arr.shape[0])]
            else:
                data_df[header] = full_arr
        return data_df

    symbols = list(data_dict.keys())
    file_length = calc_file_length(data_dict.values())
    convert_time_list_to_array(data_dict.values())
    df_headers = [header for header in data_dict[symbols[0]].keys()]
    dtypes = [arr.dtype for arr in data_dict[symbols[0]].values()]
    complete_data, symbol_indices = combine_data(df_headers, dtypes, data_dict.values())
    convert_to_32_bits(complete_data)
    index_df = create_index_df(symbol_indices)
    data_df = create_data_df(complete_data)
    main_filename = 'daily.feather'
    index_filename = 'daily_indices.feather'
    data_filepath = FILEPATH + '\\' + main_filename
    index_filepath = FILEPATH + '\\' + index_filename
    print('Saving to file...')
    data_df.to_feather(data_filepath)
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


def est_tuple_to_utc(date: Tuple[int, int, int]) -> Type[dt.datetime]:
    """Converts tuple representing data in EST to UTC time.

    Args:
        date: EST date in the form of (YYYY, MM, dd)

    Returns:
        Same time in UTC timezone.
    """
    return dt.datetime(date[0], date[1], date[2]).replace(tzinfo=EST_ZONE).astimezone(UTC_ZONE)


def main():
    # Enter start date and a list of symbols
    start_date = (2017, 1, 1)
    end_date = (2022, 12, 31)
    #  Manually select symbols with a list, e.g. ['AAPL', 'SPY', 'MSFT', 'TSLA']
    symbols = import_stock_csv()

    start_date = est_tuple_to_utc(start_date)
    end_date = est_tuple_to_utc(end_date)
    date_range = (generate(start_date), generate(end_date))
    data_dict = {}
    import_data(date_range, symbols, data_dict)
    calculate_indicators(data_dict)
    save_month(data_dict)


if __name__ == '__main__':
    main()
