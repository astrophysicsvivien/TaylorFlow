import tensorflow as tf
import pycbc.types
import numpy as np

"""
Class representing a Frequency Series, heavily modeled off of PyCBC:
https://pycbc.org/pycbc/latest/html/_modules/pycbc/types/frequencyseries.html#FrequencySeries

"""


class FreqSeries:
    def __init__(self, initial_array, delta_f):
        self.data = tf.constant(initial_array)
        self.df = delta_f

    def get_sample_frequencies(self):
        """
        Returns a variable tensor of sample frequencies
        to use numpy array of data use output.numpy()
        """
        frequencies = np.arange(len(self.data)) * self.df
        return frequencies

    sample_frequencies = property(get_sample_frequencies, doc="Array of the sample frequencies.")

    def sample_rate(self):
        """
        Return the sample rate this FD series would have in the time domain. Assumes even Length Time series
        """
        return (len(self) - 1) * self.df * 2.0

    # Convert to PyCBC FreqSeries type

    def to_pycbc(self):
        """
        Converts Taylorflow Frequency Series into PyCBC Frequency Series
        """
        _initial_array = np.array(self.data, dtype=np.complex128)
        _delta_f = self.df
        _epoch = ""
        _dtype = None
        _copy = True
        return pycbc.types.FrequencySeries(_initial_array, delta_f=_delta_f, epoch=_epoch, dtype=_dtype, copy=_copy)


class TimeSeries:
    def __init__(self, initial_array, delta_t):
        self.array = tf.constant(initial_array)
        self.dt = delta_t
        pass
