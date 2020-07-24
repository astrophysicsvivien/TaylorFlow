import tensorflow as tf
import numpy as np


class TensorShapeError(Exception):
    """" raised when the input Tensor is not the expected shape"""
    pass


def normalise(tensor):
    """

    :param tensor:
    :return:
    """

    max_value = tf.reduce_max(tf.abs(tensor))
    norm_condition = tf.complex(tf.math.reciprocal(max_value), 0.)

    return tf.multiply(tensor, norm_condition)


def ComplexInnerProduct(temp, data, psd, df):
    """

    :param temp:
    :param data:
    :param psd:
    :param df:
    :return:
    """
    if tf.shape(temp) == tf.shape(data):
        if psd is None:
            psd = tf.ones(shape=tf.shape(temp))
        else:
            if tf.shape(temp) != tf.shape(psd):
                raise TensorShapeError('PSD must be same shape as template and data. Temp is {}, Data is {}.'.format(tf.shape(temp), tf.shape(psd)))
    else:
        raise TensorShapeError('Template and Data must be same shape. Temp is {}, Data is {}.'.format(tf.shape(temp), tf.shape(data)))

    return tf.multiply(4*df, tf.math.reduce_sum(tf.multiply(temp, tf.math.divide_no_nan(tf.math.conj(data), psd))))


def InnerProduct(temp, data, psd, df):
    """

    :param temp:
    :param data:
    :param psd:
    :param df:
    :return:
    """

    return tf.math.real(ComplexInnerProduct(temp, data, psd, df))


def overlap(temp, data, psd, df):
    """

    :param temp:
    :param data:
    :param psd:
    :param df:
    :return:
    """
    inner_prod = InnerProduct(temp, data, psd, df)
    normalisation = tf.sqrt(tf.multiply(InnerProduct(temp, temp, psd, df), InnerProduct(data, data, psd, df)))
    return tf.divide(inner_prod, normalisation)


def loglikelihood(temp, data, psd, df):
    """

    :param temp:
    :param data:
    :param psd:
    :param df:
    :return:
    """

    return tf.subtract(InnerProduct(temp, data, psd, df), 0.5*InnerProduct(temp, temp, psd, df))


def snr(temp, data, psd, df):
    """

    :param temp:
    :param data:
    :param psd:
    :param df:
    :return:
    """

    return


def snr_cpu(temp, data, psd, df):
    """

    :param temp:
    :param data:
    :param psd:
    :param df:
    :return:
    """

    return
