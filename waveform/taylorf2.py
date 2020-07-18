# Script to generate Taylorf2 waveforms in the frequency domain
# Author - Matthew Woods

import tensorflow as tf
import numpy as np
from taylorflow.types import FreqSeries


class FailedWaveformError(Exception):
    """
    Raised if the Waveform generation fails or an invalid call is made
    """
    pass


def pn_phases(freq, m_total, eta, phase_order):
    """

    :param freq:
    :param m_total:
    :param eta:
    :param phase_order:
    :return:
    """
    # constants in equation
    piM = tf.multiply(m_total * 4.92549102554e-6, np.pi)  # Total mass times pi
    v = tf.pow(tf.multiply(piM, freq), (1 / 3))  # characteristic velocity of binary
    etasq = tf.pow(eta, 2)  # square of the symmetric mass ratio
    etacb = tf.pow(eta, 3)  # cube of the symmetric mass ratio
    gamma = 0.577215664901532  # Euler–Mascheroni constant

    # v parameters
    v0 = tf.pow(v, -5)
    v2 = tf.pow(v, 2)
    v3 = tf.pow(v, 3)
    v4 = tf.pow(v, 4)
    v5 = tf.pow(v, 5)
    v5log = tf.math.log(v)
    v6 = tf.pow(v, 6)
    v6log = tf.math.log(4 * v)  # constants in equation
    piM = tf.multiply(m_total * 4.92549102554e-6, np.pi)  # Total mass times pi
    v = tf.pow(tf.multiply(piM, freq), (1 / 3))  # characteristic velocity of binary
    etasq = tf.pow(eta, 2)  # square of the symmetric mass ratio
    etacb = tf.pow(eta, 3)  # cube of the symmetric mass ratio
    gamma = 0.577215664901532  # Euler–Mascheroni constant

    # v parameters
    v0 = tf.pow(v, -5)
    v2 = tf.pow(v, 2)
    v3 = tf.pow(v, 3)
    v4 = tf.pow(v, 4)
    v5 = tf.pow(v, 5)
    v5log = tf.math.log(v)
    v6 = tf.pow(v, 6)
    v6log = tf.math.log(4 * v)
    v7 = tf.pow(v, 7)

    # produce PN phase coefficients
    P0 = tf.multiply((3. / 128), tf.math.reciprocal(eta))
    P2 = tf.multiply(tf.add(743 / 84, (11 * eta)), (5 / 9))
    P3 = (-16 * np.pi)
    P4 = tf.multiply(tf.add((3058673 / 1016064), tf.add(((5429 / 1008) * eta), (617 / 144) * etasq)), (10))
    P5 = tf.multiply(tf.add(7729 / 84, (-13 * eta)), (np.pi * 5 / 9))
    Pl5 = tf.multiply(tf.add(7729 / 84, (-13 * eta)), (np.pi * 5 / 3))
    P6 = tf.add(((11583231236531 / 4694215680) - (640 * np.pi * np.pi / 3) - (6848 * gamma / 21)), \
                tf.add(((-15737765635 / 3048192 + 2255 * np.pi * np.pi / 12) * eta), \
                       tf.add(((76055 / 1728) * etasq), (-127825 / 1296) * etacb)))
    Pl6 = -(6848 / 21)
    P7 = tf.multiply(tf.add((77096675 / 254016), tf.add(((378515 / 1512) * eta), (-74045 / 756) * etasq)), np.pi)

    # Produce full PN terms
    PN0 = tf.multiply(P0, v0)
    PN2 = tf.multiply(P2, v2)
    PN3 = tf.multiply(P3, v3)
    PN4 = tf.multiply(P4, v4)
    PN5 = tf.multiply(tf.add(P5, tf.multiply(Pl5, v5log)), v5)
    PN6 = tf.multiply(tf.add(P6, tf.multiply(Pl6, v6log)), v6)
    PN7 = tf.multiply(P7, v7)

    if phase_order == 7:
        phases = tf.multiply(PN0, (1 + PN2 + PN3 + PN4 + PN5 + PN6 + PN7))
    elif phase_order == 6:
        phases = tf.multiply(PN0, (1 + PN2 + PN3 + PN4 + PN5 + PN6))
    elif phase_order == 5:
        phases = tf.multiply(PN0, (1 + PN2 + PN3 + PN4 + PN5))
    elif phase_order == 4:
        phases = tf.multiply(PN0, (1 + PN2 + PN3 + PN4))
    elif phase_order == 3:
        phases = tf.multiply(PN0, (1 + PN2 + PN3))
    elif phase_order == 2:
        phases = tf.multiply(PN0, (1 + PN2))
    elif phase_order == 1 or phase_order == 0:
        phases = PN0
    else:
        phases = PN7
        print('phase order {} not implemented, defaulting to 3.5PN order'.format(phase_order))

    return phases


def pn_amplitude(freq, chirp_mass, luminosity_distance):
    """

    :param freq:
    :param chirp_mass:
    :param luminosity_distance:
    :return:
    """

    amp_term1 = tf.math.reciprocal(luminosity_distance / 299792458)  # divide by factor of c to convert to units of time
    amp_term2 = tf.multiply(np.sqrt(5 / 24) * (np.pi ** (-2 / 3)),
                            tf.pow(chirp_mass * 4.92549102554e-6, (5 / 6)))  # times by mtsol to get units of time
    amp_term3 = tf.pow(freq, (-7 / 6))

    amplitude = tf.multiply(amp_term1, tf.multiply(amp_term2, amp_term3))

    return amplitude


def getwaveform(mass1, mass2, luminositydistance=1., f_low=10.,
                      df=1. / 512, phase_order=7):
    """

    :param mass1:
    :param mass2:
    :param luminositydistance:
    :param f_low:
    :param df: frequency spacing between data points
    :param phase_order: the pn expansion order to calculate phase to, currently 0PN to 3.5PN implemented, input as an
    int from 0 to 7
    :return: Returns an eager tensor of type float32
    """

    # constants
    l_distance = tf.constant((3.086e+22 * luminositydistance), name="LuminosityDistance", dtype=tf.float32)

    # generate mass terms
    mass1 = tf.constant(mass1, dtype='float32')
    mass2 = tf.constant(mass2, dtype='float32')
    mass_total = tf.add(mass1, mass2)
    mass_eta = tf.divide(tf.multiply(mass1, mass2), tf.pow(mass_total, 2))
    mass_chirp = tf.multiply(mass_total, tf.pow(mass_eta, (3 / 5)))

    # generate frequency terms circular
    f_iso = 1 / (
            6. ** 1.5 * np.pi * mass_total * 4.92549102554e-6)  # frequency of innermost stable orbit (SI units)
    k_min = int(f_low / float(df))
    k_max = int(f_iso / float(df))

    # define frequencies
    f = tf.range(0., f_iso, df)
    frequencies = tf.Variable(f, name='frequencies', dtype=tf.float32)
    calculation_frequencies = frequencies[k_min:k_max+1]

    # calculate phases and amplitudes
    phases = pn_phases(calculation_frequencies, mass_total, mass_eta, phase_order)
    i_phases = tf.multiply(tf.complex(0., 1.), tf.complex(phases - (np.pi / 4) + np.pi, 0.))
    amplitude = pn_amplitude(calculation_frequencies, mass_chirp, l_distance)

    # Calculate waveform
    waveform_shell = tf.Variable(tf.zeros(tf.shape(frequencies), dtype='complex64'))
    waveform_calculation = tf.multiply(tf.complex(amplitude, 0.), tf.exp(i_phases))
    waveform_indices = tf.constant(tf.range(k_min, k_max+1, 1))
    waveform = tf.compat.v1.scatter_update(waveform_shell, waveform_indices, waveform_calculation)

    # calculate plus and cross polarizations
    plus = waveform.numpy()
    cross = tf.multiply(tf.complex(0., 1.), plus)

    htildep = FreqSeries(plus, delta_f=df)
    htildec = FreqSeries(cross, delta_f=df)

    return htildep, htildec


def getwaveform_sequence(mass1, mass2, sample_frequencies=None, luminositydistance=1., df=1. / 512, phase_order=7):
    """

    :param mass1:
    :param mass2:
    :param sample_frequencies:
    :param luminositydistance:
    :param df: frequency spacing between data points
    :param phase_order: the pn expansion order to calculate phase to, currently 0PN to 3.5PN implemented, input as an
    int from 0 to 7
    :return: Returns an eager tensor of type float32
    """

    # constants
    l_distance = tf.constant((3.086e+22 * luminositydistance), name="LuminosityDistance", dtype=tf.float32)

    # generate mass terms
    mass1 = tf.constant(mass1, dtype='float32')
    mass2 = tf.constant(mass2, dtype='float32')
    mass_total = tf.add(mass1, mass2)
    mass_eta = tf.divide(tf.multiply(mass1, mass2), tf.pow(mass_total, 2))
    mass_chirp = tf.multiply(mass_total, tf.pow(mass_eta, (3 / 5)))

    # generate frequency terms
    f_iso = 1 / (
            6. ** 1.5 * np.pi * mass_total * 4.92549102554e-6)  # frequency of innermost stable orbit (SI units)

    # define frequencies
    if sample_frequencies is not None:
        frequencies = tf.Variable(sample_frequencies, name='frequencies', dtype=tf.float32)
    else:
        raise FailedWaveformError('A valid array of sample frequencies must be entered')

    # calculate phases and amplitudes
    phases = pn_phases(frequencies, mass_total, mass_eta, phase_order)
    i_phases = tf.multiply(tf.complex(0., 1.), tf.complex(phases - (np.pi / 4) + np.pi, 0.))
    amplitude = pn_amplitude(frequencies, mass_chirp, l_distance)

    # Calculate waveform
    waveform = tf.multiply(tf.complex(amplitude, 0.), tf.exp(i_phases))

    # calculate plus and cross polarizations
    plus = waveform.numpy()
    cross = tf.multiply(tf.complex(0., 1.), plus)

    htildep = FreqSeries(plus, delta_f=df)
    htildec = FreqSeries(cross, delta_f=df)

    return htildep, htildec

