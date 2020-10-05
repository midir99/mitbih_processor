# -*- coding: utf-8 -*-
"""Feature extraction of signals using Wavelet decomposition.
"""

from os import path

import numpy as np
import pandas as pd
import pywt
import scipy.stats as stats
import scipy.io as sio

from mitbih_processor.config import PROJECT_ROOT


def _get_dataframe_headers(decomposition_level):
    """Obtain the dataframe headers.

    Parameters
    ----------
    decomposition_level : int
        The signal decomposition level. This is used to calculate the number
        of nodes.

    Returns
    -------
    list
        A list containing the dataframe headers.
    """
    # Generate the headers of the DataFrame
    header_prefixes = [
        'mean_', 'std_dev_', 'median_', 'skewness_', 'kurtosis_', 'rms_']
    headers = []
    for hp in header_prefixes:
        for approx_node_number in range(decomposition_level):
            headers.append(hp + (approx_node_number + 1) * 'a')

        for detail_node_number in range(decomposition_level):
            headers.append(hp + (detail_node_number + 1) * 'd')

    # The ratio values for each sub-band is a special case, so it is not in the
    # header_prefixes
    approx_node_number = 0
    for approx_node_number in range(decomposition_level - 1):
        headers.append(
            'ratio_' + (approx_node_number + 1) * 'a' + '/' +
            (approx_node_number + 2) * 'a')

    if decomposition_level > 1:
        approx_node_number += 1

    headers.append(
        'ratio_' + (approx_node_number + 1) * 'a' + '/' +
        'd'
    )

    for detail_node_number in range(decomposition_level - 1):
        headers.append(
            'ratio_' + (detail_node_number + 1) * 'd' + '/' +
            (detail_node_number + 2) * 'd')

    # return the generated dataframe
    return headers


def _wavelet_feature_mat_extraction(extracted_features_mat, signal, i, wavelet,
                                    decomposition_level):
    """Extract the signal features using Wavelet decomposition.

    This function extracts features like mean, standard deviation, median,
    Skewness, Kurtosis, RMS and ratio values of each signal sub-bands.

    Parameters
    ----------
    extracted_features_mat : numpy.ndarray
        A matrix where the extracted features values will be put.
    signal : array_like
        The signal that is going to be decomposed.
    i : int
        The row where the extracted features values will be put.
    wavelet : pywt.Wavelet
        The wavelet to decompose the signal.
    decomposition_level : int
        The decomposition level of the signal.
    """
    wp = pywt.WaveletPacket(signal, wavelet, mode='symmetric',
                            maxlevel=decomposition_level)
    # Mean values for each sub-bands
    for j in range(decomposition_level * 0, decomposition_level * 1):
        approx_node = 'a' * (j + 1 - 0 * decomposition_level)
        extracted_features_mat[i, j] = np.mean(abs(wp[approx_node].data))

    for j in range(decomposition_level * 1, decomposition_level * 2):
        detail_node = 'd' * (j + 1 - 1 * decomposition_level)
        extracted_features_mat[i, j] = np.mean(abs(wp[detail_node].data))

    # Standard deviation of each sub-bands
    for j in range(decomposition_level * 2, decomposition_level * 3):
        approx_node = 'a' * (j + 1 - 2 * decomposition_level)
        extracted_features_mat[i, j] = np.std(wp[approx_node].data)

    for j in range(decomposition_level * 3, decomposition_level * 4):
        detail_node = 'd' * (j + 1 - 3 * decomposition_level)
        extracted_features_mat[i, j] = np.std(wp[detail_node].data)

    # Median values of each sub-band
    for j in range(decomposition_level * 4, decomposition_level * 5):
        approx_node = 'a' * (j + 1 - 4 * decomposition_level)
        extracted_features_mat[i, j] = np.median(wp[approx_node].data)

    for j in range(decomposition_level * 5, decomposition_level * 6):
        detail_node = 'd' * (j + 1 - 5 * decomposition_level)
        extracted_features_mat[i, j] = np.median(wp[detail_node].data)

    # Skewness values of each sub-bands
    for j in range(decomposition_level * 6, decomposition_level * 7):
        approx_node = 'a' * (j + 1 - 6 * decomposition_level)
        extracted_features_mat[i, j] = stats.skew(wp[approx_node].data)

    for j in range(decomposition_level * 7, decomposition_level * 8):
        detail_node = 'd' * (j + 1 - 7 * decomposition_level)
        extracted_features_mat[i, j] = stats.skew(wp[detail_node].data)

    # Kurtosis values of each sub-bands
    for j in range(decomposition_level * 8, decomposition_level * 9):
        approx_node = 'a' * (j + 1 - 8 * decomposition_level)
        extracted_features_mat[i, j] = stats.kurtosis(wp[approx_node].data)

    for j in range(decomposition_level * 9, decomposition_level * 10):
        detail_node = 'd' * (j + 1 - 9 * decomposition_level)
        extracted_features_mat[i, j] = stats.kurtosis(wp[detail_node].data)

    # RMS values of each sub-bands
    for j in range(decomposition_level * 10, decomposition_level * 11):
        approx_node = 'a' * (j + 1 - 10 * decomposition_level)
        extracted_features_mat[i, j] = np.sqrt(
            np.mean(wp[approx_node].data ** 2))

    for j in range(decomposition_level * 11, decomposition_level * 12):
        detail_node = 'd' * (j + 1 - 11 * decomposition_level)
        extracted_features_mat[i, j] = np.sqrt(
            np.mean(wp[detail_node].data ** 2))

    # Ratio of sub-bands (this is an special case)
    for j in range(decomposition_level * 12, decomposition_level * 13 - 1):
        approx_node = 'a' * (j + 1 - 12 * decomposition_level)
        extracted_features_mat[i, j] = np.mean(
            abs(wp[approx_node].data)) / np.mean(
            abs(wp[approx_node + 'a'].data))

    j += 1
    if decomposition_level > 1:
        approx_node += 'a'

    extracted_features_mat[i, j] = np.mean(
        abs(wp[approx_node].data)) / np.mean(abs(wp['d'].data))

    for j in range(decomposition_level * 13, decomposition_level * 14 - 1):
        detail_node = 'd' * (j + 1 - 13 * decomposition_level)
        extracted_features_mat[i, j] = np.mean(
            abs(wp[detail_node].data)) / np.mean(
            abs(wp[detail_node + 'd'].data))


def _w_feature_extraction(ecg_signals, classes, signals_per_class, wavelet,
                          decomposition_level):
    """Extract features from ECG signals (QRS intervals) using Wavelets.

    Parameters
    ----------
    ecg_signals : dict
        A dict containing the ECG signals (QRS intervals).
    classes : list
        A list containing the classes of the ECG signals.
    signals_per_class : int
        Number of signals for each classification class.
    wavelet : pywt.Wavelet
        The wavelet that will be used to decompose the signals.
    decomposition_level : int
        The decomposition level for the signals.

    Returns
    -------
    tuple
        A tuple with a numpy.ndarray containing the extracted features and a
        list containing the labels for the processed signals.
    """
    features_number = 14 * decomposition_level - 1
    extracted_features_mat = np.zeros(shape=(
        len(classes) * signals_per_class,
        features_number), dtype=float)
    labels = []
    for i, class_ in enumerate(classes):
        for j in range(signals_per_class * i, signals_per_class * (i + 1)):
            signal = ecg_signals[class_][j - signals_per_class * i, :]
            _wavelet_feature_mat_extraction(
                extracted_features_mat, signal, j, wavelet,
                decomposition_level)
            # We append the class_
            labels.append(i)

    return extracted_features_mat, labels


def extract_features(ecg_signals, classes, signals_per_class, wavelet,
                     decomposition_level):
    """Generates a dataframe with the extracted features of the ECG signals.

    Parameters
    ----------
    ecg_signals : dict
        A dict containing the ECG signals (QRS intervals).
    classes : list
        A list containing the classes of the ECG signals.
    signals_per_class : int
        Number of signals for each classification class.
    wavelet : pywt.Wavelet
        The wavelet that will be used to decompose the signals.
    decomposition_level : int
        The decomposition level for the signals.

    Returns
    -------
    pandas.DataFrame
        A dataframe with the the extracted features of the ECG signals.
    """
    df_headers = _get_dataframe_headers(decomposition_level)
    extracted_features_mat, labels = _w_feature_extraction(
        ecg_signals, classes, signals_per_class, wavelet, decomposition_level)
    dataframe = pd.DataFrame(extracted_features_mat, columns=df_headers)
    dataframe['classes'] = labels
    return dataframe


def main():
    # Load the signals
    ecg_signals = sio.loadmat(
        path.join(PROJECT_ROOT, 'data/ecg_signals_101.mat'))

    # Define the parameters
    classes = ['A', 'N']
    signal_length = len(ecg_signals['A'][0, :])
    signals_per_class = 3
    wavelet = pywt.Wavelet('db1')

    # Compute variables
    max_decomposition_level = 1
    # Extract features
    df = extract_features(ecg_signals, classes, signals_per_class, wavelet,
                          max_decomposition_level)
    # Save the dataframe to a csv file
    filename = path.join(PROJECT_ROOT, 'data/ecg_signals_101.preprocessed.csv')
    print('Saving generated dataframe in', filename)
    df.to_csv(filename, index=False)


if __name__ == '__main__':
    main()
