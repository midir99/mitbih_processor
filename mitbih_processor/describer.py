# -*- coding: utf-8 -*-
"""Functions to describe a MIT-BIH Arrhythmia Database record with a classifier
"""

import numpy as np
import pandas as pd

from mitbih_processor import qrsintervals
from mitbih_processor import preprocessing


def get_qrs_intervals_with_classes(record, qrs_inds, clf, clf_classes,
                                   decomposition_level, wavelet,
                                   interval_length=qrsintervals.QRS_INTERVAL_LENGTH):
    """Obtain classified QRS Intervals of a record.

    Parameters
    ----------
    record : wfdb.Record
        The record to extract the QRS Intervals.
    qrs_inds : array_like
        An array containing the QRS indices of the record.
    clf
        A TRAINED classifier from sklearn.
    clf_classes : array_like
        An array containing the labels of the classes used for classification.
    decomposition_level : int
        The decomposition level of the signal.
    wavelet : pywt.Wavelet
        The Wavelet from pywt to decompose the signal.
    interval_length : int
        The length of the QRS Interval.

    Returns
    -------
    list
        An array containing the QRS Intervals.
    """
    # Extract the QRS Intervals
    qrs_intervals = qrsintervals.extract_qrs_intervals_with_qrs_indices(
        qrs_inds, record, interval_length)
    # Define the parameters for the feature extraction
    features_number = 14 * decomposition_level - 1
    extracted_features_mat = np.zeros(shape=(1, features_number))
    df_headers = preprocessing._get_dataframe_headers(decomposition_level)
    for qrs_int in qrs_intervals:
        # Extract features
        signal = qrs_int.interval
        preprocessing._wavelet_feature_mat_extraction(
            extracted_features_mat, signal, 0, wavelet, decomposition_level)
        # Label the extracted features
        dataframe = pd.DataFrame(extracted_features_mat, columns=df_headers)
        # At this point, we do not know the class, so we are going to use 0 by
        # default. This will not affect the final result.
        dataframe['classes'] = [0]
        X = dataframe[dataframe.columns.difference(['classes'])]
        y = dataframe['classes']
        # Classify the extracted signal
        y_pred = clf.predict(X)[0]
        qrs_int.class_ = clf_classes[y_pred]

    return qrs_intervals


def describe_mitbih_record(qrs_intervals, clf_classes):
    """Prints the description of the extracted QRS Intervals.

    Parameters
    ----------
    qrs_intervals : array_like
        An array containing the QRS Intervals.
    clf_classes : array_like
        The names of the classes used for classification.

    """
    msg = 'There are: \n'
    for clf_class in clf_classes:
        count = sum(map(lambda qrs_int: qrs_int.class_ == clf_class,
                        qrs_intervals))
        msg += f' - {count} QRS Intervals classified as {clf_class}\n'

    print(msg)
