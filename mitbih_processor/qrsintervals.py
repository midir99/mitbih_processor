# -*- coding: utf-8 -*-
"""Functions to extract specific labeled QRS interval from the MIT-BIH DB.
"""

from os import path

import numpy as np
import wfdb
from wfdb import processing
import scipy.io as sio

from mitbih_processor.datasets import CHAZAL_TEST_DATASET, CHAZAL_TRAIN_DATASET
from mitbih_processor.config import PROJECT_ROOT

QRS_INTERVAL_LENGTH = 320
"""int: The length of a QRS interval.
"""

DEFAULT_DS = CHAZAL_TEST_DATASET + CHAZAL_TRAIN_DATASET
"""list: A default dataset.
"""


class QRSInterval:
    """A container for objects that represent a QRS interval.
    The QRSInterval object is a container for information related to a QRS
    interval.

    Attributes
    ----------
    record: str
        The record name.
    qrs_index: int
        The QRS index in the full ECG signal.
    start_index: int
        The start index of this QRS interval in the full ECG signal.
    end_index: int
        The end index of this QRS interval in the full ECG signal.
    interval: array_like
        The QRS interval, a segment from start_index to end_index (exclusive)
        in the full ECG signal.
    class_: str
        The class of this QRSInterval.
    """
    def __init__(self):
        """Instantiate a QRSInstance object.
        """
        self.record = None
        self.qrs_index = None
        self.lead = None
        self.start_index = None
        self.end_index = None
        self.interval = None
        self.class_ = None

    def __str__(self):
        return f'QRS interval QRS={self.qrs_index} [{self.start_index}, {self.end_index})'

    def __repr__(self):
        return f'QRS interval QRS={self.qrs_index} [{self.start_index}, {self.end_index})'


def cut_qrs_interval_from_signal(signal, qrs_index, interval_length, left=True):
    """Cut a QRS interval from a signal.

    This function is used to extract QRS intervals. Also you can use this
    function to extract a slice from an array.
    What this function does is slicing the signal in order to the qrs_index
    remains at the center of the resulting slice.

    Parameters
    ----------
    signal : array_like
        The array where the QRS interval is extracted from.
    qrs_index : int
        The index of the QRS interval center.
    interval_length : int
        The desired length of the resulting QRS interval.
    left : bool
        If set to true, the resulting interval will have more elements to the
        left from the qrs_index center.

    Returns
    -------
    dict
        A dictionary containing three keys:
        - start_index: the start index of the sliced signal in the full signal
        - start_index: the end index of the sliced signal in the full signal
        - interval: the original signal sliced

    Raises
    ------
    ValueError
        If the signal cannot be sliced because the indices get out of bounds.

    Examples:
    >>> signal = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    >>> qrs_index = 4
    >>> length = 3
    >>> interval = cut_qrs_interval_from_signal(signal, qrs_index, length)
    >>> print(interval)
    """
    if interval_length % 2 == 0:
        start_index = qrs_index - interval_length // 2
        end_index = qrs_index + (interval_length // 2)
    else:
        interval_length -= 1
        start_index = qrs_index - interval_length // 2
        end_index = qrs_index + (interval_length // 2) + 1
        interval_length += 1

    if not left:
        start_index += 1
        end_index += 1

    if start_index < 0 or end_index > len(signal):
        raise ValueError('Indices out of bounds')

    return {
        'interval': signal[start_index:end_index],
        'start_index': start_index,
        'end_index': end_index,
    }


def indices_of_samples_labeled_with(label, record_ann):
    """Get the indices of the samples in the signal with an specific label.

    This function returns an array that contains the indices of the samples
    that were labeled with an specific label.

    Parameters
    ----------
    label : str
        The label of the elements you want.
    record_ann : wfdb.Annotation
        The annotation object from the wfdb library.

    Returns
    -------
    list
        A list containing the indices of the samples labeled with the specific
        label.
    """
    indices_of_samples_labeled = []
    for i, curr_label in enumerate(record_ann.symbol):
        if curr_label == label:
            sample_index = record_ann.sample[i]
            indices_of_samples_labeled.append(sample_index)

    return indices_of_samples_labeled


def nearest_qrs_index(qrs_indices, index_of_sample):
    """Get the nearest QRS index to the index of an specific sample.

    Parameters
    ----------
    qrs_indices : numpy.ndarray
        A numpy.ndarray containing the indices of the QRSs in the signal.
    index_of_sample : int
        The index of an specific sample.

    Returns
    -------
    int
        The index of the nearest QRS.

    Examples:
    >>> qrs_indices = [121, 235, 354, 422, 567, 651]
    >>> index_of_sample = 200
    >>> index_of_nearest_qrs = nearest_qrs_index(qrs_indices, index_of_sample)
    >>> print(f'The neares QRS index to the sample {index_of_sample} is {index_of_nearest_qrs}')
    """
    abs_qrs_indices_substracted = np.abs(qrs_indices - index_of_sample)
    nearest_qrs = np.argmin(abs_qrs_indices_substracted)
    return nearest_qrs


def extract_qrs_intervals_with_qrs_indices(
        qrs_indices, record, qrs_interval_length=QRS_INTERVAL_LENGTH,
        verbose=False):
    """Extract the QRS intervals from a record using the QRS indices.

    This functions generates a list of QRSInterval depending on the qrs_indices
    it receives. The signal is extracted from a MIT-BIH Arrhythmia Database
    record.

    Parameters
    ----------
    qrs_indices : array_like
        The indices of the QRS complexes.
    record : wfdb.Record
        The wfdb.Record object, whose signal is extracted.
    qrs_interval_length : int
        The length of the QRS intervals generated.
    verbose:
        If true, this function prints error messages when it cannot extract
        a QRS interval.

    Returns
    -------
    list
        A list of QRSInterval.
    """
    qrs_intervals = []
    for qrs_index in qrs_indices:
        error = False
        try:
            qrs_interval = cut_qrs_interval_from_signal(
                record.p_signal[:, 0], qrs_index, qrs_interval_length)
        except ValueError:
            try:
                qrs_interval = cut_qrs_interval_from_signal(
                    record.p_signal[:, 0], qrs_index, qrs_interval_length,
                    left=False)
            except ValueError:
                if verbose:
                    print(
                        f'Could not cut RR interval from QRS index {qrs_index}')
                error = True

        if error:
            continue

        qrs_interval_obj = QRSInterval()
        qrs_interval_obj.qrs_index = qrs_index
        qrs_interval_obj.lead = record.sig_name[0]
        qrs_interval_obj.record = record.record_name
        qrs_interval_obj.start_index = qrs_interval['start_index']
        qrs_interval_obj.end_index = qrs_interval['end_index']
        qrs_interval_obj.interval = qrs_interval['interval']
        qrs_intervals.append(qrs_interval_obj)

    return qrs_intervals


def extract_qrs_intervals(record_path=None, record=None, lead='MLII', label='N',
                          sampfrom=None, sampto=None, qrs_inds=None,
                          record_ann=None, verbose=False,
                          qrs_interval_length=QRS_INTERVAL_LENGTH):
    """Extract QRS intervals from a MIT-BIH Arrhythmia Database record.

    Use this function to extract QRS intervals from a record of the MIT-BIH
    Arrhythmia Database.

    Parameters
    ----------
    record_path : str
        The path of the record of the MIT-BIH Arrhythmia Database.
    record : wfdb.Record
        The record object. If you supply this, there's no need to supply the
        record_path, sampfrom, sampto and lead arguments.
    lead : str
        The lead from you want to extract the ECG signals (MLII, V1, V2, etc.).
    label : str
        The MIT-BIH Arrhythmia Database annotation of the QRS intervals that
        you want to extract.
    sampfrom : int
        The starting sample index to read the signal.
    sampto : int
        The sample number at which to stop reading the signal.
    qrs_inds : int
        The QRS indices of the signal, if you not supply this it will be
        generated using XQRS from wfdb.
    record_ann : wfdb.Annotation
        The wfdb.Annotation object of the MIT-BIH Arrhythmia Databse record.
    verbose : bool
        If true, this function will print information about the processing.
    qrs_interval_length : int
        The length of the QRS interval.

    Returns
    -------
    list
        A list of QRSInterval of the specified label.
    """
    # Read the record
    if not record:
        record = wfdb.rdrecord(record_path, sampfrom=sampfrom, sampto=sampto,
                               channel_names=[lead])
    # Detect QRS indices only if not calculated before
    if qrs_inds is None:
        qrs_inds = processing.xqrs_detect(
            sig=record.p_signal[:, 0], fs=record.fs, verbose=False)

    # Read the annotations file
    if not record_ann:
        record_ann = wfdb.rdann(record_path, 'atr', sampfrom=sampfrom,
                                sampto=sampto,
                                return_label_elements=['symbol'])

    # Get the samples labeled with `label`
    indices_of_samples_labeled = indices_of_samples_labeled_with(
        label, record_ann)
    # Find the QRS interval the samples belong to
    selected_qrs_indices = []
    for index_of_sample in indices_of_samples_labeled:
        nearest_qrs = nearest_qrs_index(qrs_inds, index_of_sample)
        selected_qrs_indices.append(qrs_inds[nearest_qrs])

    # Extract the QRS intervals
    qrs_intervals = extract_qrs_intervals_with_qrs_indices(
        selected_qrs_indices, record, qrs_interval_length, verbose=verbose)
    return qrs_intervals


def extract_qrs_intervals_of_labels(labels, signal_length, dataset, db_path,
                                    verbose=False, lead='MLII', sampfrom=0,
                                    sampto=650000):
    """Extract QRS intervals from MIT-BIH records and store them in a dict.

    Extract labeled QRS intervals from selected record of the MIT-BIH
    Arrhythmia Database. This functions generates a dictionary with the labels
    desired as keys, each key has a matrix as a value. Each row in the matrix
    is a QRS interval.

    Parameters
    ----------
    labels : list
        A list of the labels that you want to extract.
    signal_length : int
        The length of the QRS interval.
    dataset : list
        A list containing the records of the MIT-BIH Arrhythmia Database you
        want to extract the QRS intervals from.
    db_path : str
        The path of the MIT-BIH Arrhythmia Database in your file system.
    verbose : bool
        If true, this function prints information about the process.
    lead : str
        The lead from you want to extract the ECG signals (MLII, V1, V2, etc.).
    sampfrom : int
        The starting sample index to read the each signal.
    sampto : int
        The sample number at which to stop reading each signal.

    Returns
    -------
    dict
        A dictionary that contains the desired labels as the keys, and a matrix
        containing the extracted QRS intervals as the value.

    Examples:
    Obtain QRS intervals labeled with A and N. Each interval must have a length
    of 320 samples. The information will be obtained from record 101 and 102 of
    the MIT-BIH Arrhythmia Database, from the lead MLII.
    >>> labels = ['A', 'N']
    >>> dataset = ['101', '102']
    >>> length = 320
    >>> #extract_qrs_intervals_of_labels(labels, length, dataset, 'pathto/mitbih', lead='MLII')
    """
    qrs_intervals_dict = {key: list([]) for key in labels}
    for record_name in dataset:
        if verbose:
            print(f'Extracting QRS intervals from record {record_name}')
        record_path = path.join(db_path, record_name)
        record = wfdb.rdrecord(record_path, channel_names=[lead],
                               sampfrom=sampfrom, sampto=sampto)
        record_ann = wfdb.rdann(record_path, 'atr', sampfrom=sampfrom,
                                sampto=sampto,
                                return_label_elements=['symbol'])
        qrs_inds = processing.xqrs_detect(sig=record.p_signal[:, 0],
                                          fs=record.fs, verbose=False)
        for label in labels:
            labeled_rr_intervals = extract_qrs_intervals(
                record=record, record_ann=record_ann, qrs_inds=qrs_inds,
                label=label, qrs_interval_length=signal_length, verbose=verbose)
            if verbose:
                print(f' - Label {label}, {len(labeled_rr_intervals)} item(s)')
            qrs_intervals_dict[label] += labeled_rr_intervals

    return qrs_intervals_dict


def qrs_intervals_to_mat_file(filename, qrs_intervals):
    """Save the dict of QRS intervals in a matlab file.
    
    Parameters
    ----------
    filename : str
        The path of the mat file.
    qrs_intervals : dict
        The dict with the QRS intervals.
    """
    rr_intervals_dict = dict.fromkeys(qrs_intervals.keys())
    for label in qrs_intervals.keys():
        signals_matrix = []
        for rr_interval in qrs_intervals[label]:
            signals_matrix.append(rr_interval.interval)

        rr_intervals_dict[label] = np.array(signals_matrix)

    sio.savemat(filename, rr_intervals_dict)


def extract_qrs_intervals_and_save(db_path, filename, labels, lead='MLII',
                                   sampfrom=0, sampto=650000,
                                   dataset=DEFAULT_DS,
                                   interval_length=QRS_INTERVAL_LENGTH):
    """Extract specific QRS intervals and save them to a mat file.

    Parameters
    ----------
    db_path : str
        The path of the MIT-BIH Arrhythmia Database in your file system.
    filename : str
        The path of the mat file.
    labels : list
        A list of the labels that you want to extract.
    lead : str
        The lead from you want to extract the ECG signals (MLII, V1, V2, etc.).
    sampfrom : int
        The starting sample index to read the each signal.
    sampto : int
        The sample number at which to stop reading each signal.
    dataset : list
        A list containing the records of the MIT-BIH Arrhythmia Database you
        want to extract the QRS intervals from.
    interval_length : int
        The length of the QRS interval
    """
    print('Extracting QRS intervals for labels', ', '.join(labels))
    qrs_intervals_data = extract_qrs_intervals_of_labels(
        labels, interval_length, dataset=dataset, db_path=db_path,
        verbose=True, lead=lead, sampfrom=sampfrom, sampto=sampto)

    # Describe the data extracted and save it to a mat file
    print('Summary of extracted signals from MIT-BIH Arrhythmia Database:')
    for label in qrs_intervals_data:
        print(f' -> {label} interval(s): {len(qrs_intervals_data[label])}')

    print(f'Saving signals in {filename}')
    qrs_intervals_to_mat_file(filename, qrs_intervals_data)


def main():
    db_path = path.join(PROJECT_ROOT, 'data/mitdb')
    filename = path.join(PROJECT_ROOT, 'data/ecg_signals_101.mat')
    labels = ['A', 'V', 'N', 'L', 'R']
    dataset = ['101']
    extract_qrs_intervals_and_save(db_path, filename, labels, dataset=dataset)


if __name__ == '__main__':
    main()
