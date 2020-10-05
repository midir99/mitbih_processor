# -*- coding: utf-8 -*-

import os
import unittest
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import scipy.io as sio
import scipy.stats as stats
import wfdb
from wfdb import processing

from mitbih_processor import preprocessing
from mitbih_processor import qrsintervals

PROJECT_ROOT = Path(__file__).parent.parent
MITBIH_ARRHYTHMIA_DB_DIR = path.join(PROJECT_ROOT, 'data/mitdb')
ECG_SIGNALS_101_FILE = path.join(PROJECT_ROOT, 'data/ecg_signals_101.mat')
ECG_SIGNALS_101_PREP_FILE = path.join(
    PROJECT_ROOT, 'data/ecg_signals_101.preprocessed.csv')
QRS_INTERVALS_DATA_FILE = path.join(
    PROJECT_ROOT, 'data/testgen/qrs_intervals_data.mat')


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.ecg_signals_101 = sio.loadmat(ECG_SIGNALS_101_FILE)

    def test__get_dataframe_headers(self):
        level = 3
        expected_df_headers = [
            # mean values
            'mean_a', 'mean_aa', 'mean_aaa',
            'mean_d', 'mean_dd', 'mean_ddd',
            # standard deviantion values
            'std_dev_a', 'std_dev_aa', 'std_dev_aaa',
            'std_dev_d', 'std_dev_dd', 'std_dev_ddd',
            # median values
            'median_a', 'median_aa', 'median_aaa',
            'median_d', 'median_dd', 'median_ddd',
            # Skewness values
            'skewness_a', 'skewness_aa', 'skewness_aaa',
            'skewness_d', 'skewness_dd', 'skewness_ddd',
            # Kurtosis values
            'kurtosis_a', 'kurtosis_aa', 'kurtosis_aaa',
            'kurtosis_d', 'kurtosis_dd', 'kurtosis_ddd',
            # RMS values
            'rms_a', 'rms_aa', 'rms_aaa',
            'rms_d', 'rms_dd', 'rms_ddd',
            # ratio values
            'ratio_a/aa', 'ratio_aa/aaa', 'ratio_aaa/d',
            'ratio_d/dd', 'ratio_dd/ddd',
        ]
        true_df_headers = preprocessing._get_dataframe_headers(level)
        self.assertEqual(expected_df_headers, true_df_headers)

    def test__wavelet_feature_mat_extraction(self):
        classes = 'A', 'N'
        signals_per_class = 3
        wavelet = pywt.Wavelet('db1')
        decomposition_level = 1
        # Compute variables
        features_number = 14 * decomposition_level - 1
        extracted_features_mat = np.zeros(shape=(
            len(classes) * signals_per_class,
            features_number), dtype=float)
        # Preprocess the signal
        i = 0
        signal = self.ecg_signals_101['A'][i, :]
        preprocessing._wavelet_feature_mat_extraction(
            extracted_features_mat, signal, i, wavelet, decomposition_level)

        wp = pywt.WaveletPacket(signal, wavelet, mode='symmetric',
                                maxlevel=decomposition_level)
        # Mean values
        self.assertEqual(
            np.mean(abs(wp['a'].data)), extracted_features_mat[i, 0])
        self.assertEqual(
            np.mean(abs(wp['d'].data)), extracted_features_mat[i, 1])
        # Standard deviation values
        self.assertEqual(
            np.std(wp['a'].data), extracted_features_mat[i, 2])
        self.assertEqual(
            np.std(wp['d'].data), extracted_features_mat[i, 3])
        # Median values
        self.assertEqual(
            np.median(wp['a'].data), extracted_features_mat[i, 4])
        self.assertEqual(
            np.median(wp['d'].data), extracted_features_mat[i, 5])
        # Skewness values
        self.assertEqual(
            stats.skew(wp['a'].data), extracted_features_mat[i, 6])
        self.assertEqual(
            stats.skew(wp['d'].data), extracted_features_mat[i, 7])
        # Kurtosis values
        self.assertEqual(
            stats.kurtosis(wp['a'].data), extracted_features_mat[i, 8])
        self.assertEqual(
            stats.kurtosis(wp['d'].data), extracted_features_mat[i, 9])
        # RMS values
        self.assertEqual(
            np.sqrt(np.mean(wp['a'].data ** 2)), extracted_features_mat[i, 10])
        self.assertEqual(
            np.sqrt(np.mean(wp['d'].data ** 2)), extracted_features_mat[i, 11])
        # Ratio values
        self.assertEqual(
            np.mean(abs(wp['a'].data)) / np.mean(abs(wp['d'].data)),
            extracted_features_mat[i, 12])

    def test__w_feature_extraction(self):
        classes = 'A', 'N'
        signals_per_class = 3
        wavelet = pywt.Wavelet('db1')
        decomposition_level = 1
        extracted_features_mat, labels = preprocessing._w_feature_extraction(
            self.ecg_signals_101, classes, signals_per_class, wavelet,
            decomposition_level)
        expected_labels = [0, 0, 0, 1, 1, 1]
        expected_extracted_features_mat_shape = (6, 13)
        self.assertEqual(expected_labels, labels)
        self.assertEqual(expected_extracted_features_mat_shape,
                         extracted_features_mat.shape)

    def test_extract_features(self):
        classes = ['A', 'N']
        signals_per_class = 3
        wavelet = pywt.Wavelet('db1')
        decomposition_level = 1
        true_df = preprocessing.extract_features(
            self.ecg_signals_101, classes, signals_per_class, wavelet,
            decomposition_level)
        expected_df = pd.read_csv(ECG_SIGNALS_101_PREP_FILE)
        self.assertTrue(np.isclose(expected_df, true_df).all())


class TestQRSIntervals(unittest.TestCase):
    def setUp(self):
        self.db_path = MITBIH_ARRHYTHMIA_DB_DIR
        self.record101 = wfdb.rdrecord(
            path.join(self.db_path, '101'), channel_names=['MLII'])
        self.record101_ann = wfdb.rdann(
            path.join(self.db_path, '101'), 'atr',
            return_label_elements=['symbol'])
        self.record101_xqrs = processing.XQRS(
            sig=self.record101.p_signal[:, 0], fs=self.record101.fs)
        self.record101_xqrs.detect(verbose=False)

    def test_cut_qrs_interval_from_signal(self):
        signal = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        qrs_index = 3
        interval_length = 3
        qrs_interval = qrsintervals.cut_qrs_interval_from_signal(
            signal, qrs_index, interval_length)
        self.assertEqual(
            qrs_interval['interval'],
            [2, 3, 4])
        self.assertEqual(qrs_interval['start_index'], 2)
        # Remember that end_index is exclusive
        self.assertEqual(qrs_interval['end_index'], 5)

        qrs_index = 9
        interval_length = 5
        with self.assertRaises(ValueError):
            qrsintervals.cut_qrs_interval_from_signal(
                signal, qrs_index, interval_length)

        qrs_index = 1
        interval_length = 4
        with self.assertRaises(ValueError):
            qrsintervals.cut_qrs_interval_from_signal(
                signal, qrs_index, interval_length)

        qrs_index = 1
        interval_length = 4
        qrs_interval = qrsintervals.cut_qrs_interval_from_signal(
            signal, qrs_index, interval_length, left=False)
        self.assertEqual(
            qrs_interval['interval'],
            [0, 1, 2, 3])
        self.assertEqual(qrs_interval['start_index'], 0)
        self.assertEqual(qrs_interval['end_index'], 4)

    def test_indices_of_samples_labeled_with(self):
        record101_Normal = 1860
        record101_APC = 3
        record101_Unclassifiable = 2
        self.assertEqual(
            record101_Normal,
            len(qrsintervals.indices_of_samples_labeled_with(
                'N', self.record101_ann)))
        self.assertEqual(
            record101_APC, len(qrsintervals.indices_of_samples_labeled_with(
                'A', self.record101_ann)))
        self.assertEqual(
            record101_Unclassifiable,
            len(qrsintervals.indices_of_samples_labeled_with(
                'Q', self.record101_ann)))

    def test_nearest_qrs_index(self):
        qrs_indices = np.array([21, 54, 65, 78, 90, 120])
        sample_index = 74
        nearest_qrs_index = qrsintervals.nearest_qrs_index(
            qrs_indices, sample_index)
        self.assertEqual(3, nearest_qrs_index)
        self.assertEqual(78, qrs_indices[nearest_qrs_index])

    def test_extract_qrs_intervals_with_qrs_indices(self):
        rrs = qrsintervals.extract_qrs_intervals_with_qrs_indices(
            self.record101_xqrs.qrs_inds, self.record101, 320)
        first = rrs[0]
        self.assertEqual(236, first.start_index)
        self.assertEqual(556, first.end_index)
        self.assertEqual(396, first.qrs_index)
        self.assertEqual(320, first.end_index - first.start_index)
        self.assertEqual('MLII', first.lead)
        self.assertEqual('101', first.record)

    def test_extract_qrs_intervals(self):
        rr_intervals_N = qrsintervals.extract_qrs_intervals(
            record=self.record101, label='N', record_ann=self.record101_ann,
            qrs_inds=self.record101_xqrs.qrs_inds)
        first = rr_intervals_N[0]
        self.assertEqual(236, first.start_index)
        self.assertEqual(556, first.end_index)
        self.assertEqual(396, first.qrs_index)
        self.assertEqual(320, first.end_index - first.start_index)
        self.assertEqual('MLII', first.lead)
        self.assertEqual('101', first.record)

    def test_extract_qrs_intervals_of_labels(self):
        # In the 101 record, there are 3 A signals and 1859 signals.
        labels = ['A', 'N']
        existing_qrs_intervals = [3, 1859]
        dataset = ['101']
        length = 320
        extracted_qrs = qrsintervals.extract_qrs_intervals_of_labels(
            labels, length, dataset, MITBIH_ARRHYTHMIA_DB_DIR, lead='MLII')
        # Test that it contains the correct labels and quantity of signals per
        # label.
        for label, intervals_for_label in zip(labels, existing_qrs_intervals):
            self.assertIn(label, extracted_qrs)
            self.assertEqual(intervals_for_label, len(extracted_qrs[label]))

    def test_qrs_intervals_to_mat_file(self):
        # In the 101 record, there are 3 A signals and 1859 signals.
        labels = ['A', 'N']
        existing_qrs_intervals = [3, 1859]
        dataset = ['101']
        length = 320
        extracted_qrs = qrsintervals.extract_qrs_intervals_of_labels(
            labels, length, dataset, MITBIH_ARRHYTHMIA_DB_DIR, lead='MLII')
        # Save the intervals to a mat file
        qrsintervals.qrs_intervals_to_mat_file(
            QRS_INTERVALS_DATA_FILE, extracted_qrs)
        # Load the mat file
        matfile = sio.loadmat(QRS_INTERVALS_DATA_FILE)
        # Test that it contains the correct labels and quantity of signals per
        # label.
        for label, intervals_for_label in zip(labels, existing_qrs_intervals):
            self.assertIn(label, matfile)
            self.assertEqual(intervals_for_label, len(matfile[label]))

        # Delete the generated file
        os.remove(QRS_INTERVALS_DATA_FILE)

    def test_extract_qrs_intervals_and_save(self):
        # In the 101 record, there are 3 A signals and 1859 signals.
        labels = ['A', 'N']
        existing_qrs_intervals = [3, 1859]
        dataset = ['101']
        length = 320
        qrsintervals.extract_qrs_intervals_and_save(
            MITBIH_ARRHYTHMIA_DB_DIR, QRS_INTERVALS_DATA_FILE, labels,
            dataset=dataset, interval_length=length)
        # Load the generated mat file
        matfile = sio.loadmat(QRS_INTERVALS_DATA_FILE)
        # Test that it contains the correct labels and quantity of signals per
        # label.
        for label, intervals_for_label in zip(labels, existing_qrs_intervals):
            self.assertIn(label, matfile)
            self.assertEqual(intervals_for_label, len(matfile[label]))

        # Delete the generated file
        os.remove(QRS_INTERVALS_DATA_FILE)


if __name__ == '__main__':
    unittest.main()
