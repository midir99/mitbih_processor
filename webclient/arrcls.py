import locale

locale.setlocale(locale.LC_ALL, '')
from mitbih_processor import describer
from mitbih_processor import qrsintervals
from mitbih_processor import preprocessing
from mitbih_processor import datasets
from os import path
import wfdb
from wfdb import processing
import pywt
from scipy import io as sio
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import joblib

from flask import (
    Blueprint,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

bp = Blueprint('arrcls', __name__, url_prefix='/arrcls')


def get_arrhythmia_types(request):
    arrhythmias = []

    if request.form.getlist('arrNormalBeat'):
        arrhythmias.append('N')

    if request.form.getlist('arrLeftBundleBranchBlockBeat'):
        arrhythmias.append('L')

    if request.form.getlist('arrRightBundleBranchBlockBeat'):
        arrhythmias.append('R')

    if request.form.getlist('arrAtrialPrematureBeat'):
        arrhythmias.append('A')

    if request.form.getlist('arrAberratedAtrialPrematureBeat'):
        arrhythmias.append('a')

    if request.form.getlist('arrNodalJunctionalPrematureBeat'):
        arrhythmias.append('J')

    if request.form.getlist('arrSupraventricularPrematureBeat'):
        arrhythmias.append('S')

    if request.form.getlist('arrPrematureVentricularContraction'):
        arrhythmias.append('V')

    if request.form.getlist('arrFusionOfVentricularAndNormalBeat'):
        arrhythmias.append('F')

    if request.form.getlist('arrStartOfVentricularFlutterFibrillation'):
        arrhythmias.append('[')

    if request.form.getlist('arrVentricularFlutterWave'):
        arrhythmias.append('!')

    if request.form.getlist('arrEndOfVentricularFlutterFibrillation'):
        arrhythmias.append(']')

    if request.form.getlist('arrAtrialEscapeBeat'):
        arrhythmias.append('e')

    if request.form.getlist('arrNodalJunctionalEscapeBeat'):
        arrhythmias.append('j')

    if request.form.getlist('arrVentricularEscapeBeat'):
        arrhythmias.append('E')

    if request.form.getlist('arrPacedBeat'):
        arrhythmias.append('/')

    if request.form.getlist('arrFusionOfPacedAndNormalBeat'):
        arrhythmias.append('f')

    if request.form.getlist('arrNonConductedPWave'):
        arrhythmias.append('x')

    if request.form.getlist('arrUnclassifiableBeat'):
        arrhythmias.append('Q')

    if request.form.getlist('arrIsolatedQRSLikeArtifact'):
        arrhythmias.append('|')

    return arrhythmias


def get_records(request):
    selected_records = []
    records = filter(lambda r: r.startswith('record'), request.form.keys())
    for record in records:
        if request.form.getlist(record):
            selected_records.append(record[6:])

    return selected_records


def extract_signals(arrhythmias, records, mitdb_path, ecg_lead, sampfrom,
                    sampto, signal_length, filename):
    qrs_intervals_dict = {key: list([]) for key in arrhythmias}
    theaders = """<th scope="col">Arrhythmias\Records</th>"""
    arrcount = {key: dict() for key in arrhythmias}
    for record_name in records:
        theaders += f"""<th scope="col">{record_name}</th>"""
        record_path = path.join(mitdb_path, record_name)
        record = wfdb.rdrecord(record_path, channel_names=[ecg_lead],
                               sampfrom=sampfrom, sampto=sampto)
        if not record.p_signal is None:
            record_ann = wfdb.rdann(record_path, 'atr', sampfrom=sampfrom,
                                    sampto=sampto,
                                    return_label_elements=['symbol'])
            qrs_inds = processing.xqrs_detect(sig=record.p_signal[:, 0],
                                              fs=record.fs, verbose=False)
            for label in arrhythmias:
                try:
                    labeled_rr_intervals = qrsintervals.extract_qrs_intervals(
                        record=record, record_ann=record_ann,
                        qrs_inds=qrs_inds,
                        label=label, qrs_interval_length=signal_length,
                        verbose=False)
                except:
                    labeled_rr_intervals = []

                arrcount[label][record_name] = len(labeled_rr_intervals)
                qrs_intervals_dict[label] += labeled_rr_intervals
        else:
            for label in arrhythmias:
                arrcount[label][record_name] = 0

    trows = "<tr>"
    for arr in arrcount:
        trows += f"""<th scope="row">{arr}</th>"""
        for record in arrcount[arr]:
            trows += f"<td>{arrcount[arr][record]:n}</td>"

        trows += f"<td>{sum(arrcount[arr].values()):n}</td>"
        trows += "</tr>"

    return theaders, trows, qrs_intervals_dict


@bp.route('/signal_extraction', methods=('GET', 'POST'))
def signal_extraction():
    if request.method == 'POST':
        mitdb_path = request.form.get('mitdbPath')
        arrhythmias = get_arrhythmia_types(request)
        records = get_records(request)
        ecg_lead = request.form.get('ecgLead')
        sampfrom = request.form.get('sampFrom', type=int)
        sampto = request.form.get('sampTo', type=int)
        signal_length = request.form.get('resultingSignalLength', type=int)
        filename = request.form.get('extractedSignalsFilename')

        if len(arrhythmias) <= 0:
            flash('You have to select at least 1 arrhythmia', 'error')
            return redirect(url_for('arrcls.signal_extraction'))

        if len(records) <= 0:
            flash('You have to select at least 1 record', 'error')
            return redirect(url_for('arrcls.signal_extraction'))

        if sampto <= sampfrom:
            flash('Samp to must be bigger than samp from', 'error')
            return redirect(url_for('arrcls.signal_extraction'))

        if signal_length > sampto - sampfrom:
            flash(
                f'The signal length is bigger than the entire signal ({sampto - sampfrom} samples)',
                'error')
            return redirect(url_for('arrcls.signal_extraction'))

        theaders, trows, qrs_intervals_dict = extract_signals(
            arrhythmias, records, mitdb_path, ecg_lead, sampfrom, sampto,
            signal_length, filename)

        try:
            qrsintervals.qrs_intervals_to_mat_file(filename, qrs_intervals_dict)
        except:
            flash(
                f'There was a problem saving the extracted signals in the file: "{filename}".',
                'error')
            return redirect(url_for('arrcls.signal_extraction'))

        return render_template(
            'arrcls/signal_extraction_report.html', theaders=theaders,
            trows=trows, filename=filename, ecg_lead=ecg_lead)

    return render_template('arrcls/signal_extraction.html')


@bp.route('/feature_extraction', methods=('GET', 'POST'))
def feature_extraction():
    wavelets = pywt.wavelist()
    if request.method == 'POST':
        signalsfile = request.form.get('extractedSignalsFile')
        featuresfile = request.form.get('extractedFeaturesFilename')
        arrhythmias = get_arrhythmia_types(request)
        decomposition_level = request.form.get('decompositionLevel', type=int)
        wavelet_name = request.form.get('wavelet')
        signals_per_arr = request.form.get('signalsPerArr', type=int)
        try:
            ecg_signals = sio.loadmat(signalsfile)
        except:
            flash(
                'There was an error loading the signals from the specified file.',
                'error')
            return render_template('arrcls/feature_extraction.html',
                                   wavelets=wavelets)

        if len(arrhythmias) <= 0:
            flash('You have to select at least 1 arrhythmia', 'error')
            return render_template('arrcls/feature_extraction.html',
                                   wavelets=wavelets)

        for arr in arrhythmias:
            if arr not in ecg_signals:
                flash(
                    f'The file with the signals does not contain signals for arrhythmia "{arr}".',
                    'error')
                return render_template('arrcls/feature_extraction.html',
                                       wavelets=wavelets)

            actual_signals_per_arr = ecg_signals[arr].shape[0]
            if actual_signals_per_arr < signals_per_arr:
                flash(f'Not enough signals for arrhythmia "{arr}".', 'error')
                return render_template('arrcls/feature_extraction.html',
                                       wavelets=wavelets)

        wavelet = pywt.Wavelet(wavelet_name)
        features = preprocessing.extract_features(
            ecg_signals, arrhythmias, signals_per_arr, wavelet,
            decomposition_level)
        try:
            features.to_csv(featuresfile, index=False)
        except:
            flash(
                f'There was a problem saving the extracted features in the file: "{featuresfile}".',
                'error')
            return render_template('arrcls/feature_extraction.html',
                                   wavelets=wavelets)

        flash(f'The extracted features were saved in "{featuresfile}".')

    return render_template('arrcls/feature_extraction.html', wavelets=wavelets)


@bp.route('/get_classifier', methods=('GET', 'POST'))
def get_classifier():
    if request.method == 'POST':
        featuresfile = request.form.get('extractedFeaturesFile')
        percentage_for_training = request.form.get('percentOfDataForTraining',
                                                   type=int) / 100
        clffile = request.form.get('clfFilename')
        try:
            features = pd.read_csv(featuresfile)
        except:
            flash(
                'There was an error loading the features from the specified file.',
                'error')
            return render_template('arrcls/get_classifier.html')

        X = features[features.columns.difference(['classes'])]
        y = features['classes']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=percentage_for_training, random_state=5)
        clf = MLPClassifier(alpha=1, max_iter=1000)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = np.round(metrics.accuracy_score(y_test, y_pred), 4)
        precision = np.round(
            metrics.precision_score(y_test, y_pred, average='weighted'), 4)
        recall = np.round(
            metrics.recall_score(y_test, y_pred, average='weighted'), 4)
        f1 = np.round(metrics.f1_score(y_test, y_pred, average='weighted'), 4)
        cks = np.round(metrics.cohen_kappa_score(y_test, y_pred))
        mcc = np.round(metrics.matthews_corrcoef(y_test, y_pred))

        try:
            joblib.dump(clf, clffile)
        except:
            flash(
                f'There was a problem saving the classifier in the file: "{clffile}".',
                'error')
            return render_template('arrcls/get_classifier.html')

        return render_template(
            'arrcls/get_classifier_report.html', accuracy=accuracy,
            precision=precision, recall=recall, f1=f1, cks=cks, mcc=mcc,
            filename=clffile)

    return render_template('arrcls/get_classifier.html')


@bp.route('/signal_analyzer', methods=('GET', 'POST'))
def signal_analyzer():
    record_names = datasets.MIT_BIH_ARRHYTHMIA_DB
    wavelets = pywt.wavelist()
    if request.method == 'POST':
        mitdbpath = request.form.get('mitdbPath')
        clffile = request.form.get('clfFile')
        record_name = request.form.get('recordName')
        ecg_lead = request.form.get('ecgLead')
        signal_length = request.form.get('signalLength', type=int)
        arrhythmias = get_arrhythmia_types(request)
        wavelet_name = request.form.get('wavelet')
        decomposition_level = request.form.get('decompositionLevel', type=int)
        try:
            print(path.join(mitdbpath, record_name))
            record = wfdb.rdrecord(path.join(mitdbpath, record_name),
                                   channel_names=[ecg_lead])
        except:
            flash('Could not read the record, verify the database path.',
                  'error')
            return render_template('arrcls/signal_analyzer.html',
                                   record_names=record_names,
                                   wavelets=wavelets)

        if record.p_signal is None:
            flash(
                f'The record does not contain the {ecg_lead} lead, try with the one you used for signal extraction.',
                'error')
            return render_template('arrcls/signal_analyzer.html',
                                   record_names=record_names,
                                   wavelets=wavelets)

        try:
            clf = joblib.load(clffile)
        except:
            flash(f'Could not load the classifier from "{clffile}"')
            return render_template('arrcls/signal_analyzer.html',
                                   record_names=record_names,
                                   wavelets=wavelets)

        qrs_inds = processing.xqrs_detect(sig=record.p_signal[:, 0], fs=record.fs, verbose=False)
        wavelet = pywt.Wavelet(wavelet_name)
        qrs_intervals = describer.get_qrs_intervals_with_classes(
            record,
            qrs_inds,
            clf,
            arrhythmias,
            decomposition_level,
            wavelet,
            signal_length
        )
        return render_template('arrcls/signal_analyzer_report.html', arrhythmias=qrs_intervals, record_name=record_name)

    return render_template('arrcls/signal_analyzer.html', record_names=record_names, wavelets=wavelets)
