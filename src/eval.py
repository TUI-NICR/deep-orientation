# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import argparse as ap
import copy
import glob
import os

import numpy as np
from sklearn.metrics import confusion_matrix

from deep_orientation.outputs import OUTPUT_TYPES
from deep_orientation.outputs import (OUTPUT_REGRESSION, OUTPUT_CLASSIFICATION,
                                      OUTPUT_BITERNION)
import deep_orientation.postprocessing as post
import nicr_rgb_d_orientation_data_set as ds
from utils.io import read_json, write_json
from utils.io import read_keras_csv_logfile
from utils.io import create_directory_if_not_exists


def _parse_args():
    """Parse command-line arguments"""
    desc = 'Evaluate trained neural networks for orientation estimation'
    parser = ap.ArgumentParser(description=desc,
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('output_type',
                        type=str,
                        default=OUTPUT_BITERNION,
                        choices=OUTPUT_TYPES,
                        help=(f"Output type. One of {OUTPUT_TYPES} (default: "
                              f"{OUTPUT_BITERNION})"))

    # output ------------------------------------------------------------------
    parser.add_argument('-o', '--output_path',
                        type=str,
                        default=None,
                        help=("Path where to store created output files, "
                              "default: '../eval_outputs/' relative to the "
                              "location of this script"))

    # dataset -----------------------------------------------------------------
    parser.add_argument('-tb', '--training_basepath',
                        type=str,
                        default='/results/rotator',
                        help=("Path to training outputs (default: "
                              "'/results/rotator')"))

    parser.add_argument('-s', '--set',
                        type=str,
                        default=ds.TEST_SET,
                        choices=(ds.VALID_SET, ds.TEST_SET),
                        help=(f"Set to use for evaluation, default: "
                              f"{ds.TEST_SET}"))

    parser.add_argument('-ss', '--selection_set',
                        type=str,
                        default=ds.VALID_SET,
                        choices=(ds.TRAIN_SET, ds.VALID_SET, ds.TEST_SET),
                        help=(f"Set to use for deriving the best epoch, "
                              f"default: {ds.VALID_SET}"))

    parser.add_argument('-db', '--dataset_basepath',
                        type=str,
                        default='/datasets/rotator',
                        help=("Path to downloaded dataset (default: "
                              "'/datasets/rotator')"))

    parser.add_argument('-ds', '--dataset_size',
                        type=str,
                        choices=ds.SIZES,
                        default=ds.SIZE_SMALL,
                        help=(f"Dataset image size to use. One of :"
                              f"{ds.SIZES}, default: "
                              f"{ds.SIZE_SMALL}"))

    # other -------------------------------------------------------------------
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help="Enable verbose output")

    # return parsed args
    return parser.parse_args()


class _TrainingRun(object):
    """
    Simple container for a single training run.

    Parameters
    ----------
    path : str
        Path to trainings run folder containing all outputs.
    """
    def __init__(self, path):
        self._identifier = os.path.basename(path)
        self._path = path

        # load config json
        self._config = read_json(os.path.join(self._path, 'config.json'))

        # load csv
        self._log = read_keras_csv_logfile(os.path.join(self._path, 'log.csv'))

    @property
    def identifier(self):
        return self._identifier

    @property
    def config(self):
        return self._config

    @property
    def log(self):
        return self._log

    def get_best_epoch(self, metric='valid_loss'):
        if metric in self._log.keys():
            if metric.find('loss') > -1:
                # note that epoch starts from 0
                return np.argmin(self._log[metric])
            else:
                # TODO(dase6070): implement when required
                raise NotImplementedError()
        else:
            # TODO(dase6070): implement when required
            raise NotImplementedError()

    def load_network_outputs(self, epoch, dataset_name):
        assert epoch < self._config['n_epochs']
        assert dataset_name in (ds.VALID_SET, ds.TEST_SET)

        # map 'validation' to 'valid'
        if dataset_name == ds.VALID_SET:
            dataset_name = 'valid'

        filepath = os.path.join(self._path,
                                f'outputs_{dataset_name}_{epoch:04d}.npy')
        output = np.load(filepath)

        # skip loss since it is part of the log as well
        # loss = output[:, 0]
        n_cols_y = (output.shape[1]-1) // 2
        y_true = output[:, 1:1+n_cols_y]
        y_pred = output[:, -n_cols_y:]

        return y_true, y_pred


def _parse_training_basepath(training_basepath):
    """Parse training basepath for training runs"""
    directories = glob.glob(os.path.join(training_basepath,
                                         f'*__*__*__*__*__*'))
    return [_TrainingRun(path) for path in directories]


def _evaluate(training_runs,
              output_type,
              dataset_name,
              dataset,
              best_epoch_metric):
    """Run evaluation"""

    assert output_type in OUTPUT_TYPES
    assert dataset_name in (ds.TRAIN_SET, ds.VALID_SET, ds.TEST_SET)

    # filter training runs by given output type
    training_runs = [tr for tr in training_runs
                     if tr.config['output_type'] == output_type]

    # group training runs
    grouped_training_runs = {}
    for tr in training_runs:
        # get identifier without run id
        keys = tr.identifier.split('__')
        new_identifier = '__'.join(keys[:-1])

        if new_identifier in grouped_training_runs.keys():
            grouped_training_runs[new_identifier].append(tr)
        else:
            grouped_training_runs[new_identifier] = [tr]

    # filter trainings with less than 3 runs
    for tr_id in list(grouped_training_runs.keys()):
        if len(grouped_training_runs[tr_id]) < 3:
            print(f'{tr_id} skipped since there are less than 3 runs')
            del grouped_training_runs[tr_id]

    # create dict for final results
    res = {tr_id: {} for tr_id in grouped_training_runs}

    # run evaluation
    for tr_id in grouped_training_runs:
        # copy config from first training run to final dict
        config = copy.copy(grouped_training_runs[tr_id][0].config)
        del config['run_id']
        res[tr_id]['config'] = config

        # store overall number of runs
        res[tr_id]['n_runs'] = len(grouped_training_runs[tr_id])

        # evaluate each run and store results in final dict
        res_runs = []
        for tr in grouped_training_runs[tr_id]:
            run_id = tr.config['run_id']

            # determine best epoch (note: epoch is in range [0, n_epochs]
            best_epoch = tr.get_best_epoch(best_epoch_metric)

            # get metrics at best epoch from log
            metrics = {'loss': float(tr.log['loss'][best_epoch]),
                       'valid_loss': float(tr.log['valid_loss'][best_epoch]),
                       'test_loss': float(tr.log['test_loss'][best_epoch])}

            # load network output to determine additional metrics
            y_true, y_pred = tr.load_network_outputs(best_epoch, dataset_name)

            # just a small consistency check
            assert len(dataset) == y_true.shape[0] == y_pred.shape[0]

            # check for NaNs
            y_pred_contains_nan = np.any(np.isnan(y_pred))

            # skip evaluation if outputs contain NaN
            if y_pred_contains_nan:
                res_runs.append({
                    'run_id': int(run_id),
                    'nan_detected': bool(y_pred_contains_nan),
                    'n_samples': len(dataset),
                    'set_used_for_evaluation': dataset_name})
                continue

            # determine additional classification metrics
            if output_type == OUTPUT_CLASSIFICATION:
                y_true_max = np.argmax(y_true, axis=-1)
                y_pred_max = np.argmax(y_pred, axis=-1)
                # calculate confusion matrix
                cm = confusion_matrix(y_true_max, y_pred_max)
                metrics['cm'] = cm.tolist()

                # calculate cmc curve values
                y_pred_ = np.argsort(y_pred, axis=-1)[:, ::-1]
                ranks = np.argmax(y_pred_ == y_true_max[:, None], axis=-1)
                metrics['cmc'] = np.cumsum(np.bincount(ranks)).tolist()

            # determine regression metrics
            # postprocess network output
            if output_type == OUTPUT_REGRESSION:
                y_true = post.rad2deg(y_true)
                y_pred = post.rad2deg(y_pred)
            elif output_type == OUTPUT_BITERNION:
                y_true = post.biternion2deg(y_true)
                y_pred = post.biternion2deg(y_pred)
            elif output_type == OUTPUT_CLASSIFICATION:
                # convert classification results
                y_pred = post.class2deg(y_pred_max, config['n_classes'])
                # load ground truth from dataset
                y_true = np.array([s.orientation for s in dataset],
                                  dtype='float32')

            # determine error
            error = y_true - y_pred
            error[error > 180] -= 360
            error[error < -180] += 360
            # get error metrics
            abs_error = np.abs(error)
            metrics['abs_error_mean'] = float(np.mean(abs_error))
            metrics['abs_error_std'] = float(np.std(abs_error))
            percentiles = np.percentile(abs_error, (25, 50, 75))
            metrics['abs_error_25percentile'] = float(percentiles[0])
            metrics['abs_error_med'] = float(percentiles[1])
            metrics['abs_error_75percentile'] = float(percentiles[2])
            metrics['abs_error_min'] = float(np.min(abs_error))
            metrics['abs_error_max'] = float(np.max(abs_error))

            # calculate error histogram
            hist_values, _ = np.histogram(
                error, bins=np.linspace(-180, 180, 360+1))

            # determine n samples with largest error
            n = 100
            order = np.argsort(abs_error)[::-1]
            samples_with_largest_error = []
            for i in range(n):
                idx = order[i]
                sample = {'basename': dataset[idx].basename,
                          'error': float(error[idx])}
                samples_with_largest_error.append(sample)

            # determine n samples with smallest error
            n = 100
            order = np.argsort(abs_error)
            samples_with_smallest_error = []
            for i in range(n):
                idx = order[i]
                sample = {'basename': dataset[idx].basename,
                          'error': float(error[idx])}
                samples_with_smallest_error.append(sample)

            # determine some additional metrics for each person
            grouped_abs_error = {}
            for s, abs_err in zip(dataset, abs_error):
                if s.person_id in grouped_abs_error.keys():
                    grouped_abs_error[s.person_id].append(abs_err)
                else:
                    grouped_abs_error[s.person_id] = [abs_err]
            person_metrics = {}
            for person_id, values in grouped_abs_error.items():
                person_metrics[person_id] = {
                    'abs_error_mean': float(np.mean(values)),
                    'abs_error_std': float(np.std(values)),
                    'n_samples': len(values)}

            # append results
            res_runs.append({
                'run_id': int(run_id),
                'best_epoch': int(best_epoch),
                'best_epoch_metric': best_epoch_metric,
                'nan_detected': bool(y_pred_contains_nan),
                'n_samples': len(dataset),
                'error_histogram': hist_values.tolist(),
                'error_metrics': metrics,
                'error_person_metrics': person_metrics,
                'set_used_for_evaluation': dataset_name,
                'samples_with_largest_error': samples_with_largest_error,
                'samples_with_smallest_error': samples_with_smallest_error})

        # store results for all runs
        res[tr_id]['runs'] = res_runs

    return res


def main():
    # parse args --------------------------------------------------------------
    args = _parse_args()

    # output path -------------------------------------------------------------
    if args.output_path is None:
        output_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(output_path, 'eval_outputs')
    else:
        output_path = args.output_path
    create_directory_if_not_exists(output_path)

    # prepare evaluation ------------------------------------------------------
    # get all training runs
    runs = _parse_training_basepath(args.training_basepath)

    # define metric for determining best epoch
    if args.selection_set == ds.VALID_SET:
        best_epoch_metric = 'valid_loss'
    else:
        best_epoch_metric = args.selection_set+'_loss'

    # load dataset to get access to samples (same order in outputs of all runs)
    dataset = ds.load_set(dataset_basepath=args.dataset_basepath,
                          set_name=args.set,
                          default_size=args.dataset_size)

    # run evaluation ----------------------------------------------------------
    res = _evaluate(runs,
                    output_type=args.output_type,
                    dataset_name=args.set,
                    dataset=dataset,
                    best_epoch_metric=best_epoch_metric)

    # save output
    filepath = os.path.join(output_path,
                            f'results_{args.set}_{args.output_type}.json')
    write_json(filepath, res)


if __name__ == '__main__':
    main()
