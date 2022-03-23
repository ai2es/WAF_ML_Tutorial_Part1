""" 
Description:
-----------
This script hosts many helper functions to make notebooks cleaner. The hope is to not distract users with ugly code. 

Alot of these were sourced from Dr. Lagerquist and found originally in his gewitter repo (https://github.com/thunderhoser/GewitterGefahr).

"""

#additional libraries needed here
import scipy.stats as st 
import copy 
import sklearn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


############# Global default variables ####################
# if you want to see the functions go see after line 92

NUM_TRUE_POSITIVES_KEY = 'num_true_positives'
NUM_FALSE_POSITIVES_KEY = 'num_false_positives'
NUM_FALSE_NEGATIVES_KEY = 'num_false_negatives'
NUM_TRUE_NEGATIVES_KEY = 'num_true_negatives'

FREQ_BIAS_COLOUR = np.full(3, 152. / 255)
FREQ_BIAS_WIDTH = 2.
FREQ_BIAS_STRING_FORMAT = '%.2f'
FREQ_BIAS_PADDING = 5

TOLERANCE = 1e-6
DUMMY_TARGET_NAME = 'tornado_lead-time=0000-3600sec_distance=00000-10000m'

MIN_PROB_FOR_XENTROPY = np.finfo(float).eps
MAX_PROB_FOR_XENTROPY = 1. - np.finfo(float).eps

MIN_OPTIMIZATION_STRING = 'min'
MAX_OPTIMIZATION_STRING = 'max'
VALID_OPTIMIZATION_STRINGS = [
    MIN_OPTIMIZATION_STRING, MAX_OPTIMIZATION_STRING
]

NUM_TRUE_POSITIVES_KEY = 'num_true_positives'
NUM_FALSE_POSITIVES_KEY = 'num_false_positives'
NUM_FALSE_NEGATIVES_KEY = 'num_false_negatives'
NUM_TRUE_NEGATIVES_KEY = 'num_true_negatives'

BSS_KEY = 'brier_skill_score'
BRIER_SCORE_KEY = 'brier_score'
RESOLUTION_KEY = 'resolution'
RELIABILITY_KEY = 'reliability'
UNCERTAINTY_KEY = 'uncertainty'

POD_BY_THRESHOLD_KEY = 'pod_by_threshold'
POFD_BY_THRESHOLD_KEY = 'pofd_by_threshold'
SR_BY_THRESHOLD_KEY = 'success_ratio_by_threshold'
MEAN_FORECAST_BY_BIN_KEY = 'mean_forecast_by_bin'
EVENT_FREQ_BY_BIN_KEY = 'event_frequency_by_bin'

FORECAST_PROBABILITIES_KEY = 'forecast_probabilities'
OBSERVED_LABELS_KEY = 'observed_labels'
BEST_THRESHOLD_KEY = 'best_prob_threshold'
ALL_THRESHOLDS_KEY = 'all_prob_thresholds'
NUM_EXAMPLES_BY_BIN_KEY = 'num_examples_by_forecast_bin'
DOWNSAMPLING_DICT_KEY = 'downsampling_dict'
EVALUATION_TABLE_KEY = 'evaluation_table'

POD_KEY = 'pod'
POFD_KEY = 'pofd'
SUCCESS_RATIO_KEY = 'success_ratio'
FOCN_KEY = 'focn'
ACCURACY_KEY = 'accuracy'
CSI_KEY = 'csi'
FREQUENCY_BIAS_KEY = 'frequency_bias'
PEIRCE_SCORE_KEY = 'peirce_score'
HEIDKE_SCORE_KEY = 'heidke_score'
AUC_KEY = 'auc'
AUPD_KEY = 'aupd'

EVALUATION_TABLE_COLUMNS = [
    NUM_TRUE_POSITIVES_KEY, NUM_FALSE_POSITIVES_KEY, NUM_FALSE_NEGATIVES_KEY,
    NUM_TRUE_NEGATIVES_KEY, POD_KEY, POFD_KEY, SUCCESS_RATIO_KEY, FOCN_KEY,
    ACCURACY_KEY, CSI_KEY, FREQUENCY_BIAS_KEY, PEIRCE_SCORE_KEY,
    HEIDKE_SCORE_KEY, POD_BY_THRESHOLD_KEY, POFD_BY_THRESHOLD_KEY, AUC_KEY,
    SR_BY_THRESHOLD_KEY, AUPD_KEY, MEAN_FORECAST_BY_BIN_KEY,
    EVENT_FREQ_BY_BIN_KEY, RELIABILITY_KEY, RESOLUTION_KEY, BSS_KEY
]

EVALUATION_DICT_KEYS = [
    FORECAST_PROBABILITIES_KEY, OBSERVED_LABELS_KEY, BEST_THRESHOLD_KEY,
    ALL_THRESHOLDS_KEY, NUM_EXAMPLES_BY_BIN_KEY, DOWNSAMPLING_DICT_KEY,
    EVALUATION_TABLE_KEY
]

MIN_BINARIZATION_THRESHOLD = 0.
MAX_BINARIZATION_THRESHOLD = 1. + TOLERANCE

DEFAULT_NUM_RELIABILITY_BINS = 20
DEFAULT_FORECAST_PRECISION = 1e-4
THRESHOLD_ARG_FOR_UNIQUE_FORECASTS = 'unique_forecasts'

###########################################################

################### Helper functions ######################

def get_performance_diagram_boot(forecast_labels, observed_labels,sample_size=10000,desired_size=100000,ci=0.95):
    n_iter = int(desired_size/sample_size)
    if n_iter < 30:
        n_iter = 30
    pod_l = np.zeros(n_iter)
    sr_l = np.zeros(n_iter)
    for i in np.arange(0,n_iter):
        idx_boot = np.random.choice(np.arange(0,forecast_labels.shape[0]),replace=True,size=sample_size)
        fore_boot = forecast_labels[idx_boot]
        obse_boot = observed_labels[idx_boot]
        table = get_contingency_table(fore_boot, obse_boot)
        pod_l[i] = get_pod(table)
        sr_l[i] = get_success_ratio(table)

    sr_r = st.t.interval(alpha=ci, df=len(sr_l)-1, loc=np.mean(sr_l), scale=st.sem(sr_l)) 
    pod_r = st.t.interval(alpha=ci, df=len(pod_l)-1, loc=np.mean(pod_l), scale=st.sem(pod_l)) 
    
    return np.asarray(pod_r),np.asarray(sr_r)

def get_contingency_table(forecast_labels, observed_labels):
    """Computes contingency table.
    N = number of forecasts
    :param forecast_labels: See documentation for
        _check_forecast_and_observed_labels.
    :param observed_labels: See doc for _check_forecast_and_observed_labels.
    :return: contingency_table_as_dict: Dictionary with the following keys.
    contingency_table_as_dict['num_true_positives']: Number of true positives.
    contingency_table_as_dict['num_false_positives']: Number of false positives.
    contingency_table_as_dict['num_false_negatives']: Number of false negatives.
    contingency_table_as_dict['num_true_negatives']: Number of true negatives.
    """
    
    true_positive_indices = np.where(np.logical_and(
        forecast_labels == 1, observed_labels == 1
    ))[0]
    false_positive_indices = np.where(np.logical_and(
        forecast_labels == 1, observed_labels == 0
    ))[0]
    false_negative_indices = np.where(np.logical_and(
        forecast_labels == 0, observed_labels == 1
    ))[0]
    true_negative_indices = np.where(np.logical_and(
        forecast_labels == 0, observed_labels == 0
    ))[0]

    return {
        NUM_TRUE_POSITIVES_KEY: len(true_positive_indices),
        NUM_FALSE_POSITIVES_KEY: len(false_positive_indices),
        NUM_FALSE_NEGATIVES_KEY: len(false_negative_indices),
        NUM_TRUE_NEGATIVES_KEY: len(true_negative_indices)
    }

def get_pod(contingency_table_as_dict):
    """Computes POD (probability of detection).
    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: probability_of_detection: POD.
    """

    denominator = (
        contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY] +
        contingency_table_as_dict[NUM_FALSE_NEGATIVES_KEY]
    )

    if denominator == 0:
        return np.nan

    numerator = float(contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY])
    return numerator /  denominator

def get_sr(contingency_table_as_dict):
    """Computes success ratio.
    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: success_ratio: Success ratio.
    """

    denominator = (
        contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY] +
        contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY]
    )

    if denominator == 0:
        return np.nan

    numerator = float(contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY])
    return numerator / denominator

def get_acc(contingency_table_as_dict):
    """Computes accuracy.
    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: accuracy: accuracy.
    """
    denominator = (contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY] + \
                   contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY] + \
                   contingency_table_as_dict[NUM_FALSE_NEGATIVES_KEY] + \
                   contingency_table_as_dict[NUM_TRUE_NEGATIVES_KEY])
    numerator = contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY] + contingency_table_as_dict[NUM_TRUE_NEGATIVES_KEY]
    
    return 100*(numerator/denominator)

def csi_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes CSI (critical success index) from success ratio and POD.
    POD = probability of detection
    :param success_ratio_array: np array (any shape) of success ratios.
    :param pod_array: np array (same shape) of POD values.
    :return: csi_array: np array (same shape) of CSI values.
    """
    return (success_ratio_array ** -1 + pod_array ** -1 - 1.) ** -1

def frequency_bias_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes frequency bias from success ratio and POD.
    POD = probability of detection
    :param success_ratio_array: np array (any shape) of success ratios.
    :param pod_array: np array (same shape) of POD values.
    :return: frequency_bias_array: np array (same shape) of frequency biases.
    """
    return pod_array / success_ratio_array

def get_far(contingency_table_as_dict):
    """Computes FAR (false-alarm rate).
    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: false_alarm_rate: FAR.
    """
    return 1. - get_success_ratio(contingency_table_as_dict)

def get_pofd(contingency_table_as_dict):
    """Computes POFD (probability of false detection).
    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: probability_of_false_detection: POFD.
    """

    denominator = (
        contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY] +
        contingency_table_as_dict[NUM_TRUE_NEGATIVES_KEY]
    )

    if denominator == 0:
        return np.nan

    numerator = float(contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY])
    return numerator / denominator

def get_points_in_roc_curve(
        forecast_probabilities=None, observed_labels=None, threshold_arg=None,
        forecast_precision=DEFAULT_FORECAST_PRECISION):
    """Determines points in ROC (receiver operating characteristic) curve.
    N = number of forecasts
    T = number of binarization thresholds
    :param forecast_probabilities: See documentation for
        `_check_forecast_probs_and_observed_labels`.
    :param observed_labels: See doc for
        `_check_forecast_probs_and_observed_labels`.
    :param threshold_arg: See documentation for get_binarization_thresholds.
    :param forecast_precision: See doc for get_binarization_thresholds.
    :return: pofd_by_threshold: length-T np array of POFD values, to be
        plotted on the x-axis.
    :return: pod_by_threshold: length-T np array of POD values, to be plotted
        on the y-axis.
    """


    binarization_thresholds = get_binarization_thresholds(
        threshold_arg=threshold_arg,
        forecast_probabilities=forecast_probabilities,
        forecast_precision=forecast_precision)

    num_thresholds = len(binarization_thresholds)
    pofd_by_threshold = np.full(num_thresholds, np.nan)
    pod_by_threshold = np.full(num_thresholds, np.nan)

    for i in range(num_thresholds):
        these_forecast_labels = binarize_forecast_probs(
            forecast_probabilities, binarization_thresholds[i]
        )
        this_contingency_table_as_dict = get_contingency_table(
            these_forecast_labels, observed_labels)

        pofd_by_threshold[i] = get_pofd(this_contingency_table_as_dict)
        pod_by_threshold[i] = get_pod(this_contingency_table_as_dict)

    return pofd_by_threshold, pod_by_threshold

def get_binarization_thresholds(
        threshold_arg, forecast_probabilities=None,
        forecast_precision=DEFAULT_FORECAST_PRECISION):
    """Returns list of binarization thresholds.
    To understand the role of binarization thresholds, see
    binarize_forecast_probs.
    :param threshold_arg: Main threshold argument.  May be in one of 3 formats.
    [1] threshold_arg = "unique_forecasts".  In this case all unique forecast
        probabilities will become binarization thresholds.
    [2] 1-D np array.  In this case threshold_arg will be treated as an array
        of binarization thresholds.
    [3] Positive integer.  In this case threshold_arg will be treated as the
        number of binarization thresholds, equally spaced from 0...1.
    :param forecast_probabilities:
        [used only if threshold_arg = "unique_forecasts"]
        1-D np array of forecast probabilities to binarize.
    :param forecast_precision:
        [used only if threshold_arg = "unique_forecasts"]
        Before computing unique forecast probabilities, they will all be rounded
        to the nearest `forecast_precision`.  This prevents the number of
        thresholds from becoming ridiculous (millions).
    :return: binarization_thresholds: 1-D np array of binarization
        thresholds.
    :raises: ValueError: if threshold_arg cannot be interpreted.
    """

    if isinstance(threshold_arg, str):
        if threshold_arg != THRESHOLD_ARG_FOR_UNIQUE_FORECASTS:
            error_string = (
                'If string, threshold_arg must be "{0:s}".  Instead, got '
                '"{1:s}".'
            ).format(THRESHOLD_ARG_FOR_UNIQUE_FORECASTS, threshold_arg)

            raise ValueError(error_string)

        binarization_thresholds = np.unique(rounder.round_to_nearest(
            forecast_probabilities + 0., forecast_precision
        ))

    elif isinstance(threshold_arg, np.ndarray):
        binarization_thresholds = copy.deepcopy(threshold_arg)
    else:
        num_thresholds = copy.deepcopy(threshold_arg)

        binarization_thresholds = np.linspace(
            0, 1, num=num_thresholds, dtype=float)

    return _pad_binarization_thresholds(binarization_thresholds)

def _pad_binarization_thresholds(thresholds):
    """Pads an array of binarization thresholds.
    Specifically, this method ensures that the array contains 0 and a number
        slightly greater than 1.  This ensures that:
    [1] For the lowest threshold, POD = POFD = 1, which is the top-right corner
        of the ROC curve.
    [2] For the highest threshold, POD = POFD = 0, which is the bottom-left
        corner of the ROC curve.
    :param thresholds: 1-D np array of binarization thresholds.
    :return: thresholds: 1-D np array of binarization thresholds (possibly
        with new elements).
    """

    thresholds = np.sort(thresholds)

    if thresholds[0] > MIN_BINARIZATION_THRESHOLD:
        thresholds = np.concatenate((
            np.array([MIN_BINARIZATION_THRESHOLD]), thresholds
        ))

    if thresholds[-1] < MAX_BINARIZATION_THRESHOLD:
        thresholds = np.concatenate((
            thresholds, np.array([MAX_BINARIZATION_THRESHOLD])
        ))

    return thresholds

def binarize_forecast_probs(forecast_probabilities, binarization_threshold):
    """Binarizes probabilistic forecasts, turning them into deterministic ones.
    N = number of forecasts
    :param forecast_probabilities: length-N numpy array with forecast
        probabilities of some event (e.g., tornado).
    :param binarization_threshold: Binarization threshold (f*).  All forecasts
        >= f* will be turned into "yes" forecasts; all forecasts < f* will be
        turned into "no".
    :return: forecast_labels: length-N integer numpy array of deterministic
        forecasts (1 for "yes", 0 for "no").
    """

    forecast_labels = np.full(len(forecast_probabilities), 0, dtype=int)
    forecast_labels[forecast_probabilities >= binarization_threshold] = 1

    return forecast_labels

def get_area_under_roc_curve(pofd_by_threshold, pod_by_threshold):
    """Computes area under ROC curve.
    This calculation ignores NaN's.  If you use `sklearn.metrics.auc` without
    this wrapper, if either input array contains any NaN, the result will be
    NaN.
    T = number of binarization thresholds
    :param pofd_by_threshold: length-T numpy array of POFD values.
    :param pod_by_threshold: length-T numpy array of corresponding POD values.
    :return: area_under_curve: Area under ROC curve.
    """

    num_thresholds = len(pofd_by_threshold)
    expected_dim = np.array([num_thresholds], dtype=int)

    sort_indices = np.argsort(-pofd_by_threshold)
    pofd_by_threshold = pofd_by_threshold[sort_indices]
    pod_by_threshold = pod_by_threshold[sort_indices]

    nan_flags = np.logical_or(
        np.isnan(pofd_by_threshold),
        np.isnan(pod_by_threshold)
    )
    if np.all(nan_flags):
        return np.nan

    real_indices = np.where(np.invert(nan_flags))[0]

    return sklearn.metrics.auc(
        pofd_by_threshold[real_indices], pod_by_threshold[real_indices]
    )

pretty_names = [ '$\lambda_{\downarrow}$', '$T_{d}$', '$V_{fric}$', 'Gflux', '$Cloud_{high}$',
 '$Lat_{F}$', '$Cloud_{low}$', '$Cloud_{mid}$', 'IRBT', '$Sens_{F}$',
 'Hours $T_{sfc}$ $>$ 0', 'Hours $T_{sfc} \leq 0$', 'SfcRough', '$T_{sfc}$',
 '$I_{S}$', '$T_{2m}$', 'Hours $T_{2m}$ $>$ 0', 'Hours $T_{2m}$ $\leq $ 0',
 '$Cloud_{Tot}$', r'$\lambda_{\uparrow}$', 'VBD', 'VDD', '10m wind',
 'Date marker', 'Urban', 'Rural', 'Diff1', 'Diff2', 'Diff3',
 '$T_{sfc}$ - $T_{2m}$']


def make_performance_diagram_axis(ax=None,figsize=(5,5),CSIBOOL=True,FBBOOL=True,csi_cmap='Greys_r'):
    import matplotlib.patheffects as path_effects
    pe = [path_effects.withStroke(linewidth=2,
                                 foreground="k")]
    pe2 = [path_effects.withStroke(linewidth=2,
                                 foreground="w")]

    if ax is None:
        fig=plt.figure(figsize=figsize)
        fig.set_facecolor('w')
        ax = plt.gca()
    
    
    if CSIBOOL:
        sr_array = np.linspace(0.001,1,200)
        pod_array = np.linspace(0.001,1,200)
        X,Y = np.meshgrid(sr_array,pod_array)
        csi_vals = csi_from_sr_and_pod(X,Y)
        pm = ax.contourf(X,Y,csi_vals,levels=np.arange(0,1.1,0.1),cmap=csi_cmap)
        plt.colorbar(pm,ax=ax,label='CSI')
    
    if FBBOOL:
        fb = frequency_bias_from_sr_and_pod(X,Y)
        bias = ax.contour(X,Y,fb,levels=[0.25,0.5,1,1.5,2,3,5],linestyles='--',colors='Grey')
        plt.clabel(bias, inline=True, inline_spacing=FREQ_BIAS_PADDING,fmt=FREQ_BIAS_STRING_FORMAT, fontsize=10,colors='LightGrey')
    
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel('SR')
    ax.set_ylabel('POD')
    return ax

def get_mae(y,yhat):
    """ Calcualte the mean absolute error """
    return np.mean(np.abs(y-yhat))
def get_rmse(y,yhat):
    """ Calcualte the root mean squared error """
    return np.sqrt(np.mean((y-yhat)**2))
def get_bias(y,yhat):
    """ Calcualte the mean bias (i.e., error) """
    return np.mean(y-yhat)
def get_r2(y,yhat):
    """ Calcualte the coef. of determination (R^2) """
    ybar = np.mean(y)
    return 1 - (np.sum((y-yhat)**2))/(np.sum((y-ybar)**2))

###########################################################