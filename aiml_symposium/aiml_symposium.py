"""Normal Python version of notebook."""

import copy
import glob
import random
import os.path
import collections
import numpy
from scipy.ndimage import median_filter
from scipy.interpolate import RectBivariateSpline
import netCDF4
import keras.layers
import keras.backend as K
import keras.callbacks
import tensorflow
import matplotlib.colors
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import model_evaluation as binary_eval
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import model_eval_plotting
from generalexam.machine_learning import keras_metrics
from generalexam.machine_learning import cnn
from generalexam.machine_learning import evaluation_utils as eval_utils

print('Keras version = {0:s}'.format(str(keras.__version__)))

# Input and output directories.
TOP_INPUT_DIR_NAME = '/localdata/ryan.lagerquist/aiml_symposium/data'
OUTPUT_DIR_NAME = '/localdata/ryan.lagerquist/aiml_symposium/output'
file_system_utils.mkdir_recursive_if_necessary(directory_name=OUTPUT_DIR_NAME)

TOP_TRAINING_DIR_NAME = '{0:s}/training'.format(TOP_INPUT_DIR_NAME)
TOP_VALIDATION_DIR_NAME = '{0:s}/validation'.format(TOP_INPUT_DIR_NAME)
TOP_TESTING_DIR_NAME = '{0:s}/testing'.format(TOP_INPUT_DIR_NAME)
BEST_MODEL_FILE_NAME = '{0:s}/pretrained_model/model.h5'.format(
    TOP_INPUT_DIR_NAME)

# Used to find input files.
TIME_FORMAT = '%Y%m%d%H'
TIME_FORMAT_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9][0-2][0-9]'
BATCH_NUMBER_REGEX = '[0-9][0-9][0-9][0-9][0-9][0-9][0-9]'
NUM_BATCHES_PER_DIRECTORY = 1000

# Names of predictor variables.
TEMPERATURE_NAME = 'temperature_kelvins'
HEIGHT_NAME = 'height_m_asl'
SPECIFIC_HUMIDITY_NAME = 'specific_humidity_kg_kg01'
WET_BULB_THETA_NAME = 'wet_bulb_potential_temperature_kelvins'
U_WIND_GRID_RELATIVE_NAME = 'u_wind_grid_relative_m_s01'
V_WIND_GRID_RELATIVE_NAME = 'v_wind_grid_relative_m_s01'

VALID_PREDICTOR_NAMES = [
    TEMPERATURE_NAME, HEIGHT_NAME, SPECIFIC_HUMIDITY_NAME, WET_BULB_THETA_NAME,
    U_WIND_GRID_RELATIVE_NAME, V_WIND_GRID_RELATIVE_NAME
]

DUMMY_SURFACE_PRESSURE_MB = 1013

# Predictors and labels.
NO_FRONT_ENUM = 0
WARM_FRONT_ENUM = 1
COLD_FRONT_ENUM = 2

WARM_FRONT_PROB_THRESHOLD = 0.65
COLD_FRONT_PROB_THRESHOLD = 0.65

PREDICTOR_NAMES_FOR_CNN = [
    U_WIND_GRID_RELATIVE_NAME, V_WIND_GRID_RELATIVE_NAME, TEMPERATURE_NAME,
    SPECIFIC_HUMIDITY_NAME,
    U_WIND_GRID_RELATIVE_NAME, V_WIND_GRID_RELATIVE_NAME, TEMPERATURE_NAME,
    SPECIFIC_HUMIDITY_NAME
]

PRESSURE_LEVELS_FOR_CNN_MB = numpy.array([
    DUMMY_SURFACE_PRESSURE_MB, DUMMY_SURFACE_PRESSURE_MB,
    DUMMY_SURFACE_PRESSURE_MB, DUMMY_SURFACE_PRESSURE_MB,
    850, 850, 850, 850
], dtype=int)

# Pooling functions, activation functions, performance metrics.
MAX_POOLING_TYPE_STRING = 'max'
MEAN_POOLING_TYPE_STRING = 'avg'
VALID_POOLING_TYPE_STRINGS = [MAX_POOLING_TYPE_STRING, MEAN_POOLING_TYPE_STRING]

SIGMOID_FUNCTION_NAME = 'sigmoid'
TANH_FUNCTION_NAME = 'tanh'
RELU_FUNCTION_NAME = 'relu'
SELU_FUNCTION_NAME = 'selu'
ELU_FUNCTION_NAME = 'elu'
LEAKY_RELU_FUNCTION_NAME = 'leaky_relu'

VALID_ACTIVATION_FUNCTION_NAMES = [
    SIGMOID_FUNCTION_NAME, TANH_FUNCTION_NAME, RELU_FUNCTION_NAME,
    SELU_FUNCTION_NAME, ELU_FUNCTION_NAME, LEAKY_RELU_FUNCTION_NAME
]

LIST_OF_METRIC_FUNCTIONS = [
    keras_metrics.accuracy, keras_metrics.binary_pod,
    keras_metrics.binary_pofd, keras_metrics.binary_peirce_score,
    keras_metrics.binary_csi, keras_metrics.binary_frequency_bias,
    keras_metrics.binary_success_ratio
]

# Preset (not learned) convolution kernels.
EDGE_DETECTOR_MATRIX1 = numpy.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
], dtype=float)

EDGE_DETECTOR_MATRIX2 = numpy.array([
    [1, 0, -1],
    [0, 0, 0],
    [-1, 0, 1]
], dtype=float)

EDGE_DETECTOR_MATRIX3 = numpy.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=float)

# Plotting.
WIND_COLOUR_MAP_OBJECT = pyplot.get_cmap('binary')
FEATURE_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')
SALIENCY_COLOUR_MAP_OBJECT = pyplot.get_cmap('binary')

FONT_SIZE = 30
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

# Dictionary keys.
PREDICTOR_MATRIX_KEY = 'predictor_matrix'
TARGET_MATRIX_KEY = 'target_matrix'
VALID_TIMES_KEY = 'target_times_unix_sec'
ROW_INDICES_KEY = 'row_indices'
COLUMN_INDICES_KEY = 'column_indices'
NORMALIZATION_TYPE_KEY = 'normalization_type_string'
PREDICTOR_NAMES_KEY = 'narr_predictor_names'
PRESSURE_LEVELS_KEY = 'pressure_levels_mb'
DILATION_DISTANCE_KEY = 'dilation_distance_metres'
MASK_MATRIX_KEY = 'narr_mask_matrix'

FIRST_NORM_PARAM_KEY = 'first_normalization_param_matrix'
SECOND_NORM_PARAM_KEY = 'second_normalization_param_matrix'

METADATA_KEYS_TO_REPORT = [
    VALID_TIMES_KEY, NORMALIZATION_TYPE_KEY, PREDICTOR_NAMES_KEY,
    PRESSURE_LEVELS_KEY, FIRST_NORM_PARAM_KEY, SECOND_NORM_PARAM_KEY
]


def _check_predictor_name(predictor_name):
    """Error-checks name of predictor variable.

    :param predictor_name: Name of predictor variable.
    :raises: ValueError: if name is unrecognized.
    """

    error_checking.assert_is_string(predictor_name)

    if predictor_name not in VALID_PREDICTOR_NAMES:
        error_string = (
            '\n{0:s}\nValid predictor names (listed above) do not include '
            '"{1:s}".'
        ).format(str(VALID_PREDICTOR_NAMES), predictor_name)

        raise ValueError(error_string)


def _check_pooling_type(pooling_type_string):
    """Error-checks pooling type.

    :param pooling_type_string: Pooling type ("max" or "avg").
    :raises: ValueError: if pooling type is unrecognized.
    """

    error_checking.assert_is_string(pooling_type_string)

    if pooling_type_string not in VALID_POOLING_TYPE_STRINGS:
        error_string = (
            '\n{0:s}\nValid pooling types (listed above) do not include '
            '"{1:s}".'
        ).format(str(VALID_POOLING_TYPE_STRINGS), pooling_type_string)

        raise ValueError(error_string)


def _check_activation_function(function_name):
    """Error-checks activation function.

    :param function_name: Name of activation function.
    :raises: ValueError: if activation function is unrecognized.
    """

    error_checking.assert_is_string(function_name)

    if function_name not in VALID_ACTIVATION_FUNCTION_NAMES:
        error_string = (
            '\n{0:s}\nValid activation functions (listed above) do not '
            'include "{1:s}".'
        ).format(str(VALID_ACTIVATION_FUNCTION_NAMES), function_name)

        raise ValueError(error_string)


def _floor_to_nearest(input_value, rounding_base):
    """Rounds number(s) *down* to the nearest multiple of `rounding_base`.

    :param input_value: Scalar or numpy array of real numbers.
    :param rounding_base: Number(s) will be rounded down to the nearest multiple
        of this base.
    :return: output_value: Rounded version of `input_value`.
    """

    if isinstance(input_value, collections.Iterable):
        error_checking.assert_is_real_numpy_array(input_value)
    else:
        error_checking.assert_is_real_number(input_value)

    error_checking.assert_is_greater(rounding_base, 0)
    return rounding_base * numpy.floor(input_value / rounding_base)


def _add_colour_bar(
        axes_object, colour_map_object, values_to_colour, min_colour_value,
        max_colour_value, colour_norm_object=None,
        orientation_string='vertical', extend_min=True, extend_max=True,
        fraction_of_axis_length=1., font_size=FONT_SIZE):
    """Adds colour bar to existing axes.

    :param axes_object: Existing axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param values_to_colour: numpy array of values to colour.
    :param min_colour_value: Minimum value in colour map.
    :param max_colour_value: Max value in colour map.
    :param colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.  If `colour_norm_object is None`,
        will assume that scale is linear.
    :param orientation_string: Orientation of colour bar ("vertical" or
        "horizontal").
    :param extend_min: Boolean flag.  If True, the bottom of the colour bar will
        have an arrow.  If False, it will be a flat line, suggesting that lower
        values are not possible.
    :param extend_max: Same but for top of colour bar.
    :param fraction_of_axis_length: Fraction of axis length (y-axis if
        orientation is "vertical", x-axis if orientation is "horizontal")
        occupied by colour bar.
    :param font_size: Font size for labels on colour bar.
    :return: colour_bar_object: Colour bar (instance of
        `matplotlib.pyplot.colorbar`) created by this method.
    """

    error_checking.assert_is_real_numpy_array(values_to_colour)
    error_checking.assert_is_greater(max_colour_value, min_colour_value)
    error_checking.assert_is_string(orientation_string)
    error_checking.assert_is_boolean(extend_min)
    error_checking.assert_is_boolean(extend_max)
    error_checking.assert_is_greater(fraction_of_axis_length, 0.)
    # error_checking.assert_is_leq(fraction_of_axis_length, 1.)

    if colour_norm_object is None:
        colour_norm_object = matplotlib.colors.Normalize(
            vmin=min_colour_value, vmax=max_colour_value, clip=False)

    scalar_mappable_object = pyplot.cm.ScalarMappable(
        cmap=colour_map_object, norm=colour_norm_object)
    scalar_mappable_object.set_array(values_to_colour)

    if extend_min and extend_max:
        extend_string = 'both'
    elif extend_min:
        extend_string = 'min'
    elif extend_max:
        extend_string = 'max'
    else:
        extend_string = 'neither'

    if orientation_string == 'horizontal':
        padding = 0.01
        tick_rotation_deg = 90.
    else:
        padding = 0.02
        tick_rotation_deg = 0.

    colour_bar_object = pyplot.colorbar(
        ax=axes_object, mappable=scalar_mappable_object,
        orientation=orientation_string, pad=padding, extend=extend_string,
        shrink=fraction_of_axis_length)

    colour_bar_object.ax.tick_params(
        labelsize=font_size, rotation=tick_rotation_deg)

    return colour_bar_object


def create_paneled_figure(
        num_rows, num_columns, horizontal_spacing=0.05, vertical_spacing=0.05,
        shared_x_axis=False, shared_y_axis=False, keep_aspect_ratio=True,
        grid_spec_dict=None):
    """Creates paneled figure.

    This method only initializes the panels.  It does not plot anything.

    J = number of panel rows
    K = number of panel columns

    :param num_rows: J in the above discussion.
    :param num_columns: K in the above discussion.
    :param horizontal_spacing: Spacing (in figure-relative coordinates, from
        0...1) between adjacent panel columns.
    :param vertical_spacing: Spacing (in figure-relative coordinates, from
        0...1) between adjacent panel rows.
    :param shared_x_axis: Boolean flag.  If True, all panels will share the same
        x-axis.
    :param shared_y_axis: Boolean flag.  If True, all panels will share the same
        y-axis.
    :param keep_aspect_ratio: Boolean flag.  If True, the aspect ratio of each
        panel will be preserved (reflect the aspect ratio of the data plotted
        therein).
    :param grid_spec_dict: Dictionary of grid specifications (accepted by
        `matplotlib.pyplot.subplots`).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object_matrix: J-by-K numpy array of axes handles (instances
        of `matplotlib.axes._subplots.AxesSubplot`).
    """

    error_checking.assert_is_geq(horizontal_spacing, 0.)
    error_checking.assert_is_less_than(horizontal_spacing, 1.)
    error_checking.assert_is_geq(vertical_spacing, 0.)
    error_checking.assert_is_less_than(vertical_spacing, 1.)
    error_checking.assert_is_boolean(shared_x_axis)
    error_checking.assert_is_boolean(shared_y_axis)
    error_checking.assert_is_boolean(keep_aspect_ratio)

    if grid_spec_dict is None:
        grid_spec_dict = dict()

    figure_object, axes_object_matrix = pyplot.subplots(
        num_rows, num_columns, sharex=shared_x_axis, sharey=shared_y_axis,
        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
        gridspec_kw=grid_spec_dict
    )

    if num_rows == num_columns == 1:
        axes_object_matrix = numpy.full(
            (1, 1), axes_object_matrix, dtype=object
        )

    if num_rows == 1 or num_columns == 1:
        axes_object_matrix = numpy.reshape(
            axes_object_matrix, (num_rows, num_columns)
        )

    pyplot.subplots_adjust(
        left=0.02, bottom=0.02, right=0.98, top=0.95,
        hspace=horizontal_spacing, wspace=vertical_spacing)

    if not keep_aspect_ratio:
        return figure_object, axes_object_matrix

    for i in range(num_rows):
        for j in range(num_columns):
            axes_object_matrix[i, j].set(aspect='equal')

    return figure_object, axes_object_matrix


def plot_feature_map(
        feature_matrix, axes_object=None,
        colour_map_object=FEATURE_COLOUR_MAP_OBJECT, min_colour_value=None,
        max_colour_value=None):
    """Plots feature map.

    A "feature map" is a spatial grid containing either a raw or transformed
    input variable.  The "raw" variables are the predictors, whose names are
    listed at the top of this notebook.

    :param feature_matrix: Feature map as M-by-N numpy array.
    :param axes_object: Handle for axes on which feature map will be plotted
        (instance of `matplotlib.axes._subplots.AxesSubplot`).  If `axes_object
        is None`, this method will create a new set of axes.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param min_colour_value: Minimum value in colour map.
    :param max_colour_value: Max value in colour map.
    :return: axes_object: Handle for axes on which feature map was plotted
        (instance of `matplotlib.axes._subplots.AxesSubplot`).
    """

    error_checking.assert_is_numpy_array_without_nan(feature_matrix)
    error_checking.assert_is_numpy_array(feature_matrix, num_dimensions=2)

    if min_colour_value is None or max_colour_value is None:
        max_colour_value = numpy.percentile(
            numpy.absolute(feature_matrix), 99.
        )
        max_colour_value = numpy.maximum(max_colour_value, 1e-6)
        min_colour_value = -1 * max_colour_value

    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    axes_object.pcolormesh(
        feature_matrix, cmap=colour_map_object, vmin=min_colour_value,
        vmax=max_colour_value, shading='flat', edgecolors='None')

    axes_object.set_xlim(0., feature_matrix.shape[1])
    axes_object.set_ylim(0., feature_matrix.shape[0])
    axes_object.set_xticks([])
    axes_object.set_yticks([])

    _add_colour_bar(
        axes_object=axes_object, colour_map_object=colour_map_object,
        values_to_colour=feature_matrix, min_colour_value=min_colour_value,
        max_colour_value=max_colour_value, orientation_string='horizontal')

    return axes_object


def plot_wind_barbs(
        u_wind_matrix, v_wind_matrix, axes_object=None,
        colour_map_object=WIND_COLOUR_MAP_OBJECT, min_colour_speed=-1.,
        max_colour_speed=0., barb_length=8, empty_barb_radius=0.1,
        plot_every=2):
    """Uses barbs to plot wind field.

    Default input args for `colour_map_object`, `min_colour_speed`, and
    `max_colour_speed` will make all wind barbs black, regardless of their
    speed.

    :param u_wind_matrix: M-by-N numpy array of eastward velocities.
    :param v_wind_matrix: M-by-N numpy array of northward velocities.
    :param axes_object: See doc for `plot_feature_map`.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param min_colour_speed: Minimum speed (velocity magnitude) in colour map.
    :param max_colour_speed: Max speed in colour map.
    :param barb_length: Length of each wind barb.
    :param empty_barb_radius: Radius for "empty" wind barb (zero speed).
    :param plot_every: Will plot wind barb every K grid cells, where
        K = `plot_every`.
    :return: axes_object: See doc for `plot_feature_map`.
    """

    error_checking.assert_is_numpy_array_without_nan(u_wind_matrix)
    error_checking.assert_is_numpy_array(u_wind_matrix, num_dimensions=2)

    error_checking.assert_is_numpy_array_without_nan(v_wind_matrix)
    error_checking.assert_is_numpy_array(
        v_wind_matrix, exact_dimensions=numpy.array(u_wind_matrix.shape)
    )

    error_checking.assert_is_greater(max_colour_speed, min_colour_speed)
    error_checking.assert_is_geq(max_colour_speed, 0.)
    error_checking.assert_is_integer(plot_every)
    error_checking.assert_is_geq(plot_every, 1)

    barb_size_dict = {
        'emptybarb': empty_barb_radius
    }
    barb_increment_dict = {
        'half': 0.3,
        'full': 0.6,
        'flag': 3.
    }

    wind_speed_matrix = numpy.sqrt(u_wind_matrix ** 2 + v_wind_matrix ** 2)

    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    num_rows = u_wind_matrix.shape[0]
    num_columns = u_wind_matrix.shape[1]
    unique_y_coords = -0.5 + numpy.linspace(1, num_rows, num=num_rows)
    unique_x_coords = -0.5 + numpy.linspace(1, num_columns, num=num_columns)

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(
        unique_x_coords, unique_y_coords)

    axes_object.barbs(
        x_coord_matrix[::plot_every, ::plot_every],
        y_coord_matrix[::plot_every, ::plot_every],
        u_wind_matrix[::plot_every, ::plot_every],
        v_wind_matrix[::plot_every, ::plot_every],
        wind_speed_matrix[::plot_every, ::plot_every],
        length=barb_length, sizes=barb_size_dict,
        barb_increments=barb_increment_dict, fill_empty=True, rounding=False,
        cmap=colour_map_object,
        clim=numpy.array([min_colour_speed, max_colour_speed])
    )

    return axes_object


def find_training_file(
        top_training_dir_name, batch_number, raise_error_if_missing=True):
    """Locates file with training examples.

    :param top_training_dir_name: Name of top-level directory with training
        examples.
    :param batch_number: Desired batch number (integer).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: training_file_name: Path to file with training examples.  If file
        is missing and `raise_error_if_missing = False`, this method will just
        return the expected path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_training_dir_name)
    error_checking.assert_is_integer(batch_number)
    error_checking.assert_is_geq(batch_number, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    first_batch_number = int(_floor_to_nearest(
        batch_number, NUM_BATCHES_PER_DIRECTORY
    ))
    last_batch_number = first_batch_number + NUM_BATCHES_PER_DIRECTORY - 1

    downsized_3d_file_name = (
        '{0:s}/batches{1:07d}-{2:07d}/downsized_3d_examples_batch{3:07d}.nc'
    ).format(
        top_training_dir_name, first_batch_number, last_batch_number,
        batch_number
    )

    if raise_error_if_missing and not os.path.isfile(downsized_3d_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            downsized_3d_file_name)
        raise ValueError(error_string)

    return downsized_3d_file_name


def _file_name_to_batch_number(training_file_name):
    """Parses batch number from file name.

    :param training_file_name: Path to file with training examples.
    :return: batch_number: Integer.
    :raises: ValueError: if batch number cannot be parsed from file name.
    """

    pathless_file_name = os.path.split(training_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    return int(
        extensionless_file_name.split('downsized_3d_examples_batch')[-1]
    )


def find_many_training_files(
        top_training_dir_name, first_batch_number, last_batch_number):
    """Finds many files with training examples.

    :param top_training_dir_name: See doc for `find_training_file`.
    :param first_batch_number: First desired batch number (integer).
    :param last_batch_number: Last desired batch number (integer).
    :return: training_file_names: 1-D list of paths to training files.
    :raises: ValueError: if no files are found.
    """

    error_checking.assert_is_string(top_training_dir_name)
    error_checking.assert_is_integer(first_batch_number)
    error_checking.assert_is_integer(last_batch_number)
    error_checking.assert_is_geq(first_batch_number, 0)
    error_checking.assert_is_geq(last_batch_number, first_batch_number)

    downsized_3d_file_pattern = (
        '{0:s}/batches{1:s}-{1:s}/downsized_3d_examples_batch{1:s}.nc'
    ).format(top_training_dir_name, BATCH_NUMBER_REGEX)

    downsized_3d_file_names = glob.glob(downsized_3d_file_pattern)

    if len(downsized_3d_file_names) == 0:
        error_string = 'Cannot find any files with the pattern: "{0:s}"'.format(
            downsized_3d_file_pattern)
        raise ValueError(error_string)

    batch_numbers = numpy.array(
        [_file_name_to_batch_number(f) for f in downsized_3d_file_names],
        dtype=int
    )

    good_indices = numpy.where(numpy.logical_and(
        batch_numbers >= first_batch_number,
        batch_numbers <= last_batch_number
    ))[0]

    if len(good_indices) == 0:
        error_string = (
            'Cannot find any files with batch number in [{0:d}, {1:d}].'
        ).format(first_batch_number, last_batch_number)

        raise ValueError(error_string)

    downsized_3d_file_names = [downsized_3d_file_names[i] for i in good_indices]
    downsized_3d_file_names.sort()
    return downsized_3d_file_names


def _shrink_predictor_grid(predictor_matrix, num_half_rows=None,
                           num_half_columns=None):
    """Shrinks predictor grid (by cropping around the center).

    M = original num rows in grid
    N = original num columns in grid
    m = final num rows in grid (after shrinking)
    n = final num columns in grid (after shrinking)

    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param num_half_rows: Number of rows in half-grid (on either side of center)
        after shrinking.  If `num_half_rows is None`, rows will not be cropped.
    :param num_half_columns: Same but for columns.
    :return: predictor_matrix: Same as input, except that dimensions are now
        E x m x n x C.
    """

    if num_half_rows is not None:
        error_checking.assert_is_integer(num_half_rows)
        error_checking.assert_is_greater(num_half_rows, 0)

        center_row_index = int(
            numpy.floor(float(predictor_matrix.shape[1]) / 2)
        )

        first_row_index = center_row_index - num_half_rows
        last_row_index = center_row_index + num_half_rows
        predictor_matrix = predictor_matrix[
            :, first_row_index:(last_row_index + 1), ...
        ]

    if num_half_columns is not None:
        error_checking.assert_is_integer(num_half_columns)
        error_checking.assert_is_greater(num_half_columns, 0)

        center_column_index = int(
            numpy.floor(float(predictor_matrix.shape[2]) / 2)
        )

        first_column_index = center_column_index - num_half_columns
        last_column_index = center_column_index + num_half_columns
        predictor_matrix = predictor_matrix[
            :, :, first_column_index:(last_column_index + 1), ...
        ]

    return predictor_matrix


def read_examples(
        netcdf_file_name, metadata_only=False, predictor_names_to_keep=None,
        pressure_levels_to_keep_mb=None, num_half_rows_to_keep=None,
        num_half_columns_to_keep=None):
    """Reads learning examples from NetCDF file.

    C = number of predictors to keep

    :param netcdf_file_name: Path to input file.
    :param metadata_only: Boolean flag.  If True, will read only metadata
        (everything except predictor and target matrices).
    :param predictor_names_to_keep: length-C list of predictors to keep.  If
        None, all predictors will be kept.
    :param pressure_levels_to_keep_mb: length-C numpy array of pressure levels
        to keep (millibars).
    :param num_half_rows_to_keep: Number of half-rows to keep in predictor
        grids.  If None, all rows will be kept.
    :param num_half_columns_to_keep: Same but for columns.
    :return: example_dict: See doc for `create_examples`.
    """

    error_checking.assert_is_boolean(metadata_only)

    # Read file.
    dataset_object = netCDF4.Dataset(netcdf_file_name)

    valid_times_unix_sec = numpy.array(
        dataset_object.variables[VALID_TIMES_KEY][:], dtype=int)
    row_indices = numpy.array(
        dataset_object.variables[ROW_INDICES_KEY][:], dtype=int)
    column_indices = numpy.array(
        dataset_object.variables[COLUMN_INDICES_KEY][:], dtype=int)

    predictor_names = netCDF4.chartostring(
        dataset_object.variables[PREDICTOR_NAMES_KEY][:]
    )
    predictor_names = [str(s) for s in predictor_names]

    if hasattr(dataset_object, 'pressure_level_mb'):
        pressure_level_mb = int(getattr(dataset_object, 'pressure_level_mb'))
        pressure_levels_mb = numpy.array([pressure_level_mb], dtype=int)
    else:
        pressure_levels_mb = numpy.array(
            dataset_object.variables[PRESSURE_LEVELS_KEY][:], dtype=int)

    if predictor_names_to_keep is None and pressure_levels_to_keep_mb is None:
        predictor_names_to_keep = copy.deepcopy(predictor_names)
        pressure_levels_to_keep_mb = pressure_levels_mb + 0

    pressure_levels_to_keep_mb = numpy.round(
        pressure_levels_to_keep_mb
    ).astype(int)

    error_checking.assert_is_numpy_array(
        numpy.array(predictor_names_to_keep), num_dimensions=1)

    num_predictors_to_keep = len(predictor_names_to_keep)
    error_checking.assert_is_numpy_array(
        pressure_levels_to_keep_mb,
        exact_dimensions=numpy.array([num_predictors_to_keep], dtype=int)
    )

    predictor_indices = [
        numpy.where(numpy.logical_and(
            numpy.array(predictor_names) == n, pressure_levels_mb == l
        ))[0][0]
        for n, l in zip(predictor_names_to_keep, pressure_levels_to_keep_mb)
    ]

    found_normalization_params = (
        FIRST_NORM_PARAM_KEY in dataset_object.variables or
        SECOND_NORM_PARAM_KEY in dataset_object.variables
    )

    if found_normalization_params:
        if hasattr(dataset_object, NORMALIZATION_TYPE_KEY):
            normalization_type_string = str(getattr(
                dataset_object, NORMALIZATION_TYPE_KEY
            ))
        else:
            normalization_type_string = 'z_score'

        first_normalization_param_matrix = numpy.array(
            dataset_object.variables[FIRST_NORM_PARAM_KEY][
                ..., predictor_indices]
        )
        second_normalization_param_matrix = numpy.array(
            dataset_object.variables[SECOND_NORM_PARAM_KEY][
                ..., predictor_indices]
        )
    else:
        normalization_type_string = None
        first_normalization_param_matrix = None
        second_normalization_param_matrix = None

    if metadata_only:
        predictor_matrix = None
        target_matrix = None
    else:
        predictor_matrix = numpy.array(
            dataset_object.variables[PREDICTOR_MATRIX_KEY][
                ..., predictor_indices]
        )
        target_matrix = numpy.array(
            dataset_object.variables[TARGET_MATRIX_KEY][:]
        )

        predictor_matrix = _shrink_predictor_grid(
            predictor_matrix=predictor_matrix,
            num_half_rows=num_half_rows_to_keep,
            num_half_columns=num_half_columns_to_keep)

    example_dict = {
        VALID_TIMES_KEY: valid_times_unix_sec,
        ROW_INDICES_KEY: row_indices,
        COLUMN_INDICES_KEY: column_indices,
        PREDICTOR_NAMES_KEY: predictor_names_to_keep,
        PRESSURE_LEVELS_KEY: pressure_levels_to_keep_mb,
        DILATION_DISTANCE_KEY: getattr(dataset_object, DILATION_DISTANCE_KEY),
        MASK_MATRIX_KEY: numpy.array(
            dataset_object.variables[MASK_MATRIX_KEY][:], dtype=int
        )
    }

    if found_normalization_params:
        example_dict.update({
            NORMALIZATION_TYPE_KEY: normalization_type_string,
            FIRST_NORM_PARAM_KEY: first_normalization_param_matrix,
            SECOND_NORM_PARAM_KEY: second_normalization_param_matrix
        })

    if not metadata_only:
        example_dict.update({
            PREDICTOR_MATRIX_KEY: predictor_matrix.astype('float32'),
            TARGET_MATRIX_KEY: target_matrix.astype('float64')
        })

    dataset_object.close()
    return example_dict


def do_2d_convolution(
        feature_matrix, kernel_matrix, pad_edges=False, stride_length_px=1):
    """Convolves 2-D feature maps with 2-D kernel.

    M = number of rows in each feature map
    N = number of columns in each feature map
    C_i = number of input feature maps (channels)

    J = number of rows in kernel
    L = number of columns in kernel
    C_o = number of output feature maps (channels)

    :param feature_matrix: Input feature maps (numpy array).  Dimensions must be
        M x N x C_i or 1 x M x N x C_i.
    :param kernel_matrix: Kernel as numpy array.  Dimensions must be
        J x K x C_i x C_o.
    :param pad_edges: Boolean flag.  If True, edges of each input feature map
        will be zero-padded during convolution, so spatial dimensions of the
        output feature maps will be the same (M x N).  If False, dimensions of
        the output feature maps will be (M - J + 1) x (N - L + 1).
    :param stride_length_px: Stride length (pixels).  The kernel will move by
        this many rows or columns at a time as it slides over each input feature
        map.
    :return: feature_matrix: Output feature maps (numpy array).  Dimensions will
        be 1 x M x N x C_o or 1 x (M - J + 1) x (N - L + 1) x C_o, depending on
        whether or not edges were padded.
    """

    error_checking.assert_is_numpy_array_without_nan(feature_matrix)
    error_checking.assert_is_numpy_array_without_nan(kernel_matrix)
    error_checking.assert_is_numpy_array(kernel_matrix, num_dimensions=4)
    error_checking.assert_is_boolean(pad_edges)
    error_checking.assert_is_integer(stride_length_px)
    error_checking.assert_is_geq(stride_length_px, 1)

    if len(feature_matrix.shape) == 3:
        feature_matrix = numpy.expand_dims(feature_matrix, axis=0)

    error_checking.assert_is_numpy_array(feature_matrix, num_dimensions=4)

    if pad_edges:
        padding_string = 'same'
    else:
        padding_string = 'valid'

    feature_tensor = K.conv2d(
        x=K.variable(feature_matrix), kernel=K.variable(kernel_matrix),
        strides=(stride_length_px, stride_length_px), padding=padding_string,
        data_format='channels_last')

    return feature_tensor.eval(session=K.get_session())


def do_activation(input_values, function_name, alpha_parameter=0.2):
    """Runs input array through activation function.

    :param input_values: Input numpy array.
    :param function_name: Name of activation function (must be accepted by
        `_check_activation_function`).
    :param alpha_parameter: Slope (used only for eLU and leaky ReLU functions).
    :return: output_values: Output numpy array (after activation).  Same
        dimensions.
    """

    _check_activation_function(function_name)
    input_object = K.placeholder()

    if function_name == ELU_FUNCTION_NAME:
        function_object = K.function(
            [input_object],
            [keras.layers.ELU(alpha=alpha_parameter)(input_object)]
        )
    elif function_name == LEAKY_RELU_FUNCTION_NAME:
        function_object = K.function(
            [input_object],
            [keras.layers.LeakyReLU(alpha=alpha_parameter)(input_object)]
        )
    else:
        function_object = K.function(
            [input_object],
            [keras.layers.Activation(function_name)(input_object)]
        )

    return function_object([input_values])[0]


def do_2d_pooling(
        feature_matrix, stride_length_px=2, pooling_type_string='max'):
    """Runs 2-D feature maps through pooling filter.

    M = number of rows before pooling
    N = number of columns after pooling
    m = number of rows after pooling
    n = number of columns after pooling

    :param feature_matrix: Input feature maps (numpy array).  Dimensions must be
        M x N x C_i or 1 x M x N x C_i.
    :param stride_length_px: Stride length (pixels).  The pooling window will
        move by this many rows or columns at a time as it slides over each input
        feature map.
    :param pooling_type_string: Pooling type (must be accepted by
        `_check_pooling_type`).
    :return: feature_matrix: Output feature maps (numpy array).  Dimensions will
        be 1 x m x n x C.
    """

    error_checking.assert_is_numpy_array_without_nan(feature_matrix)
    error_checking.assert_is_integer(stride_length_px)
    error_checking.assert_is_geq(stride_length_px, 2)
    _check_pooling_type(pooling_type_string)

    if len(feature_matrix.shape) == 3:
        feature_matrix = numpy.expand_dims(feature_matrix, axis=0)

    error_checking.assert_is_numpy_array(feature_matrix, num_dimensions=4)

    feature_tensor = K.pool2d(
        x=K.variable(feature_matrix), pool_mode=pooling_type_string,
        pool_size=(stride_length_px, stride_length_px),
        strides=(stride_length_px, stride_length_px), padding='valid',
        data_format='channels_last'
    )

    return feature_tensor.eval(session=K.get_session())


def do_batch_normalization(
        feature_matrix, scale_parameter=1., shift_parameter=0.):
    """Performs batch normalization on each feature in the batch.

    :param feature_matrix: E-by-M-by-N-by-C numpy array of feature values.
    :param scale_parameter: Scale parameter (beta in the equation on page 3 of
        Ioffe and Szegedy 2015).
    :param shift_parameter: Shift parameter (gamma in the equation).
    :return: feature_matrix: Feature matrix after batch norm (same dimensions).
    """

    error_checking.assert_is_numpy_array_without_nan(feature_matrix)
    error_checking.assert_is_numpy_array(feature_matrix, num_dimensions=4)
    error_checking.assert_is_greater(scale_parameter, 0.)

    # The following matrices will be M x N x C.
    stdev_matrix = numpy.std(feature_matrix, axis=0, ddof=1)
    mean_matrix = numpy.mean(feature_matrix, axis=0)

    # The following matrices will be E x M x N x C.
    stdev_matrix = numpy.expand_dims(stdev_matrix, axis=0)
    stdev_matrix = numpy.repeat(stdev_matrix, feature_matrix.shape[0], axis=0)
    mean_matrix = numpy.expand_dims(mean_matrix, axis=0)
    mean_matrix = numpy.repeat(mean_matrix, feature_matrix.shape[0], axis=0)

    return shift_parameter + scale_parameter * (
        (feature_matrix - mean_matrix) / (stdev_matrix + K.epsilon())
    )


def _get_activation_layer(function_name, alpha_parameter=0.2):
    """Creates activation layer.

    :param function_name: See doc for `do_activation`.
    :param alpha_parameter: Same.
    :return: layer_object: Instance of `keras.layers.Activation`,
        `keras.layers.ELU`, or `keras.layers.LeakyReLU`.
    """

    if function_name == ELU_FUNCTION_NAME:
        return keras.layers.ELU(alpha=alpha_parameter)

    if function_name == LEAKY_RELU_FUNCTION_NAME:
        return keras.layers.LeakyReLU(alpha=alpha_parameter)

    return keras.layers.Activation(function_name)


def _get_batch_norm_layer():
    """Creates batch-normalization layer.

    :return: layer_object: Instance of `keras.layers.BatchNormalization`.
    """

    return keras.layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)


def _get_2d_pooling_layer(stride_length_px, pooling_type_string):
    """Creates 2-D pooling layer.

    :param stride_length_px: See doc for `do_pooling`.
    :param pooling_type_string: Same.
    :return: layer_object: Instance of `keras.layers.MaxPooling2D` or
        `keras.layers.AveragePooling2D`.
    """

    if pooling_type_string == MAX_POOLING_TYPE_STRING:
        return keras.layers.MaxPooling2D(
            pool_size=(stride_length_px, stride_length_px),
            strides=(stride_length_px, stride_length_px),
            padding='valid', data_format='channels_last'
        )

    return keras.layers.AveragePooling2D(
        pool_size=(stride_length_px, stride_length_px),
        strides=(stride_length_px, stride_length_px),
        padding='valid', data_format='channels_last'
    )


def _get_dense_layer_dimensions(
        num_features, num_predictions, num_dense_layers):
    """Returns dimensions (num input and output features) for each dense layer.

    D = number of dense layers

    :param num_features: Number of features (inputs to the first dense layer).
    :param num_predictions: Number of predictions (outputs from the last dense
        layer).
    :param num_dense_layers: Number of dense layers.
    :return: num_inputs_by_layer: length-D numpy array with number of input
        features per dense layer.
    :return: num_outputs_by_layer: length-D numpy array with number of output
        features per dense layer.
    """

    e_folding_param = (
        float(-1 * num_dense_layers) /
        numpy.log(float(num_predictions) / num_features)
    )

    dense_layer_indices = numpy.linspace(
        0, num_dense_layers - 1, num=num_dense_layers, dtype=float)
    num_inputs_by_layer = num_features * numpy.exp(
        -1 * dense_layer_indices / e_folding_param)
    num_inputs_by_layer = numpy.round(num_inputs_by_layer).astype(int)

    num_outputs_by_layer = numpy.concatenate((
        num_inputs_by_layer[1:],
        numpy.array([num_predictions], dtype=int)
    ))

    return num_inputs_by_layer, num_outputs_by_layer


def example_generator(
        top_input_dir_name, predictor_names, pressure_levels_mb, num_half_rows,
        num_half_columns, num_examples_per_batch):
    """Generates training examples.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (predictors)

    :param top_input_dir_name: Name of top-level input directory.  Training
        files therein will be found by `find_many_training_files`.
    :param predictor_names: See doc for `read_examples`.
    :param pressure_levels_mb: Same.
    :param num_half_rows: Same.
    :param num_half_columns: Same.
    :param num_examples_per_batch: Number of examples per batch.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: target_matrix: E-by-K numpy array of target values (all 0 or 1).
        If target_matrix[i, k] = 1, the [i]th example is in the [k]th class.
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 16)

    example_file_names = find_many_training_files(
        top_training_dir_name=top_input_dir_name, first_batch_number=0,
        last_batch_number=int(1e12)
    )
    random.shuffle(example_file_names)

    num_files = len(example_file_names)
    file_index = 0
    num_examples_in_memory = 0

    batch_indices = numpy.linspace(
        0, num_examples_per_batch - 1, num=num_examples_per_batch, dtype=int)

    while True:
        predictor_matrix = None
        target_matrix = None

        while num_examples_in_memory < num_examples_per_batch:
            print('Reading data from: "{0:s}"...'.format(
                example_file_names[file_index]
            ))

            this_example_dict = read_examples(
                netcdf_file_name=example_file_names[file_index],
                metadata_only=False, predictor_names_to_keep=predictor_names,
                pressure_levels_to_keep_mb=pressure_levels_mb,
                num_half_rows_to_keep=num_half_rows,
                num_half_columns_to_keep=num_half_columns)

            file_index = file_index + 1 if file_index + 1 < num_files else 0

            this_num_examples = len(this_example_dict[VALID_TIMES_KEY])
            if this_num_examples == 0:
                continue

            if target_matrix is None or target_matrix.size == 0:
                predictor_matrix = this_example_dict[PREDICTOR_MATRIX_KEY] + 0.
                target_matrix = this_example_dict[TARGET_MATRIX_KEY] + 0
            else:
                predictor_matrix = numpy.concatenate(
                    (predictor_matrix, this_example_dict[PREDICTOR_MATRIX_KEY]),
                    axis=0
                )
                target_matrix = numpy.concatenate(
                    (target_matrix, this_example_dict[TARGET_MATRIX_KEY]),
                    axis=0
                )

            num_examples_in_memory = target_matrix.shape[0]

        numpy.random.shuffle(batch_indices)
        predictor_matrix = predictor_matrix[batch_indices, ...].astype(
            'float32')
        target_matrix = target_matrix[batch_indices, ...].astype('float64')

        num_examples_by_class = numpy.sum(target_matrix, axis=0)
        print('Number of examples in each class: {0:s}'.format(
            str(num_examples_by_class)
        ))

        num_examples_in_memory = 0
        yield (predictor_matrix, target_matrix)


def train_cnn(
        model_object, output_model_file_name, num_epochs,
        num_training_batches_per_epoch, num_validation_batches_per_epoch,
        training_generator, validation_generator):
    """Trains new CNN.

    In this context "validation" means on-the-fly validation (monitoring during
    training).

    :param model_object: Untrained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param output_model_file_name: Path to output file (will be in HDF5 format,
        so extension should be ".h5").
    :param num_epochs: Number of training epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param training_generator: Generator for training data (created by
        `training_generator`).
    :param validation_generator: Generator for training data (created by
        `validation_generator`).
    """

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 2)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 2)
    error_checking.assert_is_integer(num_validation_batches_per_epoch)
    error_checking.assert_is_geq(num_validation_batches_per_epoch, 2)

    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_model_file_name)

    checkpoint_object = keras.callbacks.ModelCheckpoint(
        output_model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', period=1)

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.005, patience=6, verbose=1, mode='min')

    model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
        verbose=1, callbacks=[checkpoint_object, early_stopping_object],
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch)


def _file_name_to_times(testing_file_name):
    """Parses start/end times from name of testing file.

    :param testing_file_name: See doc for `find_testing_file`.
    :return: first_time_unix_sec: First time in file.
    :return: last_time_unix_sec: Last time in file.
    :raises: ValueError: if times cannot be found in file name.
    """

    pathless_file_name = os.path.split(testing_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    time_strings = extensionless_file_name.split(
        'downsized_3d_examples_'
    )[-1].split('-')

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        time_strings[0], TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        time_strings[-1], TIME_FORMAT
    )

    return first_time_unix_sec, last_time_unix_sec


def find_testing_file(
        top_testing_dir_name, first_time_unix_sec, last_time_unix_sec,
        raise_error_if_missing=True):
    """Locates file with testing examples.

    :param top_testing_dir_name: Name of top-level directory with testing
        examples.
    :param first_time_unix_sec: First time in desired file.
    :param last_time_unix_sec: Last time in desired file.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: testing_file_name: Path to file with testing examples.  If file
        is missing and `raise_error_if_missing = False`, this method will just
        return the expected path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_testing_dir_name)
    error_checking.assert_is_integer(first_time_unix_sec)
    error_checking.assert_is_integer(last_time_unix_sec)
    error_checking.assert_is_geq(last_time_unix_sec, first_time_unix_sec)
    error_checking.assert_is_boolean(raise_error_if_missing)

    testing_file_name = (
        '{0:s}/downsized_3d_examples_{1:s}-{2:s}.nc'
    ).format(
        top_testing_dir_name,
        time_conversion.unix_sec_to_string(first_time_unix_sec, TIME_FORMAT),
        time_conversion.unix_sec_to_string(last_time_unix_sec, TIME_FORMAT)
    )

    if raise_error_if_missing and not os.path.isfile(testing_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            testing_file_name)
        raise ValueError(error_string)

    return testing_file_name


def find_many_testing_files(
        top_testing_dir_name, first_time_unix_sec, last_time_unix_sec):
    """Finds many files with testing examples.

    :param top_testing_dir_name: See doc for `find_testing_file`.
    :param first_time_unix_sec: First desired time.
    :param last_time_unix_sec: Last desired time.
    :return: testing_file_names: 1-D list of paths to testing files.
    :raises: ValueError: if no files are found.
    """

    error_checking.assert_is_string(top_testing_dir_name)
    error_checking.assert_is_integer(first_time_unix_sec)
    error_checking.assert_is_integer(last_time_unix_sec)
    error_checking.assert_is_geq(last_time_unix_sec, first_time_unix_sec)

    testing_file_pattern = (
        '{0:s}/downsized_3d_examples_{1:s}-{1:s}.nc'
    ).format(top_testing_dir_name, TIME_FORMAT_REGEX, TIME_FORMAT_REGEX)

    testing_file_names = glob.glob(testing_file_pattern)

    if len(testing_file_names) == 0:
        error_string = 'Cannot find any files with the pattern: "{0:s}"'.format(
            testing_file_names)
        raise ValueError(error_string)

    testing_file_names.sort()

    file_start_times_unix_sec = numpy.array(
        [_file_name_to_times(f)[0] for f in testing_file_names],
        dtype=int
    )
    file_end_times_unix_sec = numpy.array(
        [_file_name_to_times(f)[1] for f in testing_file_names],
        dtype=int
    )

    good_indices = numpy.where(numpy.invert(numpy.logical_or(
        file_start_times_unix_sec > last_time_unix_sec,
        file_end_times_unix_sec < first_time_unix_sec
    )))[0]

    if len(good_indices) == 0:
        error_string = (
            'Cannot find any files with time from {0:s} to {1:s}.'
        ).format(
            time_conversion.unix_sec_to_string(
                first_time_unix_sec, TIME_FORMAT),
            time_conversion.unix_sec_to_string(
                last_time_unix_sec, TIME_FORMAT)
        )
        raise ValueError(error_string)

    return [testing_file_names[i] for i in good_indices]


def model_to_grid_dimensions(model_object):
    """Reads grid dimensions from CNN.

    :param model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :return: num_half_rows: Number of rows in half-grid.
    :return: num_half_columns: Number of columns in half-grid.
    """

    input_dimensions = numpy.array(
        model_object.layers[0].input.get_shape().as_list()[1:], dtype=int
    )

    num_half_rows = int(numpy.round(
        (input_dimensions[0] - 1) / 2
    ))
    num_half_columns = int(numpy.round(
        (input_dimensions[1] - 1) / 2
    ))

    return num_half_rows, num_half_columns


def make_predictions(model_object, testing_file_names, predictor_names,
                     pressure_levels_mb):
    """Uses a trained CNN to make predictions.

    :param model_object: Trained CNN (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param testing_file_names: 1-D list of paths to testing files (will be read
        by `read_examples`).
    :param predictor_names: See doc for `read_examples`.
    :param pressure_levels_mb: Same.
    :return: class_probability_matrix: E-by-3 numpy array of predicted
        probabilities.  class_probability_matrix[i, k] is the predicted
        probability that the [i]th example belongs to the [k]th class.
    :return: target_values: length-E numpy array of target values (integers).
        Possible values are `NO_FRONT_ENUM`, `WARM_FRONT_ENUM`, and
        `WARM_FRONT_ENUM`, listed at the top of this notebook.
    """

    num_half_rows, num_half_columns = model_to_grid_dimensions(model_object)

    class_probability_matrix = None
    target_values = None

    for this_file_name in testing_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = read_examples(
            netcdf_file_name=this_file_name,
            predictor_names_to_keep=predictor_names,
            pressure_levels_to_keep_mb=pressure_levels_mb,
            num_half_rows_to_keep=num_half_rows,
            num_half_columns_to_keep=num_half_columns)

        these_target_values = numpy.argmax(
            this_example_dict[TARGET_MATRIX_KEY], axis=1
        )
        this_num_examples = len(these_target_values)

        print('Making predictions for all {0:d} examples in the file...'.format(
            this_num_examples
        ))

        this_probability_matrix = model_object.predict(
            this_example_dict[PREDICTOR_MATRIX_KEY],
            batch_size=this_num_examples
        )

        if class_probability_matrix is None:
            class_probability_matrix = this_probability_matrix + 0.
            target_values = these_target_values + 0.
        else:
            class_probability_matrix = numpy.concatenate(
                (class_probability_matrix, this_probability_matrix), axis=0
            )
            target_values = numpy.concatenate((
                target_values, these_target_values
            ))

    target_values = numpy.round(target_values).astype(int)
    return class_probability_matrix, target_values


def plot_3class_contingency_table(contingency_matrix, axes_object=None):
    """Plots 3-class contingency table.

    :param contingency_matrix: 3-by-3 numpy array.
    :param axes_object: See doc for `plot_feature_map`.
    :return: axes_object: See doc for `plot_feature_map`.
    """

    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    colour_map_object = pyplot.get_cmap('binary')
    text_colour = numpy.array([31, 120, 180], dtype=float) / 255

    pyplot.imshow(
        contingency_matrix, axes=axes_object, origin='upper',
        cmap=colour_map_object, vmin=numpy.max(contingency_matrix),
        vmax=numpy.max(contingency_matrix) + 1
    )

    for i in range(contingency_matrix.shape[0]):
        for j in range(contingency_matrix.shape[1]):
            axes_object.text(
                i, j, contingency_matrix[j, i], fontsize=50, color=text_colour,
                horizontalalignment='center', verticalalignment='center')

    tick_locations = numpy.array([0, 1, 2], dtype=int)
    tick_labels = ['NF', 'WF', 'CF']

    pyplot.xticks(tick_locations, tick_labels)
    pyplot.xlabel('Actual')
    pyplot.yticks(tick_locations, tick_labels)
    pyplot.ylabel('Predicted')

    return axes_object


def plot_2class_contingency_table(contingency_matrix, axes_object=None):
    """Plots 2-class contingency table.

    :param contingency_matrix: 2-by-2 numpy array.
    :param axes_object: See doc for `plot_feature_map`.
    :return: axes_object: See doc for `plot_feature_map`.
    """

    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    colour_map_object = pyplot.get_cmap('binary')
    text_colour = numpy.array([31, 120, 180], dtype=float) / 255

    pyplot.imshow(
        contingency_matrix, axes=axes_object, origin='upper',
        cmap=colour_map_object, vmin=numpy.max(contingency_matrix),
        vmax=numpy.max(contingency_matrix) + 1
    )

    for i in range(contingency_matrix.shape[0]):
        for j in range(contingency_matrix.shape[1]):
            axes_object.text(
                i, j, contingency_matrix[j, i], fontsize=50, color=text_colour,
                horizontalalignment='center', verticalalignment='center')

    tick_locations = numpy.array([0, 1], dtype=int)
    tick_labels = ['Yes', 'No']

    pyplot.xticks(tick_locations, tick_labels)
    pyplot.xlabel('Actual')
    pyplot.yticks(tick_locations, tick_labels)
    pyplot.ylabel('Predicted')

    return axes_object


def get_saliency_maps(model_object, target_class, predictor_matrix):
    """Computes saliency map for each example in `predictor_matrix`.

    :param model_object: Trained CNN (instance of `keras.models`).  Saliency
        will be computed for this CNN only.  Different models give different
        answers.
    :param target_class: Target class (integer).  Possible values are
        `NO_FRONT_ENUM`, `WARM_FRONT_ENUM`, and `WARM_FRONT_ENUM`, listed at the
        top of this notebook.
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: saliency_matrix: E-by-M-by-N-by-C numpy array of saliency values.
    """

    loss_tensor = K.mean(
        (model_object.layers[-1].output[..., target_class] - 1) ** 2
    )

    gradient_tensor = K.gradients(
        loss_tensor, [model_object.input]
    )[0]
    gradient_tensor = gradient_tensor / K.maximum(
        K.std(gradient_tensor), K.epsilon()
    )

    inputs_to_gradients_function = K.function(
        [model_object.input, K.learning_phase()], [gradient_tensor]
    )

    saliency_matrix = inputs_to_gradients_function(
        [predictor_matrix, 0]
    )[0]

    return -1 * saliency_matrix


def _plot_saliency_one_field(
        saliency_matrix, axes_object, max_absolute_contour_value,
        contour_interval, colour_map_object=SALIENCY_COLOUR_MAP_OBJECT):
    """Plots saliency map for one 2-D field.

    M = number of rows in grid
    N = number of columns in grid

    :param saliency_matrix: M-by-N numpy array of saliency values.
    :param axes_object: Existing axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param max_absolute_contour_value: Max absolute saliency value to plot.
    :param contour_interval: Interval between successive saliency contours.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    """

    error_checking.assert_is_geq(max_absolute_contour_value, 0.)
    max_absolute_contour_value = numpy.maximum(max_absolute_contour_value, 1e-3)

    error_checking.assert_is_geq(contour_interval, 0.)
    contour_interval = numpy.maximum(contour_interval, 1e-4)
    error_checking.assert_is_less_than(
        contour_interval, max_absolute_contour_value)

    num_rows = saliency_matrix.shape[0]
    num_columns = saliency_matrix.shape[1]
    unique_y_coords = -0.5 + numpy.linspace(1, num_rows, num=num_rows)
    unique_x_coords = -0.5 + numpy.linspace(1, num_columns, num=num_columns)

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(
        unique_x_coords, unique_y_coords)

    half_num_contours = int(numpy.round(
        1 + max_absolute_contour_value / contour_interval
    ))

    # Plot positive values.
    these_contour_values = numpy.linspace(
        0., max_absolute_contour_value, num=half_num_contours)

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, saliency_matrix,
        these_contour_values, cmap=colour_map_object,
        vmin=numpy.min(these_contour_values),
        vmax=numpy.max(these_contour_values),
        linewidths=2, linestyles='solid', zorder=1e6)

    # Plot negative values.
    these_contour_values = these_contour_values[1:]

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, -saliency_matrix,
        these_contour_values, cmap=colour_map_object,
        vmin=numpy.min(these_contour_values),
        vmax=numpy.max(these_contour_values),
        linewidths=2, linestyles='dashed', zorder=1e6)


def plot_saliency_one_example(
        predictor_matrix, saliency_matrix, predictor_names):
    """Plots saliency maps for one example.

    :param predictor_matrix: M-by-N-by-C numpy array of predictor values.
    :param saliency_matrix: M-by-N-by-C numpy array of saliency values.
    :param predictor_names: length-C list with names of predictor variables.
    :return: figure_object: See doc for `create_paneled_figure`.
    :return: axes_object_matrix: Same.
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    error_checking.assert_is_numpy_array(predictor_matrix, num_dimensions=3)

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix)
    error_checking.assert_is_numpy_array(
        saliency_matrix, exact_dimensions=numpy.array(predictor_matrix.shape)
    )

    num_predictors = predictor_matrix.shape[-1]
    expected_dim = numpy.array([num_predictors], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(predictor_names), exact_dimensions=expected_dim
    )

    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_predictors)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_predictors) / num_panel_rows
    ))

    figure_object, axes_object_matrix = create_paneled_figure(
        num_rows=num_panel_rows, num_columns=num_panel_columns,
        horizontal_spacing=0.2, vertical_spacing=0.2)

    max_absolute_saliency = numpy.percentile(
        numpy.absolute(saliency_matrix), 99.
    )

    for k in range(num_predictors):
        this_axes_object = numpy.ravel(axes_object_matrix)[k]
        plot_feature_map(
            feature_matrix=predictor_matrix[..., k],
            axes_object=this_axes_object
        )

        _plot_saliency_one_field(
            saliency_matrix=saliency_matrix[..., k],
            axes_object=this_axes_object,
            max_absolute_contour_value=max_absolute_saliency,
            contour_interval=max_absolute_saliency / 10)

        this_axes_object.set_title(predictor_names[k], fontsize=20)

    return figure_object, axes_object_matrix


def apply_median_filter(input_matrix_2d, num_cells_in_half_window):
    """Applies median filter to 2-D field.

    M = number of rows in grid
    N = number of columns in grid

    :param input_matrix_2d: M-by-N numpy array of raw (unfiltered) data.
    :param num_cells_in_half_window: Number of grid cells in half-window for
        smoothing filter.
    :return: output_matrix_2d: M-by-N numpy array of filtered data.
    """

    error_checking.assert_is_integer(num_cells_in_half_window)
    error_checking.assert_is_geq(num_cells_in_half_window, 1)

    return median_filter(
        input_matrix_2d, size=2 * num_cells_in_half_window + 1, mode='reflect',
        origin=0)


def _upsample_cam(class_activation_matrix, num_target_rows, num_target_columns):
    """Upsamples class-activation map (CAM) to new dimensions.

    m = number of rows in original grid
    n = number of columns in original grid
    M = number of rows in new grid
    N = number of columns in new grid

    :param class_activation_matrix: m-by-n numpy array of class activations.
    :param num_target_rows: Number of rows in new (target) grid.
    :param num_target_columns: Number of columns in new (target) grid.
    :return: class_activation_matrix: M-by-N numpy array of class activations.
    """

    row_indices_new = numpy.linspace(
        1, num_target_rows, num=num_target_rows, dtype=float
    )
    row_indices_orig = numpy.linspace(
        1, num_target_rows, num=class_activation_matrix.shape[0], dtype=float
    )

    column_indices_new = numpy.linspace(
        1, num_target_columns, num=num_target_columns, dtype=float
    )
    column_indices_orig = numpy.linspace(
        1, num_target_columns, num=class_activation_matrix.shape[1], dtype=float
    )

    interp_object = RectBivariateSpline(
        x=row_indices_orig, y=column_indices_orig, z=class_activation_matrix,
        kx=1, ky=1, s=0)

    return interp_object(x=row_indices_new, y=column_indices_new, grid=True)


def run_gradcam(model_object, predictor_matrix, target_class,
                target_layer_name):
    """Runs Grad-CAM.

    :param model_object: See doc for `get_saliency_maps`.
    :param predictor_matrix: Same.
    :param target_class: Same.
    :param target_layer_name: Name of target layer.  Class-activation map will
        be computed for this layer.
    :return: class_activation_matrix: M-by-N numpy array of class activations.
    """

    # Set up tensors.
    loss_tensor = model_object.layers[-1].input[..., target_class]
    activation_tensor = model_object.get_layer(name=target_layer_name).output
    gradient_tensor = tensorflow.gradients(
        loss_tensor, [activation_tensor]
    )[0]

    root_mean_square_tensor = K.sqrt(K.mean(K.square(gradient_tensor)))
    gradient_tensor = gradient_tensor / (root_mean_square_tensor + K.epsilon())

    gradient_function = K.function(
        [model_object.input], [activation_tensor, gradient_tensor]
    )

    # Compute activation and gradient matrices for target layer.
    activation_matrix, gradient_matrix = gradient_function([predictor_matrix])
    activation_matrix = activation_matrix[0, ...]
    gradient_matrix = gradient_matrix[0, ...]

    # Compute class-activation map.
    print(gradient_matrix.shape)
    mean_weight_by_filter = numpy.mean(gradient_matrix, axis=(0, 1))

    class_activation_matrix = numpy.ones(activation_matrix.shape[:-1])
    num_filters = len(mean_weight_by_filter)

    for k in range(num_filters):
        class_activation_matrix += (
            mean_weight_by_filter[k] * activation_matrix[..., k]
        )

    # Upsample class-activation map to input dimensions.
    class_activation_matrix = _upsample_cam(
        class_activation_matrix=class_activation_matrix,
        num_target_rows=predictor_matrix.shape[1],
        num_target_columns=predictor_matrix.shape[2]
    )

    class_activation_matrix = numpy.maximum(class_activation_matrix, 0.)
    return class_activation_matrix


def _plot_class_activn_one_field(
        class_activation_matrix, axes_object, max_contour_value,
        contour_interval, colour_map_object=SALIENCY_COLOUR_MAP_OBJECT):
    """Plots class-activation map for one 2-D field.

    M = number of rows in grid
    N = number of columns in grid

    :param class_activation_matrix: M-by-N numpy array of saliency values.
    :param axes_object: See doc for `_plot_saliency_one_field`.
    :param max_contour_value: Max value to plot.
    :param contour_interval: See doc for `_plot_saliency_one_field`.
    :param colour_map_object: Same.
    """

    error_checking.assert_is_geq(max_contour_value, 0.)
    max_contour_value = numpy.maximum(max_contour_value, 1e-3)

    error_checking.assert_is_geq(contour_interval, 0.)
    contour_interval = numpy.maximum(contour_interval, 1e-4)
    error_checking.assert_is_less_than(contour_interval, max_contour_value)

    num_rows = class_activation_matrix.shape[0]
    num_columns = class_activation_matrix.shape[1]
    unique_y_coords = -0.5 + numpy.linspace(1, num_rows, num=num_rows)
    unique_x_coords = -0.5 + numpy.linspace(1, num_columns, num=num_columns)

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(
        unique_x_coords, unique_y_coords)

    num_contours = int(numpy.round(
        1 + max_contour_value / contour_interval
    ))
    contour_values = numpy.linspace(0., max_contour_value, num=num_contours)

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, class_activation_matrix,
        contour_values, cmap=colour_map_object,
        vmin=numpy.min(contour_values), vmax=numpy.max(contour_values),
        linewidths=2, linestyles='solid', zorder=1e6)


def plot_class_activn_one_example(
        predictor_matrix, class_activation_matrix, predictor_names):
    """Plots saliency maps for one example.

    :param predictor_matrix: M-by-N-by-C numpy array of predictor values.
    :param class_activation_matrix: M-by-N numpy array of class activations.
    :param predictor_names: length-C list with names of predictor variables.
    :return: figure_object: See doc for `create_paneled_figure`.
    :return: axes_object_matrix: Same.
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    error_checking.assert_is_numpy_array(predictor_matrix, num_dimensions=3)

    error_checking.assert_is_geq_numpy_array(class_activation_matrix, 0.)
    error_checking.assert_is_numpy_array(
        class_activation_matrix,
        exact_dimensions=numpy.array(predictor_matrix.shape[:-1])
    )

    num_predictors = predictor_matrix.shape[-1]
    expected_dim = numpy.array([num_predictors], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(predictor_names), exact_dimensions=expected_dim
    )

    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_predictors)
    ))
    num_panel_columns = 1 + int(numpy.ceil(
        float(num_predictors) / num_panel_rows
    ))

    column_widths = numpy.concatenate((
        numpy.full(num_panel_columns - 1, 5), numpy.array([1])
    ))

    figure_object, axes_object_matrix = create_paneled_figure(
        num_rows=num_panel_rows, num_columns=num_panel_columns,
        horizontal_spacing=0.2, vertical_spacing=0.2,
        grid_spec_dict={'width_ratios': column_widths}
    )

    for i in range(num_panel_rows):
        axes_object_matrix[i, -1].axis('off')

    max_activation = numpy.percentile(class_activation_matrix, 99.)

    for k in range(num_predictors):
        this_axes_object = numpy.ravel(axes_object_matrix[:, :-1])[k]
        plot_feature_map(
            feature_matrix=predictor_matrix[..., k],
            axes_object=this_axes_object
        )

        _plot_class_activn_one_field(
            class_activation_matrix=class_activation_matrix,
            axes_object=this_axes_object, max_contour_value=max_activation,
            contour_interval=max_activation / 10,
            colour_map_object=SALIENCY_COLOUR_MAP_OBJECT)

        this_axes_object.set_title(predictor_names[k], fontsize=20)

    colour_bar_object = _add_colour_bar(
        axes_object=axes_object_matrix[:, -1],
        colour_map_object=SALIENCY_COLOUR_MAP_OBJECT,
        values_to_colour=class_activation_matrix, min_colour_value=0.,
        max_colour_value=max_activation, orientation_string='vertical',
        extend_min=False, extend_max=True, fraction_of_axis_length=2.)

    colour_bar_object.set_label('Class activation')

    return figure_object, axes_object_matrix


def _run():
    """Normal Python version of notebook.

    This is effectively the main method.
    """

    # Example: Finding training files
    first_batch_number = 0
    last_batch_number = 1000

    training_example_file_names = find_many_training_files(
        top_training_dir_name=TOP_TRAINING_DIR_NAME,
        first_batch_number=first_batch_number,
        last_batch_number=last_batch_number)

    for this_file_name in training_example_file_names:
        print(this_file_name)

    # Example: Reading metadata from training file
    batch_number = 179

    training_example_file_name = find_training_file(
        top_training_dir_name=TOP_TRAINING_DIR_NAME, batch_number=batch_number,
        raise_error_if_missing=True)

    training_example_dict = read_examples(
        netcdf_file_name=training_example_file_name, metadata_only=True)

    for this_key in training_example_dict:
        if this_key not in METADATA_KEYS_TO_REPORT:
            continue

        print('{0:s} ... {1:s}\n'.format(
            this_key, str(training_example_dict[this_key])
        ))

    # Example: Reading everything from training file
    predictor_names = [
        U_WIND_GRID_RELATIVE_NAME, V_WIND_GRID_RELATIVE_NAME,
        TEMPERATURE_NAME, SPECIFIC_HUMIDITY_NAME
    ]

    pressure_levels_mb = numpy.array([
        DUMMY_SURFACE_PRESSURE_MB, DUMMY_SURFACE_PRESSURE_MB,
        DUMMY_SURFACE_PRESSURE_MB, DUMMY_SURFACE_PRESSURE_MB
    ], dtype=int)

    training_example_dict = read_examples(
        netcdf_file_name=training_example_file_name, metadata_only=False,
        predictor_names_to_keep=predictor_names,
        pressure_levels_to_keep_mb=pressure_levels_mb)

    print('Shape of predictor matrix = {0:s}'.format(
        str(training_example_dict[PREDICTOR_MATRIX_KEY].shape)
    ))
    print('Shape of target matrix = {0:s}'.format(
        str(training_example_dict[TARGET_MATRIX_KEY].shape)
    ))

    # Understanding the Target Matrix
    target_matrix = training_example_dict[TARGET_MATRIX_KEY]
    print('Target matrix:\n{0:s}\n'.format(
        str(target_matrix)
    ))

    row_sums = numpy.sum(training_example_dict[TARGET_MATRIX_KEY], axis=1)
    print('Sum across each row (proves MECE property):\n{0:s}\n'.format(
        str(row_sums)
    ))

    num_examples_by_class = numpy.sum(
        training_example_dict[TARGET_MATRIX_KEY], axis=0
    ).astype(int)

    print((
        'Number of examples with no front = {0:d} ... warm front = {1:d} ... '
        'cold front = {2:d}'
    ).format(
        num_examples_by_class[NO_FRONT_ENUM],
        num_examples_by_class[WARM_FRONT_ENUM],
        num_examples_by_class[COLD_FRONT_ENUM]
    ))

    # Understanding the Predictor Matrix
    predictor_matrix = training_example_dict[PREDICTOR_MATRIX_KEY]

    for k in range(len(predictor_names)):
        this_mean = training_example_dict[FIRST_NORM_PARAM_KEY][0, k]
        this_stdev = training_example_dict[SECOND_NORM_PARAM_KEY][0, k]

        this_normalized_matrix = predictor_matrix[0, :5, :5, k]
        this_denorm_matrix = this_mean + this_stdev * this_normalized_matrix

        print((
            'Normalized {0:s} in southwest corner of first example:\n{1:s}\n'
        ).format(
            predictor_names[k], str(this_normalized_matrix)
        ))

        print((
            '*De*normalized {0:s} in southwest corner of first example:'
            '\n{1:s}\n'
        ).format(
            predictor_names[k], str(this_denorm_matrix)
        ))

    # Example 1 of random convolution
    num_kernel_rows = 3
    num_kernel_columns = 3

    predictor_matrix = training_example_dict[PREDICTOR_MATRIX_KEY]
    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    temperature_matrix = predictor_matrix[8, ..., temperature_index]

    _, axes_object_matrix = create_paneled_figure(num_rows=1, num_columns=2)
    
    plot_feature_map(
        feature_matrix=temperature_matrix, axes_object=axes_object_matrix[0, 0]
    )
    axes_object_matrix[0, 0].set_title('Before convolution')

    feature_matrix = numpy.expand_dims(temperature_matrix, axis=-1)
    kernel_matrix = numpy.random.uniform(
        low=0., high=1., size=(num_kernel_rows, num_kernel_columns, 1, 1)
    )

    feature_matrix = do_2d_convolution(
        feature_matrix=feature_matrix, kernel_matrix=kernel_matrix,
        pad_edges=True, stride_length_px=1)
    feature_matrix = feature_matrix[0, ..., 0]

    print('Shape of output map = {0:s}'.format(
        str(feature_matrix.shape)
    ))

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1]
    )
    axes_object_matrix[0, 1].set_title('After convolution')

    # Example 2 of random convolution
    num_kernel_rows = 3
    num_kernel_columns = 3

    predictor_matrix = training_example_dict[PREDICTOR_MATRIX_KEY]
    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    temperature_matrix = predictor_matrix[8, ..., temperature_index]

    _, axes_object_matrix = create_paneled_figure(num_rows=1, num_columns=2)

    plot_feature_map(
        feature_matrix=temperature_matrix, axes_object=axes_object_matrix[0, 0]
    )
    axes_object_matrix[0, 0].set_title('Before convolution')

    feature_matrix = numpy.expand_dims(temperature_matrix, axis=-1)
    kernel_matrix = numpy.random.uniform(
        low=0., high=1., size=(num_kernel_rows, num_kernel_columns, 1, 1)
    )

    feature_matrix = do_2d_convolution(
        feature_matrix=feature_matrix, kernel_matrix=kernel_matrix,
        pad_edges=False, stride_length_px=1)
    feature_matrix = feature_matrix[0, ..., 0]

    print('Shape of output map = {0:s}'.format(
        str(feature_matrix.shape)
    ))

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1]
    )
    axes_object_matrix[0, 1].set_title('After convolution')

    # Example 3 of random convolution
    num_kernel_rows = 3
    num_kernel_columns = 3

    predictor_matrix = training_example_dict[PREDICTOR_MATRIX_KEY]
    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    temperature_matrix = predictor_matrix[8, ..., temperature_index]

    _, axes_object_matrix = create_paneled_figure(num_rows=1, num_columns=2)

    plot_feature_map(
        feature_matrix=temperature_matrix, axes_object=axes_object_matrix[0, 0]
    )
    axes_object_matrix[0, 0].set_title('Before convolution')

    feature_matrix = numpy.expand_dims(temperature_matrix, axis=-1)
    kernel_matrix = numpy.random.uniform(
        low=0., high=1., size=(num_kernel_rows, num_kernel_columns, 1, 1)
    )

    feature_matrix = do_2d_convolution(
        feature_matrix=feature_matrix, kernel_matrix=kernel_matrix,
        pad_edges=False, stride_length_px=2)
    feature_matrix = feature_matrix[0, ..., 0]

    print('Shape of output map = {0:s}'.format(
        str(feature_matrix.shape)
    ))

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1]
    )
    axes_object_matrix[0, 1].set_title('After convolution')

    # Example 1 of edge detection
    predictor_matrix = training_example_dict[PREDICTOR_MATRIX_KEY]
    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    temperature_matrix = predictor_matrix[8, ..., temperature_index]

    _, axes_object_matrix = create_paneled_figure(num_rows=1, num_columns=2)

    plot_feature_map(
        feature_matrix=temperature_matrix, axes_object=axes_object_matrix[0, 0]
    )
    axes_object_matrix[0, 0].set_title('Before convolution')

    feature_matrix = numpy.expand_dims(temperature_matrix, axis=-1)
    kernel_matrix = numpy.expand_dims(EDGE_DETECTOR_MATRIX1, axis=-1)
    kernel_matrix = numpy.expand_dims(kernel_matrix, axis=-1)

    feature_matrix = do_2d_convolution(
        feature_matrix=feature_matrix, kernel_matrix=kernel_matrix,
        pad_edges=False, stride_length_px=1)
    feature_matrix = feature_matrix[0, ..., 0]

    print('Shape of output map = {0:s}'.format(
        str(feature_matrix.shape)
    ))

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1]
    )
    axes_object_matrix[0, 1].set_title('After convolution')

    # Example 2 of edge detection
    predictor_matrix = training_example_dict[PREDICTOR_MATRIX_KEY]
    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    temperature_matrix = predictor_matrix[8, ..., temperature_index]

    _, axes_object_matrix = create_paneled_figure(num_rows=1, num_columns=2)

    plot_feature_map(
        feature_matrix=temperature_matrix, axes_object=axes_object_matrix[0, 0]
    )
    axes_object_matrix[0, 0].set_title('Before convolution')

    feature_matrix = numpy.expand_dims(temperature_matrix, axis=-1)
    kernel_matrix = numpy.expand_dims(EDGE_DETECTOR_MATRIX2, axis=-1)
    kernel_matrix = numpy.expand_dims(kernel_matrix, axis=-1)

    feature_matrix = do_2d_convolution(
        feature_matrix=feature_matrix, kernel_matrix=kernel_matrix,
        pad_edges=False, stride_length_px=1)
    feature_matrix = feature_matrix[0, ..., 0]

    print('Shape of output map = {0:s}'.format(
        str(feature_matrix.shape)
    ))

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1]
    )
    axes_object_matrix[0, 1].set_title('After convolution')

    # Example 3 of edge detection
    predictor_matrix = training_example_dict[PREDICTOR_MATRIX_KEY]
    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    temperature_matrix = predictor_matrix[8, ..., temperature_index]

    _, axes_object_matrix = create_paneled_figure(num_rows=1, num_columns=2)

    plot_feature_map(
        feature_matrix=temperature_matrix, axes_object=axes_object_matrix[0, 0]
    )
    axes_object_matrix[0, 0].set_title('Before convolution')

    feature_matrix = numpy.expand_dims(temperature_matrix, axis=-1)
    kernel_matrix = numpy.expand_dims(EDGE_DETECTOR_MATRIX3, axis=-1)
    kernel_matrix = numpy.expand_dims(kernel_matrix, axis=-1)

    feature_matrix = do_2d_convolution(
        feature_matrix=feature_matrix, kernel_matrix=kernel_matrix,
        pad_edges=False, stride_length_px=1)
    feature_matrix = feature_matrix[0, ..., 0]

    print('Shape of output map = {0:s}'.format(
        str(feature_matrix.shape)
    ))

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1]
    )
    axes_object_matrix[0, 1].set_title('After convolution')

    # Multi-channel Convolution
    temperature_matrix = predictor_matrix[
        8, ..., predictor_names.index(TEMPERATURE_NAME)
    ]
    u_wind_matrix = predictor_matrix[
        8, ..., predictor_names.index(U_WIND_GRID_RELATIVE_NAME)
    ]
    v_wind_matrix = predictor_matrix[
        8, ..., predictor_names.index(V_WIND_GRID_RELATIVE_NAME)
    ]

    _, axes_object_matrix = create_paneled_figure(
        num_rows=2, num_columns=2,
        horizontal_spacing=0.2, vertical_spacing=0.2)

    plot_feature_map(
        feature_matrix=temperature_matrix, axes_object=axes_object_matrix[0, 0]
    )
    plot_wind_barbs(
        u_wind_matrix=u_wind_matrix, v_wind_matrix=v_wind_matrix,
        axes_object=axes_object_matrix[0, 0]
    )
    axes_object_matrix[0, 0].set_title('Before convolution')

    kernel_matrix_to_channel0 = numpy.repeat(
        a=numpy.expand_dims(EDGE_DETECTOR_MATRIX1, axis=-1), repeats=3, axis=-1
    )
    kernel_matrix_to_channel1 = numpy.repeat(
        a=numpy.expand_dims(EDGE_DETECTOR_MATRIX2, axis=-1), repeats=3, axis=-1
    )
    kernel_matrix_to_channel2 = numpy.repeat(
        a=numpy.expand_dims(EDGE_DETECTOR_MATRIX3, axis=-1), repeats=3, axis=-1
    )
    kernel_matrix = numpy.stack((
        kernel_matrix_to_channel0, kernel_matrix_to_channel1,
        kernel_matrix_to_channel2
    ), axis=-1)

    feature_matrix = numpy.stack(
        (temperature_matrix, u_wind_matrix, v_wind_matrix), axis=-1
    )
    feature_matrix = do_2d_convolution(
        feature_matrix=feature_matrix, kernel_matrix=kernel_matrix,
        pad_edges=False, stride_length_px=1)

    feature_matrix = feature_matrix[0, ...]
    num_output_channels = feature_matrix.shape[-1]

    for k in range(num_output_channels):
        this_axes_object = numpy.ravel(axes_object_matrix)[k + 1]
        plot_feature_map(
            feature_matrix=feature_matrix[..., k], axes_object=this_axes_object
        )

        this_axes_object.set_title('Filter {0:d} after convolution'.format(k))

    # Standard Activation Functions
    function_names_keras = [
        SIGMOID_FUNCTION_NAME, TANH_FUNCTION_NAME, RELU_FUNCTION_NAME
    ]
    function_names_fancy = ['Sigmoid', 'tanh', 'ReLU']
    input_values = numpy.linspace(-3, 3, num=1000, dtype=float)

    colour_by_function = numpy.array([
        [27, 158, 119],
        [217, 95, 2],
        [117, 112, 179]
    ])
    colour_by_function = colour_by_function.astype(float) / 255

    _, axes_object_matrix = create_paneled_figure(num_rows=1, num_columns=1)
    axes_object = axes_object_matrix[0, 0]

    axes_object.plot(
        input_values, numpy.zeros(input_values.shape),
        linewidth=2, linestyle='dashed', color=numpy.full(3, 152. / 255)
    )

    for i in range(len(function_names_keras)):
        these_output_values = do_activation(
            input_values=input_values, function_name=function_names_keras[i]
        )

        axes_object.plot(
            input_values, these_output_values, linewidth=4, linestyle='solid',
            color=colour_by_function[i, :], label=function_names_fancy[i]
        )

    axes_object.legend(loc='upper left')

    # Variants of ReLU
    function_names_keras = [
        SELU_FUNCTION_NAME, ELU_FUNCTION_NAME, LEAKY_RELU_FUNCTION_NAME
    ]
    function_names_fancy = ['SeLU', 'eLU', 'Leaky ReLU']
    input_values = numpy.linspace(-3, 3, num=1000, dtype=float)

    colour_by_function = numpy.array([
        [27, 158, 119],
        [217, 95, 2],
        [117, 112, 179]
    ])
    colour_by_function = colour_by_function.astype(float) / 255

    _, axes_object_matrix = create_paneled_figure(num_rows=1, num_columns=1)
    axes_object = axes_object_matrix[0, 0]

    axes_object.plot(
        input_values, numpy.zeros(input_values.shape),
        linewidth=2, linestyle='dashed', color=numpy.full(3, 152. / 255)
    )

    for i in range(len(function_names_keras)):
        these_output_values = do_activation(
            input_values=input_values, function_name=function_names_keras[i]
        )

        axes_object.plot(
            input_values, these_output_values, linewidth=4, linestyle='solid',
            color=colour_by_function[i, :], label=function_names_fancy[i]
        )

    axes_object.legend(loc='upper left')

    # Example 1 of pooling
    predictor_matrix = training_example_dict[PREDICTOR_MATRIX_KEY]
    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    temperature_matrix = predictor_matrix[8, ..., temperature_index]

    feature_matrix = numpy.expand_dims(temperature_matrix, axis=-1)
    feature_matrix = do_2d_pooling(
        feature_matrix=feature_matrix, stride_length_px=2,
        pooling_type_string=MAX_POOLING_TYPE_STRING)
    feature_matrix = feature_matrix[0, ..., 0]

    all_values = numpy.concatenate((
        numpy.ravel(temperature_matrix), numpy.ravel(feature_matrix)
    ))
    max_colour_value = numpy.percentile(numpy.absolute(all_values), 99.)
    min_colour_value = -1 * max_colour_value

    _, axes_object_matrix = create_paneled_figure(num_rows=1, num_columns=2)

    plot_feature_map(
        feature_matrix=temperature_matrix, axes_object=axes_object_matrix[0, 0],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[0, 0].set_title('Before max-pooling')

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[0, 1].set_title('After max-pooling')

    # Example 2 of pooling
    predictor_matrix = training_example_dict[PREDICTOR_MATRIX_KEY]
    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    temperature_matrix = predictor_matrix[8, ..., temperature_index]

    feature_matrix = numpy.expand_dims(temperature_matrix, axis=-1)
    feature_matrix = do_2d_pooling(
        feature_matrix=feature_matrix, stride_length_px=2,
        pooling_type_string=MEAN_POOLING_TYPE_STRING)
    feature_matrix = feature_matrix[0, ..., 0]

    all_values = numpy.concatenate((
        numpy.ravel(temperature_matrix), numpy.ravel(feature_matrix)
    ))
    max_colour_value = numpy.percentile(numpy.absolute(all_values), 99.)
    min_colour_value = -1 * max_colour_value

    _, axes_object_matrix = create_paneled_figure(num_rows=1, num_columns=2)

    plot_feature_map(
        feature_matrix=temperature_matrix, axes_object=axes_object_matrix[0, 0],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[0, 0].set_title('Before mean-pooling')

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[0, 1].set_title('After mean-pooling')

    # Example 3 of pooling
    predictor_matrix = training_example_dict[PREDICTOR_MATRIX_KEY]
    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    temperature_matrix = predictor_matrix[8, ..., temperature_index]

    feature_matrix = numpy.expand_dims(temperature_matrix, axis=-1)
    feature_matrix = do_2d_pooling(
        feature_matrix=feature_matrix, stride_length_px=4,
        pooling_type_string=MAX_POOLING_TYPE_STRING)
    feature_matrix = feature_matrix[0, ..., 0]

    all_values = numpy.concatenate((
        numpy.ravel(temperature_matrix), numpy.ravel(feature_matrix)
    ))
    max_colour_value = numpy.percentile(numpy.absolute(all_values), 99.)
    min_colour_value = -1 * max_colour_value

    _, axes_object_matrix = create_paneled_figure(num_rows=1, num_columns=2)

    plot_feature_map(
        feature_matrix=temperature_matrix, axes_object=axes_object_matrix[0, 0],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[0, 0].set_title('Before max-pooling')

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[0, 1].set_title('After max-pooling')

    # Example 1 of convolution block
    predictor_matrix = training_example_dict[PREDICTOR_MATRIX_KEY]
    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    temperature_matrix = predictor_matrix[8, ..., temperature_index]

    _, axes_object_matrix = create_paneled_figure(
        num_rows=2, num_columns=2, horizontal_spacing=0.2, vertical_spacing=0.2)

    plot_feature_map(
        feature_matrix=temperature_matrix, axes_object=axes_object_matrix[0, 0]
    )
    axes_object_matrix[0, 0].set_title('Before convolution')

    kernel_matrix = numpy.expand_dims(EDGE_DETECTOR_MATRIX1, axis=-1)
    kernel_matrix = numpy.expand_dims(kernel_matrix, axis=-1)

    feature_matrix_after_conv = do_2d_convolution(
        feature_matrix=numpy.expand_dims(temperature_matrix, axis=-1),
        kernel_matrix=kernel_matrix, pad_edges=False, stride_length_px=1
    )

    feature_matrix_after_activn = do_activation(
        input_values=feature_matrix_after_conv,
        function_name=RELU_FUNCTION_NAME)

    feature_matrix_after_pooling = do_2d_pooling(
        feature_matrix=feature_matrix_after_activn, stride_length_px=2,
        pooling_type_string=MAX_POOLING_TYPE_STRING)

    feature_matrix_after_conv = feature_matrix_after_conv[0, ..., 0]
    feature_matrix_after_activn = feature_matrix_after_activn[0, ..., 0]
    feature_matrix_after_pooling = feature_matrix_after_pooling[0, ..., 0]

    all_values = numpy.concatenate((
        numpy.ravel(feature_matrix_after_conv),
        numpy.ravel(feature_matrix_after_activn),
        numpy.ravel(feature_matrix_after_pooling)
    ))
    max_colour_value = numpy.percentile(numpy.absolute(all_values), 99.)
    min_colour_value = -1 * max_colour_value

    plot_feature_map(
        feature_matrix=feature_matrix_after_conv,
        axes_object=axes_object_matrix[0, 1],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[0, 1].set_title('After convolution')

    plot_feature_map(
        feature_matrix=feature_matrix_after_activn,
        axes_object=axes_object_matrix[1, 0],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[1, 0].set_title('After ReLU activation')

    plot_feature_map(
        feature_matrix=feature_matrix_after_pooling,
        axes_object=axes_object_matrix[1, 1],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[1, 1].set_title('After max-pooling')

    # Example 2 of convolution block
    predictor_matrix = training_example_dict[PREDICTOR_MATRIX_KEY]
    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    temperature_matrix = predictor_matrix[8, ..., temperature_index]

    _, axes_object_matrix = create_paneled_figure(
        num_rows=2, num_columns=3, horizontal_spacing=0.2, vertical_spacing=0.2)

    plot_feature_map(
        feature_matrix=temperature_matrix, axes_object=axes_object_matrix[0, 0]
    )
    axes_object_matrix[0, 0].set_title('Before convolution')

    kernel_matrix = numpy.expand_dims(EDGE_DETECTOR_MATRIX1, axis=-1)
    kernel_matrix = numpy.expand_dims(kernel_matrix, axis=-1)

    feature_matrix_after_conv1 = do_2d_convolution(
        feature_matrix=numpy.expand_dims(temperature_matrix, axis=-1),
        kernel_matrix=kernel_matrix, pad_edges=False, stride_length_px=1
    )

    feature_matrix_after_activn1 = do_activation(
        input_values=feature_matrix_after_conv1,
        function_name=LEAKY_RELU_FUNCTION_NAME)

    feature_matrix_after_conv2 = do_2d_convolution(
        feature_matrix=feature_matrix_after_activn1,
        kernel_matrix=kernel_matrix, pad_edges=False, stride_length_px=1)

    feature_matrix_after_activn2 = do_activation(
        input_values=feature_matrix_after_conv2,
        function_name=LEAKY_RELU_FUNCTION_NAME)

    feature_matrix_after_pooling = do_2d_pooling(
        feature_matrix=feature_matrix_after_activn2, stride_length_px=2,
        pooling_type_string=MAX_POOLING_TYPE_STRING)

    feature_matrix_after_conv1 = feature_matrix_after_conv1[0, ..., 0]
    feature_matrix_after_activn1 = feature_matrix_after_activn1[0, ..., 0]
    feature_matrix_after_conv2 = feature_matrix_after_conv2[0, ..., 0]
    feature_matrix_after_activn2 = feature_matrix_after_activn2[0, ..., 0]
    feature_matrix_after_pooling = feature_matrix_after_pooling[0, ..., 0]

    all_values = numpy.concatenate((
        numpy.ravel(feature_matrix_after_conv1),
        numpy.ravel(feature_matrix_after_activn1),
        numpy.ravel(feature_matrix_after_conv2),
        numpy.ravel(feature_matrix_after_activn2),
        numpy.ravel(feature_matrix_after_pooling)
    ))
    max_colour_value = numpy.percentile(numpy.absolute(all_values), 99.)
    min_colour_value = -1 * max_colour_value

    plot_feature_map(
        feature_matrix=feature_matrix_after_conv1,
        axes_object=axes_object_matrix[0, 1],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[0, 1].set_title('After first conv')

    plot_feature_map(
        feature_matrix=feature_matrix_after_activn1,
        axes_object=axes_object_matrix[0, 2],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[0, 2].set_title('After first leaky ReLU')

    plot_feature_map(
        feature_matrix=feature_matrix_after_conv2,
        axes_object=axes_object_matrix[1, 0],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[1, 0].set_title('After second conv')

    plot_feature_map(
        feature_matrix=feature_matrix_after_activn2,
        axes_object=axes_object_matrix[1, 1],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[1, 1].set_title('After second leaky ReLU')

    plot_feature_map(
        feature_matrix=feature_matrix_after_pooling,
        axes_object=axes_object_matrix[1, 2],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[1, 2].set_title('After max-pooling')

    # Example 1 of batch normalization
    predictor_matrix = training_example_dict[PREDICTOR_MATRIX_KEY]
    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    temperature_matrix = predictor_matrix[..., temperature_index]

    print('Batch size = {0:d} examples'.format(temperature_matrix.shape[0]))

    feature_matrix = numpy.expand_dims(temperature_matrix, axis=-1)
    feature_matrix = do_batch_normalization(
        feature_matrix=feature_matrix, scale_parameter=1., shift_parameter=0.)

    temperature_matrix = temperature_matrix[8, ...]
    feature_matrix = feature_matrix[8, ..., 0]

    all_values = numpy.concatenate((
        numpy.ravel(temperature_matrix), numpy.ravel(feature_matrix)
    ))
    max_colour_value = numpy.percentile(numpy.absolute(all_values), 99.)
    min_colour_value = -1 * max_colour_value

    _, axes_object_matrix = create_paneled_figure(num_rows=1, num_columns=2)

    plot_feature_map(
        feature_matrix=temperature_matrix, axes_object=axes_object_matrix[0, 0],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[0, 0].set_title('Before batch norm')

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[0, 1].set_title('After batch norm')

    # Example 2 of batch normalization
    predictor_matrix = training_example_dict[PREDICTOR_MATRIX_KEY]
    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    temperature_matrix = predictor_matrix[..., temperature_index]

    print('Batch size = {0:d} examples'.format(temperature_matrix.shape[0]))

    feature_matrix = numpy.expand_dims(temperature_matrix, axis=-1)
    feature_matrix = do_batch_normalization(
        feature_matrix=feature_matrix, scale_parameter=3., shift_parameter=-2.)

    temperature_matrix = temperature_matrix[8, ...]
    feature_matrix = feature_matrix[8, ..., 0]

    all_values = numpy.concatenate((
        numpy.ravel(temperature_matrix), numpy.ravel(feature_matrix)
    ))
    max_colour_value = numpy.percentile(numpy.absolute(all_values), 99.)
    min_colour_value = -1 * max_colour_value

    _, axes_object_matrix = create_paneled_figure(num_rows=1, num_columns=2)

    plot_feature_map(
        feature_matrix=temperature_matrix, axes_object=axes_object_matrix[0, 0],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[0, 0].set_title('Before batch norm')

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[0, 1].set_title('After batch norm')

    # Example 3 of convolution block
    num_examples = 100

    predictor_matrix = training_example_dict[PREDICTOR_MATRIX_KEY]
    temperature_index = predictor_names.index(TEMPERATURE_NAME)
    temperature_matrix = predictor_matrix[:num_examples, ..., temperature_index]

    _, axes_object_matrix = create_paneled_figure(
        num_rows=2, num_columns=4, horizontal_spacing=0.2, vertical_spacing=0.2)

    plot_feature_map(
        feature_matrix=temperature_matrix[8, ...],
        axes_object=axes_object_matrix[0, 0]
    )
    axes_object_matrix[0, 0].set_title('Before convolution', fontsize=20)

    kernel_matrix = numpy.expand_dims(EDGE_DETECTOR_MATRIX1, axis=-1)
    kernel_matrix = numpy.expand_dims(kernel_matrix, axis=-1)
    feature_matrix_after_conv1 = None

    for i in range(num_examples):
        if numpy.mod(i, 10) == 0:
            print('Convolving over example {0:d} of {1:d}...'.format(
                i + 1, num_examples
            ))

        this_feature_matrix = do_2d_convolution(
            feature_matrix=numpy.expand_dims(temperature_matrix[[i]], axis=-1),
            kernel_matrix=kernel_matrix, pad_edges=False, stride_length_px=1
        )

        if feature_matrix_after_conv1 is None:
            dimensions = (num_examples,) + this_feature_matrix.shape[1:]
            feature_matrix_after_conv1 = numpy.full(dimensions, numpy.nan)

        feature_matrix_after_conv1[i, ...] = this_feature_matrix[0, ...]

    feature_matrix_after_activn1 = do_activation(
        input_values=feature_matrix_after_conv1,
        function_name=LEAKY_RELU_FUNCTION_NAME)

    feature_matrix_after_bn1 = do_batch_normalization(
        feature_matrix=feature_matrix_after_activn1,
        scale_parameter=1., shift_parameter=0.)

    feature_matrix_after_conv2 = None

    for i in range(num_examples):
        if numpy.mod(i, 10) == 0:
            print('Convolving over example {0:d} of {1:d}...'.format(
                i + 1, num_examples
            ))

        this_feature_matrix = do_2d_convolution(
            feature_matrix=feature_matrix_after_bn1[i, ...],
            kernel_matrix=kernel_matrix, pad_edges=False, stride_length_px=1
        )

        if feature_matrix_after_conv2 is None:
            dimensions = (num_examples,) + this_feature_matrix.shape[1:]
            feature_matrix_after_conv2 = numpy.full(dimensions, numpy.nan)

        feature_matrix_after_conv2[i, ...] = this_feature_matrix[0, ...]

    feature_matrix_after_activn2 = do_activation(
        input_values=feature_matrix_after_conv2,
        function_name=LEAKY_RELU_FUNCTION_NAME)

    feature_matrix_after_bn2 = do_batch_normalization(
        feature_matrix=feature_matrix_after_activn2,
        scale_parameter=1., shift_parameter=0.)

    feature_matrix_after_pooling = do_2d_pooling(
        feature_matrix=feature_matrix_after_bn2, stride_length_px=2,
        pooling_type_string=MAX_POOLING_TYPE_STRING)

    feature_matrix_after_conv1 = feature_matrix_after_conv1[8, ..., 0]
    feature_matrix_after_activn1 = feature_matrix_after_activn1[8, ..., 0]
    feature_matrix_after_bn1 = feature_matrix_after_bn1[8, ..., 0]
    feature_matrix_after_conv2 = feature_matrix_after_conv2[8, ..., 0]
    feature_matrix_after_activn2 = feature_matrix_after_activn2[8, ..., 0]
    feature_matrix_after_bn2 = feature_matrix_after_bn2[8, ..., 0]
    feature_matrix_after_pooling = feature_matrix_after_pooling[8, ..., 0]

    all_values = numpy.concatenate((
        numpy.ravel(feature_matrix_after_conv1),
        numpy.ravel(feature_matrix_after_activn1),
        numpy.ravel(feature_matrix_after_bn1),
        numpy.ravel(feature_matrix_after_conv2),
        numpy.ravel(feature_matrix_after_activn2),
        numpy.ravel(feature_matrix_after_bn2),
        numpy.ravel(feature_matrix_after_pooling)
    ))
    max_colour_value = numpy.percentile(numpy.absolute(all_values), 99.)
    min_colour_value = -1 * max_colour_value

    plot_feature_map(
        feature_matrix=feature_matrix_after_conv1,
        axes_object=axes_object_matrix[0, 1],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[0, 1].set_title('After first conv', fontsize=20)

    plot_feature_map(
        feature_matrix=feature_matrix_after_activn1,
        axes_object=axes_object_matrix[0, 2],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[0, 2].set_title('After first leaky ReLU', fontsize=20)

    plot_feature_map(
        feature_matrix=feature_matrix_after_bn1,
        axes_object=axes_object_matrix[0, 3],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[0, 3].set_title('After first batch norm', fontsize=20)

    plot_feature_map(
        feature_matrix=feature_matrix_after_conv2,
        axes_object=axes_object_matrix[1, 0],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[1, 0].set_title('After second conv', fontsize=20)

    plot_feature_map(
        feature_matrix=feature_matrix_after_activn2,
        axes_object=axes_object_matrix[1, 1],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[1, 1].set_title('After second leaky ReLU', fontsize=20)

    plot_feature_map(
        feature_matrix=feature_matrix_after_bn2,
        axes_object=axes_object_matrix[1, 2],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[1, 2].set_title('After second batch norm', fontsize=20)

    plot_feature_map(
        feature_matrix=feature_matrix_after_pooling,
        axes_object=axes_object_matrix[1, 3],
        min_colour_value=min_colour_value, max_colour_value=max_colour_value
    )
    axes_object_matrix[1, 3].set_title('After max-pooling', fontsize=20)

    # Example 1 of CNN architecture
    num_grid_rows = 33
    num_grid_columns = 33
    num_channels = 8

    # Create input layer.
    input_layer_object = keras.layers.Input(
        shape=(num_grid_rows, num_grid_columns, num_channels)
    )
    last_layer_object = input_layer_object

    # Create first conv layer.
    conv_layer_object = keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), strides=(1, 1),
        padding='same', data_format='channels_last', dilation_rate=(1, 1),
        activation=None, use_bias=True
    )
    last_layer_object = conv_layer_object(last_layer_object)

    activation_layer_object = _get_activation_layer(LEAKY_RELU_FUNCTION_NAME)
    last_layer_object = activation_layer_object(last_layer_object)

    # First pooling layer.
    pooling_layer_object = _get_2d_pooling_layer(
        stride_length_px=2, pooling_type_string=MAX_POOLING_TYPE_STRING)
    last_layer_object = pooling_layer_object(last_layer_object)

    # Second conv layer.
    conv_layer_object = keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), strides=(1, 1),
        padding='same', data_format='channels_last', dilation_rate=(1, 1),
        activation=None, use_bias=True
    )
    last_layer_object = conv_layer_object(last_layer_object)

    activation_layer_object = _get_activation_layer(LEAKY_RELU_FUNCTION_NAME)
    last_layer_object = activation_layer_object(last_layer_object)

    # Second pooling layer.
    pooling_layer_object = _get_2d_pooling_layer(
        stride_length_px=2, pooling_type_string=MAX_POOLING_TYPE_STRING)
    last_layer_object = pooling_layer_object(last_layer_object)

    # Third conv layer.
    conv_layer_object = keras.layers.Conv2D(
        filters=64, kernel_size=(3, 3), strides=(1, 1),
        padding='same', data_format='channels_last', dilation_rate=(1, 1),
        activation=None, use_bias=True
    )
    last_layer_object = conv_layer_object(last_layer_object)

    activation_layer_object = _get_activation_layer(LEAKY_RELU_FUNCTION_NAME)
    last_layer_object = activation_layer_object(last_layer_object)

    # Third pooling layer.
    pooling_layer_object = _get_2d_pooling_layer(
        stride_length_px=2, pooling_type_string=MAX_POOLING_TYPE_STRING)
    last_layer_object = pooling_layer_object(last_layer_object)

    # Flattening layer.
    dimensions = numpy.array(
        last_layer_object.get_shape().as_list()[1:], dtype=int
    )
    num_scalar_features = numpy.prod(dimensions)

    flattening_layer_object = keras.layers.Flatten()
    last_layer_object = flattening_layer_object(last_layer_object)

    # Dense layers.
    dense_layer_object = keras.layers.Dense(128, activation=None, use_bias=True)
    last_layer_object = dense_layer_object(last_layer_object)

    activation_layer_object = _get_activation_layer(LEAKY_RELU_FUNCTION_NAME)
    last_layer_object = activation_layer_object(last_layer_object)

    dense_layer_object = keras.layers.Dense(3, activation=None, use_bias=True)
    last_layer_object = dense_layer_object(last_layer_object)

    activation_layer_object = _get_activation_layer('softmax')
    last_layer_object = activation_layer_object(last_layer_object)

    # Put everything together.
    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=last_layer_object)

    model_object.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(), metrics=LIST_OF_METRIC_FUNCTIONS)

    model_object.summary()
    
    # Example 1 of CNN-training
    simple_model_file_name = '{0:s}/simple_cnn.h5'.format(OUTPUT_DIR_NAME)

    # training_generator = example_generator(
    #     top_input_dir_name=TOP_TRAINING_DIR_NAME,
    #     predictor_names=PREDICTOR_NAMES_FOR_CNN,
    #     pressure_levels_mb=PRESSURE_LEVELS_FOR_CNN_MB,
    #     num_half_rows=16, num_half_columns=16, num_examples_per_batch=1024)
    #
    # validation_generator = example_generator(
    #     top_input_dir_name=TOP_VALIDATION_DIR_NAME,
    #     predictor_names=PREDICTOR_NAMES_FOR_CNN,
    #     pressure_levels_mb=PRESSURE_LEVELS_FOR_CNN_MB,
    #     num_half_rows=16, num_half_columns=16, num_examples_per_batch=1024)

    training_generator = example_generator(
        top_input_dir_name=TOP_TRAINING_DIR_NAME,
        predictor_names=PREDICTOR_NAMES_FOR_CNN,
        pressure_levels_mb=PRESSURE_LEVELS_FOR_CNN_MB,
        num_half_rows=16, num_half_columns=16, num_examples_per_batch=32)

    validation_generator = example_generator(
        top_input_dir_name=TOP_VALIDATION_DIR_NAME,
        predictor_names=PREDICTOR_NAMES_FOR_CNN,
        pressure_levels_mb=PRESSURE_LEVELS_FOR_CNN_MB,
        num_half_rows=16, num_half_columns=16, num_examples_per_batch=32)

    train_cnn(
        model_object=model_object,
        output_model_file_name=simple_model_file_name, num_epochs=10,
        num_training_batches_per_epoch=32, num_validation_batches_per_epoch=16,
        training_generator=training_generator,
        validation_generator=validation_generator)

    # Example 2 of CNN architecture
    num_grid_rows = 33
    num_grid_columns = 33
    num_channels = 8
    num_dense_layers = 3
    dense_layer_dropout_rate = 0.5
    conv_layer_regularizer = keras.regularizers.l1_l2(l1=0., l2=10 ** -2.5)

    # Create input layer.
    input_layer_object = keras.layers.Input(
        shape=(num_grid_rows, num_grid_columns, num_channels)
    )
    last_layer_object = input_layer_object

    # Create first conv layer in first conv block.
    conv_layer_object = keras.layers.Conv2D(
        filters=36, kernel_size=(3, 3), strides=(1, 1),
        padding='same', data_format='channels_last', dilation_rate=(1, 1),
        activation=None, use_bias=True,
        kernel_regularizer=conv_layer_regularizer
    )
    last_layer_object = conv_layer_object(last_layer_object)

    activation_layer_object = _get_activation_layer(LEAKY_RELU_FUNCTION_NAME)
    last_layer_object = activation_layer_object(last_layer_object)

    batch_norm_layer_object = _get_activation_layer(LEAKY_RELU_FUNCTION_NAME)
    last_layer_object = batch_norm_layer_object(last_layer_object)

    # Second conv layer in first conv block.
    conv_layer_object = keras.layers.Conv2D(
        filters=36, kernel_size=(3, 3), strides=(1, 1),
        padding='same', data_format='channels_last', dilation_rate=(1, 1),
        activation=None, use_bias=True,
        kernel_regularizer=conv_layer_regularizer
    )
    last_layer_object = conv_layer_object(last_layer_object)

    activation_layer_object = _get_activation_layer(LEAKY_RELU_FUNCTION_NAME)
    last_layer_object = activation_layer_object(last_layer_object)

    batch_norm_layer_object = _get_activation_layer(LEAKY_RELU_FUNCTION_NAME)
    last_layer_object = batch_norm_layer_object(last_layer_object)

    # Pooling layer in first conv block (this ends the block).
    pooling_layer_object = _get_2d_pooling_layer(
        stride_length_px=2, pooling_type_string=MAX_POOLING_TYPE_STRING)
    last_layer_object = pooling_layer_object(last_layer_object)

    # First conv layer, second conv block.
    conv_layer_object = keras.layers.Conv2D(
        filters=72, kernel_size=(3, 3), strides=(1, 1),
        padding='same', data_format='channels_last', dilation_rate=(1, 1),
        activation=None, use_bias=True,
        kernel_regularizer=conv_layer_regularizer
    )
    last_layer_object = conv_layer_object(last_layer_object)

    activation_layer_object = _get_activation_layer(LEAKY_RELU_FUNCTION_NAME)
    last_layer_object = activation_layer_object(last_layer_object)

    batch_norm_layer_object = _get_activation_layer(LEAKY_RELU_FUNCTION_NAME)
    last_layer_object = batch_norm_layer_object(last_layer_object)

    # Second conv layer, second conv block.
    conv_layer_object = keras.layers.Conv2D(
        filters=72, kernel_size=(3, 3), strides=(1, 1),
        padding='same', data_format='channels_last', dilation_rate=(1, 1),
        activation=None, use_bias=True,
        kernel_regularizer=conv_layer_regularizer
    )
    last_layer_object = conv_layer_object(last_layer_object)

    activation_layer_object = _get_activation_layer(LEAKY_RELU_FUNCTION_NAME)
    last_layer_object = activation_layer_object(last_layer_object)

    batch_norm_layer_object = _get_activation_layer(LEAKY_RELU_FUNCTION_NAME)
    last_layer_object = batch_norm_layer_object(last_layer_object)

    # Pooling layer in second conv block (this ends the block).
    pooling_layer_object = _get_2d_pooling_layer(
        stride_length_px=2, pooling_type_string=MAX_POOLING_TYPE_STRING)
    last_layer_object = pooling_layer_object(last_layer_object)

    dimensions = numpy.array(
        last_layer_object.get_shape().as_list()[1:], dtype=int
    )
    num_scalar_features = numpy.prod(dimensions)

    flattening_layer_object = keras.layers.Flatten()
    last_layer_object = flattening_layer_object(last_layer_object)

    _, num_outputs_by_dense_layer = _get_dense_layer_dimensions(
        num_features=num_scalar_features, num_predictions=3,
        num_dense_layers=num_dense_layers)

    for i in range(num_dense_layers):
        dense_layer_object = keras.layers.Dense(
            num_outputs_by_dense_layer[i], activation=None, use_bias=True
        )
        last_layer_object = dense_layer_object(last_layer_object)

        if i == num_dense_layers - 1:
            activation_layer_object = _get_activation_layer('softmax')
            last_layer_object = activation_layer_object(last_layer_object)
            break

        activation_layer_object = _get_activation_layer(
            LEAKY_RELU_FUNCTION_NAME)
        last_layer_object = activation_layer_object(last_layer_object)

        dropout_layer_object = keras.layers.Dropout(
            rate=dense_layer_dropout_rate)
        last_layer_object = dropout_layer_object(last_layer_object)

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=last_layer_object)

    model_object.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(), metrics=LIST_OF_METRIC_FUNCTIONS)

    model_object.summary()

    # Example 2 of CNN-training
    fancy_model_file_name = '{0:s}/fancy_cnn.h5'.format(OUTPUT_DIR_NAME)

    # training_generator = example_generator(
    #     top_input_dir_name=TOP_TRAINING_DIR_NAME,
    #     predictor_names=PREDICTOR_NAMES_FOR_CNN,
    #     pressure_levels_mb=PRESSURE_LEVELS_FOR_CNN_MB,
    #     num_half_rows=16, num_half_columns=16, num_examples_per_batch=1024)
    #
    # validation_generator = example_generator(
    #     top_input_dir_name=TOP_VALIDATION_DIR_NAME,
    #     predictor_names=PREDICTOR_NAMES_FOR_CNN,
    #     pressure_levels_mb=PRESSURE_LEVELS_FOR_CNN_MB,
    #     num_half_rows=16, num_half_columns=16, num_examples_per_batch=1024)

    training_generator = example_generator(
        top_input_dir_name=TOP_TRAINING_DIR_NAME,
        predictor_names=PREDICTOR_NAMES_FOR_CNN,
        pressure_levels_mb=PRESSURE_LEVELS_FOR_CNN_MB,
        num_half_rows=16, num_half_columns=16, num_examples_per_batch=32)

    validation_generator = example_generator(
        top_input_dir_name=TOP_VALIDATION_DIR_NAME,
        predictor_names=PREDICTOR_NAMES_FOR_CNN,
        pressure_levels_mb=PRESSURE_LEVELS_FOR_CNN_MB,
        num_half_rows=16, num_half_columns=16, num_examples_per_batch=32)

    train_cnn(
        model_object=model_object,
        output_model_file_name=fancy_model_file_name, num_epochs=10,
        num_training_batches_per_epoch=32, num_validation_batches_per_epoch=16,
        training_generator=training_generator,
        validation_generator=validation_generator)

    # Application to Testing Data
    first_testing_time_unix_sec = time_conversion.string_to_unix_sec(
        '2016-11-01-00', '%Y-%m-%d-%H')
    last_testing_time_unix_sec = time_conversion.string_to_unix_sec(
        '2016-12-31-21', '%Y-%m-%d-%H')
    best_model_object = cnn.read_model(BEST_MODEL_FILE_NAME)

    testing_file_names = find_many_testing_files(
        top_testing_dir_name=TOP_TESTING_DIR_NAME,
        first_time_unix_sec=first_testing_time_unix_sec,
        last_time_unix_sec=last_testing_time_unix_sec)

    class_probability_matrix, observed_labels = make_predictions(
        model_object=best_model_object, testing_file_names=testing_file_names,
        predictor_names=PREDICTOR_NAMES_FOR_CNN,
        pressure_levels_mb=PRESSURE_LEVELS_FOR_CNN_MB)

    # Evaluation on Testing Data
    num_examples = len(observed_labels)
    predicted_labels = numpy.full(num_examples, NO_FRONT_ENUM, dtype=int)

    warm_front_probs = class_probability_matrix[:, WARM_FRONT_ENUM]
    predicted_labels[
        warm_front_probs >= WARM_FRONT_PROB_THRESHOLD
    ] = WARM_FRONT_ENUM

    cold_front_probs = class_probability_matrix[:, COLD_FRONT_ENUM]
    predicted_labels[
        cold_front_probs >= COLD_FRONT_PROB_THRESHOLD
    ] = COLD_FRONT_ENUM

    contingency_matrix = eval_utils.get_contingency_table(
        predicted_labels=predicted_labels, observed_labels=observed_labels,
        num_classes=3)

    _, axes_object = pyplot.subplots(1, 1, figsize=(10, 10))
    plot_3class_contingency_table(contingency_matrix=contingency_matrix,
                                  axes_object=axes_object)
    axes_object.set_title('3-class contingency table')

    binary_contingency_dict = binary_eval.get_contingency_table(
        forecast_labels=(predicted_labels != NO_FRONT_ENUM).astype(int),
        observed_labels=(observed_labels != NO_FRONT_ENUM).astype(int)
    )

    a = binary_contingency_dict[binary_eval.NUM_TRUE_POSITIVES_KEY]
    b = binary_contingency_dict[binary_eval.NUM_FALSE_POSITIVES_KEY]
    c = binary_contingency_dict[binary_eval.NUM_FALSE_NEGATIVES_KEY]
    d = binary_contingency_dict[binary_eval.NUM_TRUE_NEGATIVES_KEY]

    binary_contingency_matrix = numpy.array([
        [a, b],
        [c, d]
    ])

    _, axes_object = pyplot.subplots(1, 1, figsize=(10, 10))
    plot_2class_contingency_table(contingency_matrix=binary_contingency_matrix,
                                  axes_object=axes_object)
    axes_object.set_title('2-class contingency table')

    binary_pod = binary_eval.get_pod(binary_contingency_dict)
    binary_pofd = binary_eval.get_pofd(binary_contingency_dict)
    binary_far = binary_eval.get_far(binary_contingency_dict)
    binary_csi = binary_eval.get_csi(binary_contingency_dict)
    binary_frequency_bias = binary_eval.get_frequency_bias(
        binary_contingency_dict)

    print((
        'POD (probability of detection) = fraction of fronts called fronts = '
        '{0:.3f}'
    ).format(
        binary_pod
    ))

    print((
        'POFD (probability of false detection) = fraction of non-fronts called '
        'fronts = {0:.3f}'
    ).format(
        binary_pofd
    ))

    print((
        'FAR (false-alarm ratio) = fraction of predicted fronts that are wrong '
        '= {0:.3f}'
    ).format(
        binary_far
    ))

    print((
        'CSI (critical success index) = accuracy without correct negatives = '
        '{0:.3f}'
    ).format(
        binary_csi
    ))

    print((
        'Frequency bias = number of predicted over actual fronts = {0:.3f}'
    ).format(
        binary_frequency_bias
    ))

    # ROC Curve and Performance Diagram
    any_front_probs = 1. - class_probability_matrix[:, NO_FRONT_ENUM]

    pofd_by_threshold, pod_by_threshold = binary_eval.get_points_in_roc_curve(
        forecast_probabilities=any_front_probs,
        observed_labels=(observed_labels != NO_FRONT_ENUM).astype(int),
        threshold_arg=1001)

    _, axes_object = pyplot.subplots(1, 1, figsize=(10, 10))
    model_eval_plotting.plot_roc_curve(
        axes_object=axes_object, pod_by_threshold=pod_by_threshold,
        pofd_by_threshold=pofd_by_threshold)

    auc = binary_eval.get_area_under_roc_curve(
        pod_by_threshold=pod_by_threshold, pofd_by_threshold=pofd_by_threshold)

    axes_object.set_title('Area under curve = {0:.3f}'.format(auc))

    success_ratio_by_threshold, pod_by_threshold = (
        binary_eval.get_points_in_performance_diagram(
            forecast_probabilities=any_front_probs,
            observed_labels=(observed_labels != NO_FRONT_ENUM).astype(int),
            threshold_arg=1001)
    )

    _, axes_object = pyplot.subplots(1, 1, figsize=(10, 10))
    model_eval_plotting.plot_performance_diagram(
        axes_object=axes_object, pod_by_threshold=pod_by_threshold,
        success_ratio_by_threshold=success_ratio_by_threshold)

    # Attributes Diagram
    mean_forecast_probs, observed_frequencies, example_counts = (
        binary_eval.get_points_in_reliability_curve(
            forecast_probabilities=any_front_probs,
            observed_labels=(observed_labels != NO_FRONT_ENUM).astype(int)
        )
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    model_eval_plotting.plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_forecast_by_bin=mean_forecast_probs,
        event_frequency_by_bin=observed_frequencies,
        num_examples_by_bin=example_counts)

    # Saliency example 1: WF saliency for NF example
    num_half_rows, num_half_columns = model_to_grid_dimensions(
        best_model_object)

    print('Reading data from: "{0:s}"...'.format(testing_file_names[0]))
    example_dict = read_examples(
        netcdf_file_name=testing_file_names[0],
        predictor_names_to_keep=PREDICTOR_NAMES_FOR_CNN,
        pressure_levels_to_keep_mb=PRESSURE_LEVELS_FOR_CNN_MB,
        num_half_rows_to_keep=num_half_rows,
        num_half_columns_to_keep=num_half_columns)

    example_index = numpy.where(
        example_dict[TARGET_MATRIX_KEY][:, NO_FRONT_ENUM] == 1
    )[0][0]
    predictor_matrix = example_dict[PREDICTOR_MATRIX_KEY][example_index, ...]

    saliency_matrix = get_saliency_maps(
        model_object=best_model_object, target_class=WARM_FRONT_ENUM,
        predictor_matrix=numpy.expand_dims(predictor_matrix, axis=0)
    )

    saliency_matrix = saliency_matrix[0, ...]
    num_predictors = saliency_matrix.shape[-1]

    for k in range(num_predictors):
        saliency_matrix[..., k] = apply_median_filter(
            input_matrix_2d=saliency_matrix[..., k], num_cells_in_half_window=1
        )

    plot_saliency_one_example(
        predictor_matrix=predictor_matrix[..., :4],
        saliency_matrix=saliency_matrix[..., :4],
        predictor_names=PREDICTOR_NAMES_FOR_CNN[:4]
    )

    # Saliency example 2: CF saliency for NF example
    num_half_rows, num_half_columns = model_to_grid_dimensions(
        best_model_object)

    print('Reading data from: "{0:s}"...'.format(testing_file_names[0]))
    example_dict = read_examples(
        netcdf_file_name=testing_file_names[0],
        predictor_names_to_keep=PREDICTOR_NAMES_FOR_CNN,
        pressure_levels_to_keep_mb=PRESSURE_LEVELS_FOR_CNN_MB,
        num_half_rows_to_keep=num_half_rows,
        num_half_columns_to_keep=num_half_columns)

    example_index = numpy.where(
        example_dict[TARGET_MATRIX_KEY][:, NO_FRONT_ENUM] == 1
    )[0][0]
    predictor_matrix = example_dict[PREDICTOR_MATRIX_KEY][example_index, ...]

    saliency_matrix = get_saliency_maps(
        model_object=best_model_object, target_class=COLD_FRONT_ENUM,
        predictor_matrix=numpy.expand_dims(predictor_matrix, axis=0)
    )

    saliency_matrix = saliency_matrix[0, ...]
    num_predictors = saliency_matrix.shape[-1]

    for k in range(num_predictors):
        saliency_matrix[..., k] = apply_median_filter(
            input_matrix_2d=saliency_matrix[..., k], num_cells_in_half_window=1
        )

    plot_saliency_one_example(
        predictor_matrix=predictor_matrix[..., :4],
        saliency_matrix=saliency_matrix[..., :4],
        predictor_names=PREDICTOR_NAMES_FOR_CNN[:4]
    )

    # Saliency example 3: CF saliency for CF example
    num_half_rows, num_half_columns = model_to_grid_dimensions(
        best_model_object)

    print('Reading data from: "{0:s}"...'.format(testing_file_names[0]))
    example_dict = read_examples(
        netcdf_file_name=testing_file_names[0],
        predictor_names_to_keep=PREDICTOR_NAMES_FOR_CNN,
        pressure_levels_to_keep_mb=PRESSURE_LEVELS_FOR_CNN_MB,
        num_half_rows_to_keep=num_half_rows,
        num_half_columns_to_keep=num_half_columns)

    example_index = numpy.where(
        example_dict[TARGET_MATRIX_KEY][:, COLD_FRONT_ENUM] == 1
    )[0][0]
    predictor_matrix = example_dict[PREDICTOR_MATRIX_KEY][example_index, ...]

    saliency_matrix = get_saliency_maps(
        model_object=best_model_object, target_class=COLD_FRONT_ENUM,
        predictor_matrix=numpy.expand_dims(predictor_matrix, axis=0)
    )

    saliency_matrix = saliency_matrix[0, ...]
    num_predictors = saliency_matrix.shape[-1]

    for k in range(num_predictors):
        saliency_matrix[..., k] = apply_median_filter(
            input_matrix_2d=saliency_matrix[..., k], num_cells_in_half_window=1
        )

    plot_saliency_one_example(
        predictor_matrix=predictor_matrix[..., :4],
        saliency_matrix=saliency_matrix[..., :4],
        predictor_names=PREDICTOR_NAMES_FOR_CNN[:4]
    )

    # Saliency example 4: WF saliency for CF example
    num_half_rows, num_half_columns = model_to_grid_dimensions(
        best_model_object)

    print('Reading data from: "{0:s}"...'.format(testing_file_names[0]))
    example_dict = read_examples(
        netcdf_file_name=testing_file_names[0],
        predictor_names_to_keep=PREDICTOR_NAMES_FOR_CNN,
        pressure_levels_to_keep_mb=PRESSURE_LEVELS_FOR_CNN_MB,
        num_half_rows_to_keep=num_half_rows,
        num_half_columns_to_keep=num_half_columns)

    example_index = numpy.where(
        example_dict[TARGET_MATRIX_KEY][:, COLD_FRONT_ENUM] == 1
    )[0][0]
    predictor_matrix = example_dict[PREDICTOR_MATRIX_KEY][example_index, ...]

    saliency_matrix = get_saliency_maps(
        model_object=best_model_object, target_class=WARM_FRONT_ENUM,
        predictor_matrix=numpy.expand_dims(predictor_matrix, axis=0)
    )

    saliency_matrix = saliency_matrix[0, ...]
    num_predictors = saliency_matrix.shape[-1]

    for k in range(num_predictors):
        saliency_matrix[..., k] = apply_median_filter(
            input_matrix_2d=saliency_matrix[..., k], num_cells_in_half_window=1
        )

    plot_saliency_one_example(
        predictor_matrix=predictor_matrix[..., :4],
        saliency_matrix=saliency_matrix[..., :4],
        predictor_names=PREDICTOR_NAMES_FOR_CNN[:4]
    )

    # CAM example 1
    num_half_rows, num_half_columns = model_to_grid_dimensions(
        best_model_object)

    print('Reading data from: "{0:s}"...'.format(testing_file_names[0]))
    example_dict = read_examples(
        netcdf_file_name=testing_file_names[0],
        predictor_names_to_keep=PREDICTOR_NAMES_FOR_CNN,
        pressure_levels_to_keep_mb=PRESSURE_LEVELS_FOR_CNN_MB,
        num_half_rows_to_keep=num_half_rows,
        num_half_columns_to_keep=num_half_columns)

    example_index = numpy.where(
        example_dict[TARGET_MATRIX_KEY][:, WARM_FRONT_ENUM] == 1
    )[0][0]
    predictor_matrix = example_dict[PREDICTOR_MATRIX_KEY][example_index, ...]

    conv_layer_names = [
        l.name for l in best_model_object.layers
        if 'batch_normalization' in l.name
    ]

    conv_layer_names = conv_layer_names[:4]
    num_conv_layers = len(conv_layer_names)

    for i in [0, 3]:
        class_activation_matrix = run_gradcam(
            model_object=best_model_object,
            predictor_matrix=numpy.expand_dims(predictor_matrix, axis=0),
            target_class=WARM_FRONT_ENUM, target_layer_name=conv_layer_names[i]
        )

        class_activation_matrix = apply_median_filter(
            input_matrix_2d=class_activation_matrix, num_cells_in_half_window=1)

        class_activation_matrix = numpy.maximum(class_activation_matrix, 0.)

        figure_object, _ = plot_class_activn_one_example(
            predictor_matrix=predictor_matrix[..., :4],
            class_activation_matrix=class_activation_matrix,
            predictor_names=PREDICTOR_NAMES_FOR_CNN[:4]
        )

        title_string = (
            'WF activation for {0:d}th of {1:d} conv layers'
        ).format(i + 1, num_conv_layers)

        figure_object.suptitle(title_string, y=1.01)

    # Grad-CAM: Example 2
    num_half_rows, num_half_columns = model_to_grid_dimensions(
        best_model_object)

    print('Reading data from: "{0:s}"...'.format(testing_file_names[0]))
    example_dict = read_examples(
        netcdf_file_name=testing_file_names[0],
        predictor_names_to_keep=PREDICTOR_NAMES_FOR_CNN,
        pressure_levels_to_keep_mb=PRESSURE_LEVELS_FOR_CNN_MB,
        num_half_rows_to_keep=num_half_rows,
        num_half_columns_to_keep=num_half_columns)

    example_index = numpy.where(
        example_dict[TARGET_MATRIX_KEY][:, WARM_FRONT_ENUM] == 1
    )[0][0]
    predictor_matrix = example_dict[PREDICTOR_MATRIX_KEY][example_index, ...]

    conv_layer_names = [
        l.name for l in best_model_object.layers
        if 'batch_normalization' in l.name
    ]

    conv_layer_names = conv_layer_names[:4]
    num_conv_layers = len(conv_layer_names)

    for i in [0, 3]:
        class_activation_matrix = run_gradcam(
            model_object=best_model_object,
            predictor_matrix=numpy.expand_dims(predictor_matrix, axis=0),
            target_class=COLD_FRONT_ENUM, target_layer_name=conv_layer_names[i]
        )

        class_activation_matrix = apply_median_filter(
            input_matrix_2d=class_activation_matrix, num_cells_in_half_window=1)

        class_activation_matrix = numpy.maximum(class_activation_matrix, 0.)

        figure_object, _ = plot_class_activn_one_example(
            predictor_matrix=predictor_matrix[..., :4],
            class_activation_matrix=class_activation_matrix,
            predictor_names=PREDICTOR_NAMES_FOR_CNN[:4]
        )

        title_string = (
            'CF activation for {0:d}th of {1:d} conv layers'
        ).format(i + 1, num_conv_layers)

        figure_object.suptitle(title_string, y=1.01)

    # Grad-CAM: Example 3
    num_half_rows, num_half_columns = model_to_grid_dimensions(
        best_model_object)

    print('Reading data from: "{0:s}"...'.format(testing_file_names[0]))
    example_dict = read_examples(
        netcdf_file_name=testing_file_names[0],
        predictor_names_to_keep=PREDICTOR_NAMES_FOR_CNN,
        pressure_levels_to_keep_mb=PRESSURE_LEVELS_FOR_CNN_MB,
        num_half_rows_to_keep=num_half_rows,
        num_half_columns_to_keep=num_half_columns)

    example_index = numpy.where(
        example_dict[TARGET_MATRIX_KEY][:, COLD_FRONT_ENUM] == 1
    )[0][0]
    predictor_matrix = example_dict[PREDICTOR_MATRIX_KEY][example_index, ...]

    conv_layer_names = [
        l.name for l in best_model_object.layers
        if 'batch_normalization' in l.name
    ]

    conv_layer_names = conv_layer_names[:4]
    num_conv_layers = len(conv_layer_names)

    for i in [0, 3]:
        class_activation_matrix = run_gradcam(
            model_object=best_model_object,
            predictor_matrix=numpy.expand_dims(predictor_matrix, axis=0),
            target_class=COLD_FRONT_ENUM, target_layer_name=conv_layer_names[i]
        )

        class_activation_matrix = apply_median_filter(
            input_matrix_2d=class_activation_matrix, num_cells_in_half_window=1)

        class_activation_matrix = numpy.maximum(class_activation_matrix, 0.)

        figure_object, _ = plot_class_activn_one_example(
            predictor_matrix=predictor_matrix[..., :4],
            class_activation_matrix=class_activation_matrix,
            predictor_names=PREDICTOR_NAMES_FOR_CNN[:4]
        )

        title_string = (
            'CF activation for {0:d}th of {1:d} conv layers'
        ).format(i + 1, num_conv_layers)

        figure_object.suptitle(title_string, y=1.01)

    # Grad-CAM: Example 4
    num_half_rows, num_half_columns = model_to_grid_dimensions(
        best_model_object)

    print('Reading data from: "{0:s}"...'.format(testing_file_names[0]))
    example_dict = read_examples(
        netcdf_file_name=testing_file_names[0],
        predictor_names_to_keep=PREDICTOR_NAMES_FOR_CNN,
        pressure_levels_to_keep_mb=PRESSURE_LEVELS_FOR_CNN_MB,
        num_half_rows_to_keep=num_half_rows,
        num_half_columns_to_keep=num_half_columns)

    example_index = numpy.where(
        example_dict[TARGET_MATRIX_KEY][:, COLD_FRONT_ENUM] == 1
    )[0][0]
    predictor_matrix = example_dict[PREDICTOR_MATRIX_KEY][example_index, ...]

    conv_layer_names = [
        l.name for l in best_model_object.layers
        if 'batch_normalization' in l.name
    ]

    conv_layer_names = conv_layer_names[:4]
    num_conv_layers = len(conv_layer_names)

    for i in [0, 3]:
        class_activation_matrix = run_gradcam(
            model_object=best_model_object,
            predictor_matrix=numpy.expand_dims(predictor_matrix, axis=0),
            target_class=WARM_FRONT_ENUM, target_layer_name=conv_layer_names[i]
        )

        class_activation_matrix = apply_median_filter(
            input_matrix_2d=class_activation_matrix, num_cells_in_half_window=1)

        class_activation_matrix = numpy.maximum(class_activation_matrix, 0.)

        figure_object, _ = plot_class_activn_one_example(
            predictor_matrix=predictor_matrix[..., :4],
            class_activation_matrix=class_activation_matrix,
            predictor_names=PREDICTOR_NAMES_FOR_CNN[:4]
        )

        title_string = (
            'WF activation for {0:d}th of {1:d} conv layers'
        ).format(i + 1, num_conv_layers)

        figure_object.suptitle(title_string, y=1.01)


if __name__ == '__main__':
    _run()
