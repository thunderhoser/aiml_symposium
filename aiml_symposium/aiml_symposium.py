"""Normal Python version of notebook."""

import copy
import glob
import random
import os.path
import collections
import numpy
import netCDF4
import keras.layers
import keras.backend as K
import keras.callbacks
import matplotlib.colors
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import model_evaluation as binary_eval
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
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

# These constants help to find input data.
TIME_FORMAT = '%Y%m%d%H'
TIME_FORMAT_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9][0-2][0-9]'
BATCH_NUMBER_REGEX = '[0-9][0-9][0-9][0-9][0-9][0-9][0-9]'
NUM_BATCHES_PER_DIRECTORY = 1000

# Target values.
NO_FRONT_ENUM = 0
WARM_FRONT_ENUM = 1
COLD_FRONT_ENUM = 2

# Valid pooling types.
MAX_POOLING_TYPE_STRING = 'max'
MEAN_POOLING_TYPE_STRING = 'avg'
VALID_POOLING_TYPE_STRINGS = [MAX_POOLING_TYPE_STRING, MEAN_POOLING_TYPE_STRING]

# Valid activation functions.
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

# Predictor variables.
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

# Constants for plotting.
WIND_COLOUR_MAP_OBJECT = pyplot.get_cmap('binary')
FEATURE_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

LEVELS_FOR_CSI_CONTOURS = numpy.array([
    0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
])
LEVELS_FOR_BIAS_CONTOURS = numpy.array([0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5])

# Convolutional kernels.
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

# These metrics will be reported during training of a CNN (convolutional neural
# network).
LIST_OF_METRIC_FUNCTIONS = [
    keras_metrics.accuracy, keras_metrics.binary_pod,
    keras_metrics.binary_pofd, keras_metrics.binary_peirce_score,
    keras_metrics.binary_csi, keras_metrics.binary_frequency_bias,
    keras_metrics.binary_success_ratio
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
    error_checking.assert_is_leq(fraction_of_axis_length, 1.)

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
        shared_x_axis=False, shared_y_axis=False, keep_aspect_ratio=True):
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

    figure_object, axes_object_matrix = pyplot.subplots(
        num_rows, num_columns, sharex=shared_x_axis, sharey=shared_y_axis,
        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
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
        num_half_columns_to_keep=None, first_time_to_keep_unix_sec=None,
        last_time_to_keep_unix_sec=None):
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
    :param first_time_to_keep_unix_sec: First valid time to keep.  If None, all
        valid times will be kept.
    :param last_time_to_keep_unix_sec: Last valid time to keep.  If None, all
        valid times will be kept.
    :return: example_dict: See doc for `create_examples`.
    """

    # Check input args.
    if first_time_to_keep_unix_sec is None:
        first_time_to_keep_unix_sec = 0
    if last_time_to_keep_unix_sec is None:
        last_time_to_keep_unix_sec = int(1e12)

    error_checking.assert_is_boolean(metadata_only)
    error_checking.assert_is_integer(first_time_to_keep_unix_sec)
    error_checking.assert_is_integer(last_time_to_keep_unix_sec)
    error_checking.assert_is_geq(
        last_time_to_keep_unix_sec, first_time_to_keep_unix_sec)

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

    example_indices = numpy.where(numpy.logical_and(
        valid_times_unix_sec >= first_time_to_keep_unix_sec,
        valid_times_unix_sec <= last_time_to_keep_unix_sec
    ))[0]

    example_dict = {
        VALID_TIMES_KEY: valid_times_unix_sec[example_indices],
        ROW_INDICES_KEY: row_indices[example_indices],
        COLUMN_INDICES_KEY: column_indices[example_indices],
        PREDICTOR_NAMES_KEY: predictor_names_to_keep,
        PRESSURE_LEVELS_KEY: pressure_levels_to_keep_mb,
        DILATION_DISTANCE_KEY: getattr(dataset_object, DILATION_DISTANCE_KEY),
        MASK_MATRIX_KEY:
            numpy.array(dataset_object.variables[MASK_MATRIX_KEY][:], dtype=int)
    }

    if found_normalization_params:
        example_dict.update({
            NORMALIZATION_TYPE_KEY: normalization_type_string,
            FIRST_NORM_PARAM_KEY:
                first_normalization_param_matrix[example_indices, ...],
            SECOND_NORM_PARAM_KEY:
                second_normalization_param_matrix[example_indices, ...]
        })

    if not metadata_only:
        example_dict.update({
            PREDICTOR_MATRIX_KEY:
                predictor_matrix[example_indices, ...].astype('float32'),
            TARGET_MATRIX_KEY:
                target_matrix[example_indices, ...].astype('float64')
        })

    dataset_object.close()
    return example_dict


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


def plot_wind_barbs(
        u_wind_matrix, v_wind_matrix, axes_object=None,
        colour_map_object=WIND_COLOUR_MAP_OBJECT, min_colour_speed=-1.,
        max_colour_speed=0., barb_length=8, empty_barb_radius=0.1):
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
        X=x_coord_matrix, Y=y_coord_matrix, U=u_wind_matrix, V=v_wind_matrix,
        C=wind_speed_matrix, length=barb_length, sizes=barb_size_dict,
        barb_increments=barb_increment_dict, fill_empty=True, rounding=False,
        cmap=colour_map_object,
        clim=numpy.array([min_colour_speed, max_colour_speed])
    )

    return axes_object


def _run():
    """Normal Python version of notebook.

    This is effectively the main method.
    """

    # Finding a Training File: Example 1
    batch_number = 179

    training_example_file_name = find_training_file(
        top_training_dir_name=TOP_TRAINING_DIR_NAME, batch_number=batch_number,
        raise_error_if_missing=False)
    print(training_example_file_name)

    # Finding a Training File: Example 2
    batch_number = 179

    training_example_file_name = find_training_file(
        top_training_dir_name=TOP_TRAINING_DIR_NAME, batch_number=batch_number,
        raise_error_if_missing=True)
    print(training_example_file_name)

    # Finding Many Training Files: Example
    first_batch_number = 0
    last_batch_number = 1000

    training_example_file_names = find_many_training_files(
        top_training_dir_name=TOP_TRAINING_DIR_NAME,
        first_batch_number=first_batch_number,
        last_batch_number=last_batch_number)

    for this_file_name in training_example_file_names:
        print(this_file_name)

    # Reading a Training File: Example 1
    batch_number = 179

    training_example_file_name = find_training_file(
        top_training_dir_name=TOP_TRAINING_DIR_NAME, batch_number=batch_number,
        raise_error_if_missing=True)

    training_example_dict = read_examples(
        netcdf_file_name=training_example_file_name, metadata_only=True)

    for this_key in training_example_dict:
        print('{0:s} ... {1:s}\n'.format(
            this_key, str(training_example_dict[this_key])
        ))

    # Reading a Training File: Example 2
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

    for this_key in training_example_dict:
        if this_key in [PREDICTOR_MATRIX_KEY, TARGET_MATRIX_KEY]:
            continue

        print('{0:s} ... {1:s}\n'.format(
            this_key, str(training_example_dict[this_key])
        ))

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

    for m in range(len(predictor_names)):
        print((
            'Normalized {0:s} for southwest corner of first example:\n{1:s}\n'
        ).format(
            predictor_names[m], str(predictor_matrix[8, :5, :5, m])
        ))

        print((
            'Min and max normalized {0:s} over all examples and grid cells = '
            '{1:.4f}, {2:.4f}\n'
        ).format(
            predictor_names[m],
            numpy.min(predictor_matrix[..., m]),
            numpy.max(predictor_matrix[..., m])
        ))

    # Random Convolution: Example 1
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

    print('Shape of output feature map = {0:s}'.format(
        str(feature_matrix.shape)
    ))

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1]
    )
    axes_object_matrix[0, 1].set_title('After convolution')

    # Random Convolution: Example 2
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

    print('Shape of output feature map = {0:s}'.format(
        str(feature_matrix.shape)
    ))

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1]
    )
    axes_object_matrix[0, 1].set_title('After convolution')

    # Random Convolution: Example 3
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

    print('Shape of output feature map = {0:s}'.format(
        str(feature_matrix.shape)
    ))

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1]
    )
    axes_object_matrix[0, 1].set_title('After convolution')

    # Edge Detection: Example 1
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

    print('Shape of output feature map = {0:s}'.format(
        str(feature_matrix.shape)
    ))

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1]
    )
    axes_object_matrix[0, 1].set_title('After convolution')

    # Edge Detection: Example 2
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

    print('Shape of output feature map = {0:s}'.format(
        str(feature_matrix.shape)
    ))

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1]
    )
    axes_object_matrix[0, 1].set_title('After convolution')

    # Edge Detection: Example 3
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

    print('Shape of output feature map = {0:s}'.format(
        str(feature_matrix.shape)
    ))

    plot_feature_map(
        feature_matrix=feature_matrix, axes_object=axes_object_matrix[0, 1]
    )
    axes_object_matrix[0, 1].set_title('After convolution')

    # Example: Multi-channel Convolution
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


if __name__ == '__main__':
    _run()
