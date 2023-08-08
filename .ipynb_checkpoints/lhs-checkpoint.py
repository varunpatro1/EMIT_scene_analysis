import numpy as np
import pandas as pd

"""
IDAES process: 
- creates vector of points based on lhs_points_generation -> variable_sample_creation
- generated sample points: random_shuffling(vector_of_points)
- unique_sample_points = self.sample_point_selection

"""

__author__ = "Oluwamayowa Amusat"


class FeatureScaling:
    """

    A class for scaling and unscaling input and output data. The class contains three main functions
    """

    def __init__(self):
        pass

    @staticmethod
    def data_scaling_minmax(data):
        """

        This function performs column-wise minimax scaling on the input dataset.

            Args:
                data (NumPy Array or Pandas Dataframe): The input data set to be scaled. Must be a numpy array or dataframe.

            Returns:
                scaled_data(NumPy Array): A 2-D numpy array containing the scaled data. All array values will be between [0, 1].
                data_minimum(NumPy Array): A 2-D row vector containing the column-wise minimums of the input data
                data_maximum(NumPy Array): A 2-D row vector containing the column-wise maximums of the input data

            Raises:
                TypeError: Raised when the input data is not a numpy array or dataframe
        """
        # Confirm that data type is an array or DataFrame
        if isinstance(data, np.ndarray):
            input_data = data
        elif isinstance(data, pd.DataFrame):
            input_data = data.values
        else:
            raise TypeError(
                "original_data_input: Pandas dataframe or numpy array required."
            )

        if input_data.ndim == 1:
            input_data = input_data.reshape(len(input_data), 1)
        data_minimum = np.min(input_data, axis=0)
        data_maximum = np.max(input_data, axis=0)
        scale = data_maximum - data_minimum
        scale[scale == 0.0] = 1.0
        scaled_data = (input_data - data_minimum) / scale
        # scaled_data = (input_data - data_minimum) / (data_maximum - data_minimum)
        data_minimum = data_minimum.reshape(1, data_minimum.shape[0])
        data_maximum = data_maximum.reshape(1, data_maximum.shape[0])
        return scaled_data, data_minimum, data_maximum

    def data_unscaling_minmax(x_scaled, x_min, x_max):
        """

        This function performs column-wise un-scaling on the a minmax-scaled input dataset.

            Args:
                x_scaled(NumPy Array): The input data set to be un-scaled. Data values should be between 0 and 1.
                x_min(NumPy Array): 1-D or 2-D (n-by-1) vector containing the actual minimum value for each column. Must contain same number of elements as the number of columns in x_scaled.
                x_max(NumPy Array): 1-D or 2-D (n-by-1) vector containing the actual maximum value for each column. Must contain same number of elements as the number of columns in x_scaled.

            Returns:
                unscaled_data(NumPy Array): A 2-D numpy array containing the scaled data, unscaled_data = x_min + x_scaled * (x_max - x_min)

            Raises:
                IndexError: Function raises index error when the dimensions of the arrays are inconsistent.
        """
        # Check if it can be evaluated. Will return index error if dimensions are wrong
        if x_scaled.ndim == 1:  # Check if 1D, and convert to 2D if required.
            x_scaled = x_scaled.reshape(len(x_scaled), 1)
        if (x_scaled.shape[1] != x_min.size) or (x_scaled.shape[1] != x_max.size):
            raise IndexError("Dimensionality problems with data for un-scaling.")
        unscaled_data = x_min + x_scaled * (x_max - x_min)
        return unscaled_data
class SamplingMethods:
    def nearest_neighbour(self, full_data, a):
        """
        Function determines the closest point to a in data_input (user provided data).
        This is done by determining the input data with the smallest L2 distance from a.

        The function:
        1. Calculates the L2 distance between all the input data points and a,
        2. Sorts the input data based on the calculated L2-distances, and
        3. Selects the sample point in the first row (after sorting) as the closest sample point.

        Args:
            self: contains, among other things, the input data.
            full_data: refers to the input dataset supplied by the user.
            a: a single row vector containing the sample point we want to find the closest sample to.

        Returns:
            closest_point: a row vector containing the closest point to a in self.x_data
        """
        dist = full_data - a
        l2_norm = np.sqrt(np.sum((dist**2), axis=1))
        l2_norm = l2_norm.reshape(l2_norm.shape[0], 1)
        distances = np.append(full_data, l2_norm, 1)
        sorted_distances = distances[distances[:, -1].argsort()]
        closest_point = sorted_distances[0, :-1]
        return closest_point

    def points_selection(self, full_data, generated_sample_points):
        """
        Uses L2-distance evaluation (implemented in nearest_neighbour) to find closest available points in original data to those generated by the sampling technique.
        Calls the nearest_neighbour function for each row in the input data.

        Args:
            full_data: refers to the input dataset supplied by the user.
            generated_sample_points(NumPy Array): The vector of points (number_of_sample rows) for which the closest points in the original data are to be found. Each row represents a sample point.

        Returns:
            equivalent_points: Array containing the points (in rows) most similar to those in generated_sample_points
        """

        equivalent_points = np.zeros(
            (generated_sample_points.shape[0], len(self.data_headers))
        )
        for i in range(0, generated_sample_points.shape[0]):
            closest_point = self.nearest_neighbour(
                full_data, generated_sample_points[i, :]
            )
            equivalent_points[i, :] = closest_point
        return equivalent_points

    def sample_point_selection(self, full_data, sample_points, sampling_type):
        sd = FeatureScaling()
        scaled_data, data_min, data_max = sd.data_scaling_minmax(full_data)
        points_closest_scaled = self.points_selection(scaled_data, sample_points)
        points_closest_unscaled = sd.data_unscaling_minmax(points_closest_scaled, data_min, data_max)

        unique_sample_points = np.unique(points_closest_unscaled, axis=0)

        return unique_sample_points
        
    def selection_columns_preprocessing(self, data_input, xlabels, ylabels):
        """
        Pre-processing data for multiple output selection case.

        Args:
            data_input:     data supplied by user (dataframe or numpy array)
            xlabels:        list of input variables
            ylabels:        list of output variables
        """
        self.df_flag = True

        if isinstance(data_input, pd.DataFrame):
            if xlabels is None:
                xlabels = []
            if ylabels is None:
                ylabels = []
            set_of_labels = xlabels + ylabels

            self.x_data = data_input.filter(xlabels).values
            self.data_headers = set_of_labels
            self.data_headers_xvars = xlabels
            self.data = data_input.filter(set_of_labels).values

        if isinstance(data_input, np.ndarray):
            self.df_flag = False

            if xlabels is None:
                xlabels = []
            if ylabels is None:
                ylabels = []
            set_of_labels = xlabels + ylabels

            self.x_data = data_input[:, xlabels]
            self.data_headers = set_of_labels
            self.data_headers_xvars = xlabels
            self.data = data_input[:, set_of_labels]


class LHS(SamplingMethods):
    
    def __init__(self, data_input, number_of_samples=None, sampling_type=None, xlabels=None, ylabels=None,
                ):

        if self.sampling_type == "selection":
            if isinstance(data_input, (pd.DataFrame, np.ndarray)):
                self.selection_columns_preprocessing(data_input, xlabels, ylabels)
            self.number_of_samples = number_of_samples

    def variable_sample_creation(self, variable_min, variable_max):
        """

        Function that generates the required number of sample points for a given variable within a specified range using stratification.
        The function divides the variable sample space into self.number_of_samples equal strata and generates a single random sample from each strata based on its lower and upper bound.

        Args:
            self
            variable_min(float): The lower bound of the sample space region. Should be a single number.
            variable_max(float): The upper bound of the sample space region. Should be a single number.

        Returns:
            var_samples(NumPy Array): A numpy array of size (self.number_of_samples x 1) containing the randomly generated points from each strata
        """

        strata_size = 1 / self.number_of_samples
        var_samples = np.zeros((self.number_of_samples, 1))
        for i in range(self.number_of_samples):
            strata_lb = i * strata_size
            sample_point = strata_lb + (np.random.rand() * strata_size)
            var_samples[i, 0] = (
                sample_point * (variable_max - variable_min)
            ) + variable_min
        return var_samples

    def lhs_points_generation(self):
        """
        Generate points within each strata for each variable based on stratification. When invoked, it:
        1. Determines the mimumum and maximum value for each feature (column),
        2. Calls the variable_sample_creation function on each feature, passing in its mimmum and maximum
        3. Returns an array containing the points selected in each strata of each column

        Returns:
            sample_points_vector(NumPy Array): Array containing the columns of the random samples generated in each strata.
        """

        ns, nf = np.shape(self.x_data)  # pylint: disable=unused-variable
        sample_points_vector = np.zeros(
            (self.number_of_samples, nf)
        )  # Array containing points in each interval for each variable
        for i in range(nf):
            variable_min = 0  # np.min(self.x_data[:, i])
            variable_max = 1  # np.max(self.x_data[:, i])
            var_samples = self.variable_sample_creation(
                variable_min, variable_max
            )  # Data generation step
            sample_points_vector[:, i] = var_samples[:, 0]
        return sample_points_vector

    @staticmethod
    def random_shuffling(vector_of_points):
        """
        This function carries out random shuffling of column data to generate samples.
        Data in each of the columns  in the input array is shuffled separately, meaning that the rows of the resultant array will contain random samples from the sample space.

        Args:
            vector_of_points(NumPy Array): Array containing ordered points generated from stratification. Should usually be the output of the lhs_points_generation function. Each column self.number_of_samples elements.

        Returns:
            vector_of_points(NumPy Array): 2-D array containing the shuffled data. Should contain number_of_sample rows, with each row representing a potential random sample from within the sample space.

        """

        _, nf = np.shape(vector_of_points)
        for i in range(0, nf):
            z_col = vector_of_points[:, i]
            np.random.shuffle(z_col)
            vector_of_points[:, i] = z_col
        return vector_of_points

    def sample_points(self):
        """
        ``sample_points`` generates or selects Latin Hypercube samples from an input dataset or data range. When called, it:

            1. generates samples points from stratified regions by calling the ``lhs_points_generation``,
            2. generates potential sample points by random shuffling, and
            3. when a dataset is provided, selects the closest available samples to the theoretical sample points from within the input data.

        Returns:
            NumPy Array or Pandas Dataframe:     A numpy array or Pandas dataframe containing **number_of_samples** points selected or generated by LHS.

        """
        vector_of_points = (self.lhs_points_generation())  # Assumes [X, Y] data is supplied.
        generated_sample_points = self.random_shuffling(vector_of_points)
        unique_sample_points = self.sample_point_selection(self.data, generated_sample_points, self.sampling_type)

        if len(self.data_headers) > 0 and self.df_flag:
            unique_sample_points = pd.DataFrame(unique_sample_points, columns=self.data_headers)
        return unique_sample_points



