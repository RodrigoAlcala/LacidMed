from scipy.optimize import curve_fit
import numpy as np

class Fitting:
    def __init__(self, x: np.ndarray=None, y: np.ndarray=None):
        """
        Initializes operations object.
        Args:
            x (np.array): first array.
            y (np.array): second array.
        Raises:
            TypeError: If x or y is not a numpy array.
        """
        if not isinstance(x, np.ndarray) and x is not None:
            raise TypeError("x must be a numpy array")
        
        if not isinstance(y, np.ndarray) and y is not None:
            raise TypeError("y must be a numpy array")
        
        self.x = x
        self.y = y

    def gaussian(self, x, mean, amplitude, standard_deviation):
        """
        Gaussian function.
    
        Parameters:
        x (float or numpy.ndarray): Input value(s).
        mean (float): Mean of the Gaussian distribution.
        amplitude (float): Amplitude of the Gaussian curve.
        standard_deviation (float): Standard deviation of the Gaussian distribution.
        
        Returns:
        float or numpy.ndarray: Value(s) of the Gaussian function evaluated at input x.
        """
        return amplitude * np.exp(-((x - mean) / standard_deviation) ** 2 / 2)

    def fit_gaussian_to_histogram(self, initial_guess= [0, 0, 0]):
        """
        Fit a Gaussian curve to the given histogram.
        
        Parameters:
            hist (numpy.ndarray): Image histogram.
            bins (numpy.ndarray): Bin edges of the histogram.
            
        Returns:
            tuple: Parameters of the Gaussian fit (mean, amplitude, standard deviation).
        """

        # Perform curve fitting
        popt, _ = curve_fit(self.gaussian, self.x, self.y, p0=initial_guess)

        # Return the fitted parameters
        return popt

    def n_polynomial_fit(self, n):
        coefs = np.polyfit(self.x, self.y, n)
        pol_func = np.polyval(coefs,self.x)
        return pol_func