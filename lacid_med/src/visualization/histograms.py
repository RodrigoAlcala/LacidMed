import numpy as np
import matplotlib.pyplot as plt


class HistogramGenerator:
    def __init__(self, array_2D=None, array_3D=None):
        self.array_2D = array_2D if array_2D is not None else None
        self.array_3D = array_3D if array_3D is not None else None

    def create_histogram_2D(self, offset: int = 0, show: bool = True, plot_title: str = "", xlabel: str = "", ylabel: str = ""):
        """
        Generates a 2D histogram from the provided 2D array and returns the bins and frequencies.

        Args:
            offset (int, optional): The offset to apply to the histogram bins and frequencies. Defaults to 0.
            show (bool, optional): Whether to display the histogram using matplotlib. Defaults to True.

        Returns:
            tuple: bins and frequencies of the histogram
        """
        if self.array_2D is None:
            return None, None

        arr_max = int(np.max(self.array_2D))
        hist, bins = np.histogram(self.array_2D.flatten(), bins=arr_max, density=False)

        if offset < len(bins):
            bins = bins[offset + 1 :]
            hist = hist[offset:]
        else:
            bins = []
            hist = []

        if show:
            try:
                plt.plot(bins, hist)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(plot_title)
                plt.grid()
                plt.show()
            except Exception as e:
                print("Error plotting histogram:", str(e))

        return hist, bins

    def create_histogram_of_3d_array(self, offset: int = 0, show: bool = True, plot_title: str = "", xlabel: str = "", ylabel: str = ""):
        """
        Generates a 3D histogram from the provided 3D array and returns the bins and frequencies.

        Args:
            offset (int, optional): The offset to apply to the histogram bins and frequencies. Defaults to 0.
            show (bool, optional): Whether to display the histogram using matplotlib. Defaults to True.

        Returns:
            tuple: bins and frequencies of the histogram
        """
        if self.array_3D is None:
            return None, None

        histograms = []
        arr_max = int(np.max(self.array_3D))
        for i in range(self.array_3D.shape[2]):
            arr = self.array_3D[:, :, i]
            hist, bins = np.histogram(arr.flatten(), bins=arr_max, density=False)

            if offset < len(bins):
                bins = bins[offset + 1 :]
                hist = hist[offset:]
            else:
                bins = []
                hist = []

            histograms.append(hist)
        average_hist = np.mean(histograms, axis=0)

        if show:
            try:
                plt.plot(bins, average_hist)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(plot_title)
                plt.grid()
                plt.show()
            except Exception as e:
                print("Error plotting histogram:", str(e))

        return average_hist, bins
