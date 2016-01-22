package utils;

public class Scaler {
	double[] mean;
	double[] se;
	
	public double[][] scale(double[][] arr) {

		if (Utils.is_empty(arr))
			return new double[][] {};

		if (Utils.is_empty(arr[0]))
			return new double[][] {};

		// Perform z-score normalization

		int n = arr.length;
		int d = arr[0].length;
		double root_n = Math.sqrt(n);
		this.mean = Utils.mean(arr, 0);
		double[][] demeaned = new double[n][d];

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < d; j++) {
				demeaned[i][j] = arr[i][j] - mean[j];
			}
		}
		
		double[] std_dev = Utils.standard_deviation(arr);
		for (int j = 0; j < std_dev.length; j++) {
			if (std_dev[j] == 0.0) {
				std_dev[j] = 1.0;
			}
		}
		double[] std_err = Utils.deep_copy(std_dev);
		/*for (int j = 0; j < d; j++) {
			std_err[j] = std_err[j] / root_n;
		}*/
		this.se = std_err;

		double[][] out = new double[n][d];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < d; j++) {
				out[i][j] = demeaned[i][j] / std_dev[j];
			}
		}
		return out;
	}
	
	public double[][] descale(double[][] arr) {
		int n = arr.length;
		int d = arr[0].length;
		double[][] out = new double[n][d];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < d; j++) {
				out[i][j] = (arr[i][j] * this.se[j]) + this.mean[j];
			}
		}
		return out;
	}
}
