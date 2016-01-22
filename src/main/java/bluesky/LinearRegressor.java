package bluesky;
import java.util.Arrays;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import utils.Utils;

public class LinearRegressor implements Algorithm {
	private double[][] models; 
	private int output_dim;
	private int input_dim;
	
	public LinearRegressor() {}
	
	public LinearRegressor(double[][] X, double[][] Y) throws Exception {
		this.build_models(X, Y);
	}
	
	public LinearRegressor(LightConesCollection LCC) throws Exception {
		this.build_models(LCC.get_PLCs(), LCC.get_FLCs());
	}
	
	private void build_models(double[][] X, double[][] Y) {
		this.output_dim = Y[0].length;
		this.input_dim = X[0].length;
		this.models = new double[this.output_dim][this.input_dim];
		
		for (int d = 0; d < this.output_dim; d++) {
			double[] y_data = Utils.get_column_slice(Y, d);
			OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
			ols.setNoIntercept(true);
			ols.newSampleData(y_data, X);
			this.models[d] = ols.estimateRegressionParameters();
		}
		
		this.models = Utils.transpose(this.models);
	}

	private double[][] add_ones_column(double[][] arr) {
		int n = arr.length;
		int d = arr[0].length;
		double[][] out = new double[n][d + 1];
		for (int i = 0; i < n; i++) {
			out[i][0] = 1.0;
			for (int j = 1; j < d; j++) {
				out[i][j] = arr[i][j-1];
			}
		}
		return out;
	}
	
	@Override
	public void learn(double[][] histories, double[][] futures) {
		this.build_models(histories, futures);
	}

	@Override
	public double[][] predict_batch(double[][] chunk, String chunkname) {
		return Utils.matrix_multiply(chunk, this.models);
	}

	@Override
	public double[] compute_likelihoods(double[][] PLCs, double[][] FLCs, String label) {
		double[] out = new double[PLCs.length];
		for (int i = 0; i < out.length; i++) {
			out[i] = 1.0;
		}
		return out;
	}

	@Override
	public void clear() {
		// TODO Auto-generated method stub
	}
}

