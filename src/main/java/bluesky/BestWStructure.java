package bluesky;

import utils.Utils;

public class BestWStructure {
	public double MSE;
	public double[][] W;
	
	public BestWStructure(double mean_squared_error, double[][] W_matrix) {
		this.MSE = mean_squared_error;
		this.W = Utils.deep_copy(W_matrix);
	}
}
