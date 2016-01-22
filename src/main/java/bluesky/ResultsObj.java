package bluesky;

import utils.Utils;

public class ResultsObj {

	private double[][] truth = null;
	private double[][] predictions = null;
	private double[][] errors = null;

	public ResultsObj(double[][] truth, double[][] predictions, double[][] errors) {
		this.truth = Utils.deep_copy(truth);
		this.predictions = Utils.deep_copy(predictions);
		this.errors = Utils.deep_copy(errors);
	}

	public double[][] getTruth() {
		return truth;
	}

	public double[][] getPredictions() {
		return predictions;
	}

	public double[][] getErrors() {
		return errors;
	}

}
