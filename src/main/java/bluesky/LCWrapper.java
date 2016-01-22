package bluesky;

import org.apache.commons.math3.ml.clustering.Clusterable;

public class LCWrapper implements Clusterable {
	private double[] _LC;
	private int _original_index;

	public LCWrapper(double[] LC, int orig_index) {
		this._LC = LC;
		this._original_index = orig_index;
	}

	public double[] getPoint() {
		return _LC;
	}

	public int getIndex() {
		return this._original_index;
	}
}
