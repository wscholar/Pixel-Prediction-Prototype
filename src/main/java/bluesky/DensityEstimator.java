package bluesky;

import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;

import utils.Utils;
import bluesky.enums.DE_Mode;

/**
 * @author gmontane
 *
 */
public class DensityEstimator {
	double[][] data;
	boolean fixed_bandwidth;
	DE_Mode mode;
	double MIN_COV = 1e-100;
	double[][] h = null;
	int dim;
	int num_pts_used = -1;
	int[] sampled_indices;

	public DensityEstimator() {
	}

	public DensityEstimator(double[][] d_points, int num_subsamples, int num_pts_used, DE_Mode mode) {
		if (num_subsamples > d_points.length) {
			num_subsamples = d_points.length;
		}
		int[] indices = Utils.randPermutation(d_points.length);
		int[] retained = Arrays.copyOfRange(indices, 0, num_subsamples);
		this.data = new double[num_subsamples][];
		this.sampled_indices = new int[retained.length];
		for (int i = 0; i < retained.length; i++) {
			this.data[i] = d_points[retained[i]];
			this.sampled_indices[i] = retained[i];
		}
		this.mode = mode;
		this.num_pts_used = num_pts_used;
	}
	
	public double[] pdf(double[][] query_points, String label) {
		return null;
	}
}
