package bluesky;

import java.util.Arrays;
import java.util.HashMap;

import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

import utils.Utils;
import weka.core.matrix.Matrix;
import bluesky.enums.DE_Mode;

public class WeightedKDE extends DensityEstimator {
	double MIN_COV = 1e-100;
	double[][] h = null;
	private double _norm_const_1;
	private double[][] _h_inv;
	int dim;
	double bandwidth_adjustment_factor;
	HashMap<String, double[][]> saved_kernel_evals;

	public WeightedKDE(double[][] d_points, int num_subsamples, int num_pts_used, DE_Mode mode, boolean fixed_bandwidth, double bandwidth_adjustment) {
		super(d_points, num_subsamples, num_pts_used, mode);
		assert (d_points.length > 0);
		this.dim = d_points[0].length;
		this.mode = mode;
		this.fixed_bandwidth = fixed_bandwidth;
		this.bandwidth_adjustment_factor = bandwidth_adjustment;
		this.update_bandwidth(this.calculate_bandwidth(this.data));
		this.saved_kernel_evals = new HashMap<String, double[][]>();
	}

	public void clear() {
		this.saved_kernel_evals.clear();
		this.data = null;
	}

	private double[][] add_min_cov(double[][] mat) {
		int d = mat.length;
		double[][] out = Utils.deep_copy(mat);
		for (int i = 0; i < d; i++) {
			out[i][i] = mat[i][i] < this.MIN_COV ? this.MIN_COV + mat[i][i] : mat[i][i];
		}
		return out;
	}

	private void _update_norm_consts() {
		double[][] h = this.h;
		int d = this.dim;
		Matrix m = new Matrix(h);

		if (h != null) {
			try {
				this._h_inv = m.inverse().getArray();
			} catch (Exception e) {
				System.out.println("Error -- singular matrix; cannot invert.");
				System.out.print(Arrays.deepToString(this.h));
				System.exit(1);
			}

			if (this.mode == DE_Mode.SCALAR) {
				this._norm_const_1 = 1. / (Math.pow(h[0][0], d) * Math.pow((2 * Math.PI), (d / 2.0)));
			} else {
				this._norm_const_1 = 1. / Math.pow(m.det() * Math.pow((2 * Math.PI), d), 0.5);
			}
		}
	}

	public void update_bandwidth(double[][] h) {
		this.h = h;
		this._update_norm_consts();
	}

	public double get_norm_consts() {
		return this._norm_const_1;
	}

	public double[][] calculate_bandwidth(double[][] data) {
		// Rule-of-thumb
		int n = data.length;
		int d = data[0].length;
		double a = 1.0 / (d + 4.0);
		double[][] s_mat = new double[d][d];
		StandardDeviation std = new StandardDeviation();

		if (d == 1) {
			s_mat[0][0] = std.evaluate(Utils.ravel(data));
		} else {
			Covariance covObj = new Covariance(data);
			double[][] cov_data = covObj.getCovarianceMatrix().getData();
			try {
				s_mat = Matrix.constructWithCopy(cov_data).sqrt().getArray();
			} catch (Exception MathUnsupportedOperationException) {
				System.out.println("WARNING: Non-Positive Definite covariance matrix; could not find sqrt. Using full covariance instead.");
				for (int j = 0; j < d; j++) {
					s_mat[j][j] = std.evaluate(Utils.get_column_slice(data, j));
					if (s_mat[j][j] < this.MIN_COV) {
						s_mat[j][j] = this.MIN_COV;
					}
				}
			}
		}

		switch (this.mode) {
			case SCALAR:
				double mean = 0.0;
				for (int i = 0; i < s_mat.length; i++) {
					for (int j = 0; j < s_mat[i].length; j++) {
						mean += i == j ? s_mat[i][j] : 0.0;
					}
				}
				mean = mean / s_mat.length;
				for (int i = 0; i < s_mat.length; i++) {
					for (int j = 0; j < s_mat[i].length; j++) {
						s_mat[i][j] = i == j ? mean : 0.0;
					}
				}
				break;
			case DIAG:
				s_mat = new double[d][d];
				for (int j = 0; j < d; j++) {
					s_mat[j][j] = std.evaluate(Utils.get_column_slice(data, j));
				}
				break;
			default:
				// FULL
				break;
		}

		double c = Math.pow((4.0 / (d + 2.0)), a) * Math.pow(n, -a);
		double[][] H = this.add_min_cov(s_mat);
		for (int i = 0; i < s_mat.length; i++) {
			for (int j = 0; j < s_mat[i].length; j++) {
				H[i][j] *= (c * this.bandwidth_adjustment_factor);
			}
		}
		return H;
	}

	// TODO: Check if recalculating distance from point to exemplars for all states, or if just reusing the weights 
	private double[] _kernel_evals(double[] p) {
		double[] out = new double[this.data.length];
		if (this.mode == DE_Mode.SCALAR) {
			double c2 = -0.5 / Math.pow(this.h[0][0], -2);
			for (int i = 0; i < this.data.length; i++) {
				out[i] = c2 * Math.pow(Utils.calculateDistance(this.data[i], p), 2);
			}
		} else {
			double[][] HI = Utils.deep_copy(this._h_inv);
			for (int i = 0; i < HI.length; i++) {
				for (int j = 0; j < HI[i].length; j++) {
					HI[i][j] = Math.pow(HI[i][j], 2);
				}
			}
			double[] m_dists = Utils.mahalanobis_distance(p, this.data, HI);
			for (int i = 0; i < m_dists.length; i++) {
				out[i] = -0.5 * m_dists[i];
			}
		}
		for (int i = 0; i < out.length; i++) {
			out[i] = Math.exp(out[i]);
		}
		return out;
	}

	private DensityReturnObj _density(double[] in_weights, double[] p) {
		double[] weights = new double[this.sampled_indices.length];

		if (this.h == null) {
			this.update_bandwidth(this.calculate_bandwidth(this.data));
		}

		if (in_weights == null) {
			Arrays.fill(weights, 1.0);
		} else {
			for (int i = 0; i < this.sampled_indices.length; i++) {
				weights[i] = in_weights[this.sampled_indices[i]];
			}
		}
		weights = Utils.normalize(weights);
		double[] kernel_evaluations = this._kernel_evals(p);
		double dp = Utils.dot_product(kernel_evaluations, weights);
		return new DensityReturnObj(this._norm_const_1 * dp, kernel_evaluations);
	}

	private double[] _precomputed_density(double[] in_weights, double[][] kernel_evaluations) {
		double[] weights = new double[this.sampled_indices.length];
		double const_1 = this._norm_const_1;

		if (this.h == null) {
			this.update_bandwidth(this.calculate_bandwidth(this.data));
		}

		if (in_weights == null) {
			Arrays.fill(weights, 1.0);
		} else {
			for (int i = 0; i < this.sampled_indices.length; i++) {
				weights[i] = in_weights[this.sampled_indices[i]];
			}
		}
		weights = Utils.normalize(weights);
		double[] out = new double[kernel_evaluations.length];
		double[] weighted_dot_prod = Utils.dot_product(kernel_evaluations, weights);
		for (int i = 0; i < kernel_evaluations.length; i++) {
			out[i] = const_1 * weighted_dot_prod[i];
		}
		return out;
	}

	public double[] pdf(double[][] query_points) {
		int N = query_points.length;
		double[] weights = new double[this.sampled_indices.length];
		Arrays.fill(weights, 1. / N);
		String random_label = ""; //TODO: make random label function
		return this.pdf(weights, query_points, random_label, false);
	}

	public double[] pdf(double[][] query_points, String label) {
		int N = query_points.length;
		double[] weights = new double[this.sampled_indices.length];
		Arrays.fill(weights, 1. / N);
		return this.pdf(weights, query_points, label, false);
	}

	public double[] pdf(double[] weights, double[][] query_points, String label, boolean do_not_cache) {
		// Compute weighted KDE for query points
		if (this.fixed_bandwidth && this.saved_kernel_evals.containsKey(label)) {
			return this._precomputed_density(weights, this.saved_kernel_evals.get(label));
		} else {
			// Not cached, recompute
			DensityReturnObj dro;
			double out[] = new double[query_points.length];
			double kes[][] = new double[query_points.length][];

			for (int i = 0; i < query_points.length; i++) {
				dro = this._density(weights, query_points[i]);
				out[i] = dro.result;
				kes[i] = dro.kernel_evals;
			}

			if (!do_not_cache) {
				this.saved_kernel_evals.put(label, kes);
			}

			return out;
		}
	}
}
