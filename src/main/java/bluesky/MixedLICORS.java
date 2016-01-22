package bluesky;

import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;

import utils.Utils;
import bluesky.enums.DE_Mode;

/**
 * @author gmontane
 *
 */
public class MixedLICORS extends LICORSmethod implements Algorithm {
	private int _MAX_ITERATIONS = 100;
	private double[][] _W;
	int K_max = -1;
	double delta = 1.0;
	boolean verbose = false;
	int fold_number = 1;
	boolean track_weights = false;
	boolean fixed_bandwidth = true;
	boolean do_not_cache = false;
	boolean kmeans_init = true;
	int subsample_limit = -1;
	int DE_subsample_limit = -1;
	int DE_num_pts_used = -1;
	boolean fully_nonparamteric = false;
	private double[][] _FLCs;
	private double[][] _PLCs;
	private double[][] _FLCs_validation;
	private double[][] _PLCs_validation;
	private double[][] _CE_weighted_PLC_means;
	private double[][] _CE_weighted_FLC_means;
	private double[][][] _CE_weighted_COV_matrix;
	WeightedKDE PLC_DE;
	WeightedKDE FLC_DE;
	private int[] _training_indices;
	private int[] _test_indices;
	public DE_Mode density_est_mode;
	double bandwidth_adjustment_factor;

	public MixedLICORS(int K_max, double delta, boolean verbose, int fold_number, boolean track_weights,
			boolean fixed_bandwidth, boolean do_not_cache, boolean kmeans_init, int n_lightcone_subsamples, int dens_est_n_subsamples,
			int dens_est_n_pts_used, boolean fully_nonparamteric, int max_iterations, DE_Mode den_est_mode, double bandwidth_adjustment_factor) {
		this._MAX_ITERATIONS = max_iterations;
		this._W = _random_W_initialization(1, K_max);
		this.K_max = K_max;
		this.verbose = verbose;
		this.delta = delta;
		this.fold_number = fold_number;
		this.track_weights = track_weights;
		this.fixed_bandwidth = fixed_bandwidth;
		this.do_not_cache = do_not_cache;
		this.kmeans_init = kmeans_init;
		this.subsample_limit = n_lightcone_subsamples;
		this.DE_subsample_limit = dens_est_n_subsamples;
		this.DE_num_pts_used = dens_est_n_pts_used;
		this.fully_nonparamteric = fully_nonparamteric;
		this.density_est_mode = den_est_mode;
		this.bandwidth_adjustment_factor = bandwidth_adjustment_factor;
	}

	public int N() {
		return this._W.length;
	}

	public int K() {
		return this._W[0].length;
	}

	private double[][] _random_W_assignment_initialization(int N, int K) {
		double[][] W = new double[N][K];
		int[] assignments = new int[N];

		for (int i = 0; i < N; i++) {
			assignments[i] = Utils.randInt(0, K - 1);
		}

		for (int i = 0; i < N; i++) {
			W[i][assignments[i]] = 1.0;
		}

		// Make sure each state has at least one light cone
		for (int j = 0; j < K; j++) {
			int sum = 0;
			for (int i = 0; i < N; i++) {
				sum += W[i][j];
			}
			if (sum < 1) {
				int row_index = Utils.randInt(0, N - 1);
				for (int k = 0; k < K; k++) {
					W[row_index][k] = 0.0;
				}
				W[row_index][j] = 1;
			}
		}

		return W;
	}

	private double[][] _random_W_initialization(int N, int K) {
		double[][] W = new double[N][K];

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < W[i].length; j++) {
				W[i][j] = Math.random();
			}
		}
		return Utils.normalize_rows(W);
	}

	private double[][] _kmeanspp_initialization(int N, int K) {
		int[] assignments = ClusteringMethodObj.kmeanspp(this._FLCs, K);
		double[][] W = new double[N][K];

		for (int i = 0; i < N; i++) {
			W[i][assignments[i]] = 1.0;
		}

		return W;
	}

	private double[][] _create_initialized_W_matrix(int N, int K) {
		if (this.kmeans_init) {
			return this._kmeanspp_initialization(N, K);
		} else {
			return this._random_W_initialization(N, K);
		}
	}

	public void import_model(double[][] W) {
		this._W = Utils.deep_copy(W);
		this._refresh_current_estimates();
	}

	private void _refresh_current_estimates() {
		if (this._FLCs == null || this._FLCs.length == 0) {
			return;
		}

		this._CE_weighted_FLC_means = new double[this.K()][];
		this._CE_weighted_PLC_means = new double[this.K()][];
		this._CE_weighted_COV_matrix = new double[this.K()][][];

		for (int j = 0; j < this._W[0].length; j++) {
			double[] weights = Utils.get_column_slice(this._W, j);
			double[] normalized_weights = Utils.normalize(weights);
			this._CE_weighted_FLC_means[j] = Utils.weighted_mean(normalized_weights, this._FLCs);
			this._CE_weighted_PLC_means[j] = Utils.weighted_mean(normalized_weights, this._PLCs);
			if (!this.fully_nonparamteric) {
				this._CE_weighted_COV_matrix[j] = Utils.weighted_covariance(normalized_weights, this._PLCs);
			}
		}
	}

	public void load_light_cones(double[][] PLCs, double[][] FLCs) {
		// Split into training and test/hold out sets
		assert (PLCs.length == FLCs.length);
		int N = PLCs.length;
		int[] indices = Utils.randPermutation(N);
		int stop = (int) (0.75 * N);
		this._training_indices = Utils.slice(indices, 0, stop);
		this._test_indices = Utils.slice(indices, stop, N - 1);
		this._FLCs = Utils.select_rows(this._training_indices, FLCs);
		this._PLCs = Utils.select_rows(this._training_indices, PLCs);
		this._FLCs_validation = Utils.select_rows(this._test_indices, FLCs);
		this._PLCs_validation = Utils.select_rows(this._test_indices, PLCs);

		//Covariance covObj = new Covariance(this._FLCs);
		//this._default_covariance_matrix = covObj.getCovarianceMatrix().getData();
		this.PLC_DE = new WeightedKDE(this._PLCs, this.DE_subsample_limit, this.DE_num_pts_used, this.density_est_mode, this.fixed_bandwidth, this.bandwidth_adjustment_factor);
		this.FLC_DE = new WeightedKDE(this._FLCs, this.DE_subsample_limit, this.DE_num_pts_used, this.density_est_mode, this.fixed_bandwidth, this.bandwidth_adjustment_factor);
		this.PLC_DE.update_bandwidth(this.PLC_DE.calculate_bandwidth(this._PLCs));
		this.FLC_DE.update_bandwidth(this.FLC_DE.calculate_bandwidth(this._FLCs));
	}

	private double[][] _f_hat_conditional_densities(double[][] query_points, String label, boolean no_cache) {
		// Uses weighted KDE to estimate f(x_i | S_i = s_j) (eq. 21 of MixedLICORS paper), for each extremal 
		// state j. Returns array of real-valued density estimates, rows index each point, columns are for 
		// each component state.
		double[][] out = new double[query_points.length][this.K()];
		for (int j = 0; j < this.K(); j++) {
			if (!this.fixed_bandwidth) {
				int[] argmax_indices = Utils.matching_argmax_indices(this._W, j);
				double[][] argmax_data = Utils.select_rows(argmax_indices, this._FLCs);
				if (argmax_data.length > 1)
					this.FLC_DE.update_bandwidth(this.FLC_DE.calculate_bandwidth(argmax_data));
			}
			double[] weights = Utils.get_column_slice(this._W, j);
			double[] res = this.FLC_DE.pdf(weights, query_points, label, no_cache || this.do_not_cache);
			for (int i = 0; i < query_points.length; i++) {
				out[i][j] = res[i];
			}
		}
		return out;
	}

	private double[][] _p_hat_conditional_probs(double[][] query_points, String label, boolean no_cache) {
		// Get estimates of P(S_i = s_j | PLC) for every state s_j.
		double[][] out = new double[query_points.length][this.K()];
		for (int j = 0; j < this.K(); j++) {
			double[] densities = this.PLC_densities(j, query_points, String.format("%s_%d", label, j), no_cache);
			for (int i = 0; i < query_points.length; i++) {
				out[i][j] = densities[i];
			}
		}
		return out;
	}

	public double[] PLC_densities(int j, double[][] query_points, String label, boolean no_cache) {
		if (this.fully_nonparamteric) {
			return this.PLC_densities_nonparametric(j, query_points, label, no_cache);
		} else {
			return this.PLC_densities_Gaussian(j, query_points);
		}
	}

	public double[] PLC_densities_nonparametric(int j, double[][] query_points, String label, boolean no_cache) {
		if (!this.fixed_bandwidth) {
			int[] argmax_indices = Utils.matching_argmax_indices(this._W, j);
			double[][] argmax_data = Utils.select_rows(argmax_indices, this._PLCs);
			if (argmax_data.length > 1)
				this.PLC_DE.update_bandwidth(this.PLC_DE.calculate_bandwidth(argmax_data));
		}
		double[] weights = Utils.get_column_slice(this._W, j);
		return this.PLC_DE.pdf(weights, query_points, label, no_cache || this.do_not_cache);
	}

	public double[] PLC_densities_Gaussian(int j, double[][] query_points) {
		double[] mean = this._CE_weighted_PLC_means[j];
		double[][] cov = this._CE_weighted_COV_matrix[j];
		MultivariateNormalDistribution mvn = new MultivariateNormalDistribution(mean, cov);
		int M = query_points.length;
		double[] out = new double[M];
		for (int i = 0; i < M; i++) {
			out[i] = mvn.density(query_points[i]);
		}
		return out;
	}

	private void _update_weights(boolean no_cache) {
		// Updates all weights in W weight matrix using current estimates.
		int N = this.N();
		int K = this.K();
		if (this.verbose)
			System.out.println("Refreshing current estimates...");
		this._refresh_current_estimates();
		double[][] f_hat_matrix = this._f_hat_conditional_densities(this._FLCs, "TRAIN_FLCs", no_cache);
		for (int j = 0; j < K; j++) {
			if (this.verbose)
				System.out.println("Updating state " + j);
			double[] density_array = this.PLC_densities(j, this._PLCs, "UPDATE_WEIGHTS", no_cache);
			double weight_sum = Utils.sum(Utils.get_column_slice(this._W, j));
			double n_hat_ratio = weight_sum / N;
			double[] f_hat_slice = Utils.get_column_slice(f_hat_matrix, j);
			for (int i = 0; i < N; i++) {
				this._W[i][j] = f_hat_slice[i] * density_array[i] * n_hat_ratio;
			}
		}
		this._renormalize_weights();
		this._refresh_current_estimates();
	}

	private void _renormalize_weights() {
		this._W = Utils.normalize_rows(this._W);
	}

	private void _remove_small_weight_states() {
		double[] state_weight_totals = Utils.axis_sum(this._W, 0);
		int[] strong_states = new int[] {};
		int[] weak_states = new int[] {};
		// Find strong states
		for (int j = 0; j < state_weight_totals.length; j++) {
			if (state_weight_totals[j] > 1.5) {
				strong_states = ArrayUtils.add(strong_states, j);
			} else {
				weak_states = ArrayUtils.add(weak_states, j);
			}
		}
		// Remove weak states
		if (weak_states.length > 0) {
			if (this.verbose)
				System.out.println("Removing states: " + Arrays.toString(weak_states));
			double[][] reduced_W = new double[this._W.length][strong_states.length];
			for (int s = 0; s < strong_states.length; s++) {
				for (int i = 0; i < reduced_W.length; i++) {
					reduced_W[i][s] = this._W[i][strong_states[s]];
				}
			}
			// Save new matrix
			this._W = Utils.deep_copy(reduced_W);
			// Update parameters
			this._renormalize_weights();
			this._refresh_current_estimates();
		}
	}

	public double[][] predict_batch(double[][] points, String points_label) {
		int M = points.length;
		int chunksize = 1000;
		double[][] out = new double[M][];
		for (int i = 0; i < points.length; i += chunksize) {
			if (this.verbose)
				System.out.print("Predicting chunk starting with " + i + "...");
			double[][] chunk = Utils.slice(points, i, i + chunksize);
			double[][] res = this.predict_batch(chunk, points_label + '_' + i, true);
			for (int j = 0; j < res.length; j++) {
				out[i + j] = res[j];
			}
			if (this.verbose)
				System.out.println("Done.");
		}
		return out;
	}

	public double[][] predict_batch(double[][] points, String points_label, boolean no_cache) {
		int M = points.length;
		int K = this.K();
		int N = this.N();
		double[][] lightcone_densities = new double[K][M];
		for (int j = 0; j < K; j++) {
			lightcone_densities[j] = this.PLC_densities(j, points, points_label, no_cache);
		}
		double[] n_hats = Utils.axis_sum(this._W, 0);
		for (int j = 0; j < K; j++) {
			n_hats[j] /= N;
		}
		double[][] raw_component_weights = new double[M][K];
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < K; j++) {
				raw_component_weights[i][j] = lightcone_densities[j][i] * n_hats[j];
			}
		}
		double[][] normalized_component_weights = Utils.normalize_rows(raw_component_weights);
		return Utils.matrix_multiply(normalized_component_weights, this._CE_weighted_FLC_means);
	}

	public int[] predict_states(double[][] points, boolean no_cache) {
		int N = this.N();
		int K = this.K();
		int M = points.length;
		double[][] lightcone_densities = new double[K][M];
		for (int j = 0; j < K; j++) {
			lightcone_densities[j] = this.PLC_densities(j, points, "PREDICT_STATES", no_cache);
		}
		double[] n_hats = Utils.axis_sum(this._W, 0);
		for (int j = 0; j < K; j++) {
			n_hats[j] /= N;
		}
		double[][] raw_component_weights = new double[N][K];
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < K; j++) {
				raw_component_weights[i][j] = lightcone_densities[j][i] * n_hats[j];
			}
		}
		double[][] normalized_component_weights = Utils.normalize_rows(raw_component_weights);
		int[] state_assignments = Utils.rowwise_argmax(normalized_component_weights);
		return state_assignments;
	}

	private void _merge_closest_states(boolean no_cache) {
		double[][] results_matrix = this._f_hat_conditional_densities(this._FLCs, "ALL_FLCs", no_cache);
		MergeStatesStructure smallest_distance = new MergeStatesStructure(Double.POSITIVE_INFINITY, -1, -1);
		for (int i = 0; i < this.K(); i++) {
			for (int j = i + 1; j < this.K(); j++) {
				double[] col_i = Utils.get_column_slice(results_matrix, i);
				double[] col_j = Utils.get_column_slice(results_matrix, j);
				double d = Utils.distributional_distance(col_i, col_j);
				if (d < smallest_distance.distance) {
					smallest_distance = new MergeStatesStructure(d, i, j);
				}
			}
		}
		int a = smallest_distance.state_i;
		int b = smallest_distance.state_j;
		if (this.verbose)
			System.out.println("Merging states " + a + " and " + b);
		for (int row = 0; row < this.N(); row++) {
			this._W[row][a] += this._W[row][b];
		}
		this._W = Utils.delete_column(this._W, b);
	}

	public double[] compute_likelihoods(double[][] PLCs, double[][] FLCs, String label) {
		int M = PLCs.length;
		int chunksize = 1000;
		double[] out = new double[M];
		for (int i = 0; i < PLCs.length; i += chunksize) {
			if (this.verbose)
				System.out.print("Computing likelihood for chunk starting with " + i + "...");
			double[][] chunk_PLCs = Utils.slice(PLCs, i, i + chunksize);
			double[][] chunk_FLCs = Utils.slice(FLCs, i, i + chunksize);
			double[] res = this.compute_likelihoods(chunk_PLCs, chunk_FLCs, label + '_' + i, true);
			for (int j = 0; j < res.length; j++) {
				out[i + j] = res[j];
			}
			if (this.verbose)
				System.out.println("Done.");
		}
		return out;
	}

	public double[] compute_likelihoods(double[][] PLCs, double[][] FLCs, String label, boolean no_cache) {
		double[][] future_given_state_probs = this._f_hat_conditional_densities(FLCs, label, no_cache);
		double[][] state_given_past_probs = this._p_hat_conditional_probs(PLCs, label, no_cache);
		// Weight by state likelihood
		double[] n_hats = Utils.axis_sum(this._W, 0);
		int K = this.K();
		int N = this.N();
		int M = PLCs.length;
		for (int j = 0; j < K; j++) {
			n_hats[j] /= (double) N;
			for (int i = 0; i < M; i++) {
				state_given_past_probs[i][j] *= n_hats[j];
			}
		}
		// Normalize
		state_given_past_probs = Utils.normalize_rows(state_given_past_probs);
		// Return mixed likelihoods
		double[] out = new double[M];
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < K; j++) {
				out[i] += future_given_state_probs[i][j] * state_given_past_probs[i][j];
			}
		}
		return out;
	}

	public double log_likelihood(double[][] PLCs, double[][] FLCs, String label) {
		return Utils.sum_log(this.compute_likelihoods(PLCs, FLCs, label));
	}

	public double get_current_MSE(boolean no_cache) {
		double[][] preds = this.predict_batch(this._PLCs_validation, "TRAINING_TEST_PLCs", no_cache);
		double out = 0.0;
		int count = 0;
		for (int i = 0; i < preds.length; i++) {
			for (int j = 0; j < preds[i].length; j++) {
				out += Math.pow((this._FLCs_validation[i][j] - preds[i][j]), 2);
				count++;
			}
		}
		return out / (double) count;
	}

	public void learn(double[][] histories, double[][] futures) {
		// Main loop of Mixed LICORS
		BestWStructure best_W_structure = null;
		if (this.subsample_limit > 0) {
			// Sub-sample a smaller set of light cones for tractability
			int limit = Utils.min(this.subsample_limit, futures.length);
			int[] random_indices = Utils.randPermutation(futures.length);
			int[] retained_indices = Utils.slice(random_indices, 0, limit);
			futures = Utils.select_rows(retained_indices, futures);
			histories = Utils.select_rows(retained_indices, histories);
		}
		// Load light cones
		this.load_light_cones(histories, futures);
		// Initialize weight matrix
		this._W = this._create_initialized_W_matrix(this._FLCs.length, this.K_max);
		// Iterate
		while (this.K() > 0) {
			boolean converged = false;
			int iteration = 0;
			// Run until convergence or max iterations reached. 
			while (!converged && iteration < this._MAX_ITERATIONS) {
				this._remove_small_weight_states();
				double[][] old_W = Utils.deep_copy(this._W);
				this._update_weights(this.do_not_cache);
				double MSE = this.get_current_MSE(this.do_not_cache);
				if (best_W_structure == null || best_W_structure.MSE > MSE) {
					if (this.verbose)
						System.out.println("New Best MSE " + MSE + " " + this.K());
					best_W_structure = new BestWStructure(MSE, this._W);
				}
				double difference = this._difference_of_W_matrices(this._W, old_W);
				converged = (difference < this.delta);
				if (this.verbose)
					System.out.println("Normed Difference " + difference + " " + delta + " " +
							converged + " " + this.K() + " " + best_W_structure.MSE + " " + MSE);
				iteration++;
			}
			// Merge step
			if (this.K() == 1) {
				break;
			} else {
				this._merge_closest_states(this.do_not_cache);
			}
		}
		// Load model with lowest out-of-sample MSE
		this.import_model(best_W_structure.W);
	}

	private double _difference_of_W_matrices(double[][] A, double[][] B) {
		assert (!Utils.is_empty(A) && !Utils.is_empty(B));
		assert (A.length == B.length);
		int N = A.length;
		int K = A[0].length;
		double[] res = new double[N];
		for (int i = 0; i < N; i++) {
			double row_mean = 0.0;
			for (int j = 0; j < K; j++) {
				row_mean += Math.pow((A[i][j] - B[i][j]), 2);
			}
			row_mean /= K;
			res[i] = Math.sqrt(row_mean);
		}
		return Utils.max(res);
	}

	public void clear() {
		this._W = null;
		this.FLC_DE.clear();
		this.PLC_DE.clear();
		this._FLCs = null;
		this._PLCs = null;
		this._FLCs_validation = null;
		this._PLCs_validation = null;
	}
}
