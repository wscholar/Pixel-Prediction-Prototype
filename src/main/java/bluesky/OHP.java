package bluesky;

import utils.Utils;

public class OHP implements Algorithm {
	private int _K_max;
	public int fold_number = 1;
	public PredictiveState[] states;
	public int[] cluster_assignments;
	private int _num_samples = -1;
	private int _num_pts_used = -1;
	private double[][] _states_mean_PLC_matrix;
	private double[][] _states_mean_FLC_matrix;
	private int _CHUNK_SIZE = 250;
	private double bandwidth_adjustment_factor;
	
	public OHP(int k_max, double cluster_epsilon, int fold_number, int num_pts_used) {
		this._K_max = k_max;
		this.fold_number = fold_number;
		this.states = new PredictiveState[]{};
		this.cluster_assignments = new int[]{};
		this._num_pts_used = num_pts_used;
	}

	public OHP(int k_max, double cluster_epsilon, int fold_number, int num_pts_used, int num_samples) {
		this._K_max = k_max;
		this.fold_number = fold_number;
		this.states = new PredictiveState[]{};
		this.cluster_assignments = new int[]{};
		this._num_samples = num_samples;
	}
	
	public OHP(int k_max, double cluster_epsilon, int fold_number, int num_pts_used, int num_samples, boolean verbose, double bandwidth_adjustment_factor) {
		this._K_max = k_max;
		this.fold_number = fold_number;
		this.states = new PredictiveState[]{};
		this.cluster_assignments = new int[]{};
		this._num_samples = num_samples;
		this.bandwidth_adjustment_factor = bandwidth_adjustment_factor;
	}

	public double[][] predict_batch(double[][] PLCs, String label) {
		int M = PLCs.length;
		double[][] out = new double[M][];
		for (int i = 0; i < PLCs.length; i += this._CHUNK_SIZE) { 
			System.out.print("Predicting chunk starting with " + i + "...");
			double[][] chunk = Utils.slice(PLCs, i, i + this._CHUNK_SIZE);
			double[][] res = this._predict_batch(chunk, label + '_' + i);
			for (int j = 0; j < res.length; j++) {
				out[i+j] = res[j];
			}
			System.out.println("Done.");
		}
		return out;
	}
	
	
	public double[][] _predict_batch(double[][] PLCs, String label) {
		// Uses weighted average of states 
		double[][] state_given_past_probs = this.states_PLC_densities(PLCs, label);
		for (int i = 0; i < state_given_past_probs.length; i++) {
			for (int j = 0; j < this.states.length; j++) {
				state_given_past_probs[i][j] = Math.nextAfter(state_given_past_probs[i][j], 1.0);
			}
		}
		// Weight by state likelihood
		for (int j = 0; j < this.states.length; j++) {
			for (int i = 0; i < state_given_past_probs.length; i++) {
				state_given_past_probs[i][j] = Math.nextAfter(state_given_past_probs[i][j] * 
						this.states[j].state_likelihood(), 1.0);
			}
		}
		//  Normalize
		state_given_past_probs = Utils.normalize_rows(state_given_past_probs);
		return Utils.matrix_multiply(state_given_past_probs, this._states_mean_FLC_matrix);
	}

	public double[][] states_FLC_densities(double[][] FLCs, String label) {
		int N = FLCs.length;
		int K = this.states.length;
		double[][] out = new double[N][K];
		for (int j = 0; j < K; j++) {
			double[] res = this.states[j].batch_FLC_densities(FLCs, label);
			for (int i = 0; i < N; i++) {
				out[i][j] = res[i];
			}
		}
		return out;
	}
	
	public double[][] states_PLC_densities(double[][] PLCs, String label) {
		int N = PLCs.length;
		int K = this.states.length;
		double[][] out = new double[N][K];
		for (int j = 0; j < K; j++) {
			double[] res = this.states[j].batch_PLC_densities(PLCs, label);
			for (int i = 0; i < N; i++) {
				out[i][j] = res[i];
			}
		}
		return out;
	}
	
	public double[] compute_likelihoods(double[][] PLCs, double[][] FLCs, String label) {
		double[][] future_given_state_probs = this.states_FLC_densities(FLCs, label);
		double[][] state_given_past_probs = this.states_PLC_densities(PLCs, label);
		for (int i = 0; i < state_given_past_probs.length; i++) {
			for (int j = 0; j < this.states.length; j++) {
				state_given_past_probs[i][j] = Math.nextAfter(state_given_past_probs[i][j], 1.0);	
				future_given_state_probs[i][j] = Math.nextAfter(future_given_state_probs[i][j], 1.0);
			}
		}
		// Weight by state likelihood
		for (int j = 0; j < this.states.length; j++) {
			for (int i = 0; i < state_given_past_probs.length; i++) {
				state_given_past_probs[i][j] = Math.nextAfter(state_given_past_probs[i][j] * 
						this.states[j].state_likelihood(), 1.0);
			}
		}
		//  Normalize
		state_given_past_probs = Utils.normalize_rows(state_given_past_probs);
		// Return mixed likelihoods
		double[] results = Utils.axis_sum(Utils.elementwise_multiply(future_given_state_probs, state_given_past_probs), 1);
		for (int i = 0; i < results.length; i++) {
			results[i] = Math.nextAfter(results[i], 1.0);
		}
		return results;
	}
	
	public double log_likelihood(double[][] PLCs, double[][] FLCs, String label){
		return Utils.sum_log(this.compute_likelihoods(PLCs, FLCs, label));
	}
	
	public double total_prediction_MSE(double[][] pasts, double[][] true_futures) {
		double[][] preds = this.predict_batch(pasts, "PLCs");
		double out = 0.0;
		for (int i = 0; i < preds.length; i++) {
			for (int j = 0; j < preds[i].length; j++) {
				out += Math.pow((true_futures[i][j] - preds[i][j]), 2);
			}
		}
		return out / preds.length;
	}
	
	public void learn(double[][] PLCs, double[][]FLCs){
		// Cluster by futures		
		this.cluster_assignments = ClusteringMethodObj.kmeanspp(FLCs, this._K_max);
		int valid_state_count = this._K_max;
		int min_pts_per_state = 5;
		
		// Make sure each state has at least two PLCs. Valid state count is for those that do.
		for (int j = 0; j < this._K_max; j++) {
			int[] state_indices = Utils.find_matching_rows(this.cluster_assignments, j);
			if (state_indices.length < min_pts_per_state) {
				valid_state_count -= 1;
			}
		}
		
		this.states = new PredictiveState[valid_state_count];
		System.out.println("Actual State Count: " + valid_state_count);
		int state_index = 0;
		
		// Create states from assignments
		for (int j = 0; j < this._K_max; j++) {
			int[] state_indices = Utils.find_matching_rows(this.cluster_assignments, j);
			if (state_indices.length < min_pts_per_state) {
				continue;
			}
			double[][] state_PLCs = Utils.select_rows(state_indices, PLCs);
			double[][] state_FLCs = Utils.select_rows(state_indices, FLCs);
			PredictiveState ps = new PredictiveState(state_PLCs, state_FLCs, PLCs.length, this._num_pts_used, this._num_samples, this.bandwidth_adjustment_factor);
			this.states[state_index++] = ps;
		}
		
		// Create mean futures and histories for states
		if (this.states.length > 0) {
			this._states_mean_PLC_matrix = new double[this.states.length][this.states[0].mean_PLC.length];
			this._states_mean_FLC_matrix = new double[this.states.length][this.states[0].mean_FLC.length];
			for (int j = 0; j < this.states.length; j++) {
				this._states_mean_PLC_matrix[j] = this.states[j].mean_PLC;
				this._states_mean_FLC_matrix[j] = this.states[j].mean_FLC;
			}
		}
	}  	
	
	public void clear() {
		for (int j = 0; j < this._K_max; j++) {
			this.states[j].clear();
		}
	}
}
