package bluesky;

import bluesky.enums.AlgorithmType;
import bluesky.enums.DE_Mode;

public class AlgorithmFactory {

	public static Algorithm getAlgorithm(AlgorithmType algoType, int K_max, double delta, boolean verbose, int fold_number, boolean track_weights,
			boolean fixed_bandwidth, boolean do_not_cache, boolean kmeans_init, int n_lightcone_subsamples, int dens_est_n_subsamples, int dens_est_n_pts_used,
			boolean fully_nonparamteric, int max_iterations, DE_Mode den_est_mode, double cluster_epsilon, double bandwidth_adjustment_factor) {

		if (algoType.equals(AlgorithmType.MIXEDLICORS))
		{
			return new MixedLICORS(K_max, delta, verbose, fold_number, track_weights, fixed_bandwidth, do_not_cache, kmeans_init, n_lightcone_subsamples,
					dens_est_n_subsamples, dens_est_n_pts_used, fully_nonparamteric, max_iterations, den_est_mode, bandwidth_adjustment_factor);
		}
		else if (algoType.equals(AlgorithmType.FLTP))
		{
			return new FLTP(fold_number);
		}
		else if (algoType.equals(AlgorithmType.OHP))
		{
			return new OHP(K_max, cluster_epsilon, fold_number, dens_est_n_subsamples, n_lightcone_subsamples, verbose, bandwidth_adjustment_factor);
		}
		else if (algoType.equals(AlgorithmType.LR))
		{
			return new LinearRegressor();
		}
		else if (algoType.equals(AlgorithmType.MOONSHINE))
		{
			return new Moonshine(K_max, cluster_epsilon, fold_number, bandwidth_adjustment_factor);
		}
		else
		{
			return null;
		}
	}
}
