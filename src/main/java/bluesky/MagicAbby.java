package bluesky;

import java.io.IOException;

import bluesky.enums.AlgorithmType;
import bluesky.enums.DE_Mode;

public class MagicAbby {

	//public static String DATA_DIR = "Data/frames_flat_color_txt/";
	//public static String RESULTS_DIR = "Results/";

	public static String DATA_DIR = "~/Desktop/frames_flat_color/".replaceFirst("^~",System.getProperty("user.home"));
	public static String RESULTS_DIR = "~/Desktop/Results-abby-color/".replaceFirst("^~",System.getProperty("user.home"));
		
	//public static String DATA_DIR = "~/Dropbox/GetAbby/Projects/JavaAbbyML/bluesky/Data/frames_flat/".replaceFirst("^~",System.getProperty("user.home"));
	//public static String RESULTS_DIR = "~/Dropbox/GetAbby/Projects/JavaAbbyML/bluesky/Results/".replaceFirst("^~",System.getProperty("user.home"));
	
	public static void main(String args[]) throws IOException {

		int NUM_FRAMES_TO_USE = 100; // NOTE: Due to how lightcone prediction works, you need at least 
									// NUM_FRAMES_TO_USE + h_p frames worth of data in your data directory. 
									// If using double ended light cones, you need NUM_FRAMES_TO_USE + 2 * h_p. 
		int NUM_LIGHTCONE_SAMPLES = 60000; // -1 = use all samples
		int folds = 10;
		int ITEMS_PER_OUTPUT_ROW = 1;
		int NUM_OF_STATES = 250;
		boolean RESCALE_DATA = false;
		int MAX_ITERATIONS = 100;
		double DELTA = 0.12;
		int PIXEL_DEPTH = 3;
		double BANDWIDTH_ADJUSTMENT_FACTOR = 2.0;
		boolean DOUBLE_ENDED = true;
		int H_P = 1;
		int C = 1;

		/*
		 * public ExperimentHarness(String experiment_name, AlgorithmType algoType, int[] dataGeometry, int num_frames, 
			int speed, int h_p, int K_max, double delta, boolean verbose, int num_folds, int fold_number,
			boolean track_weights, boolean fixed_bandwidth, boolean do_not_cache, boolean kmeans_init, 
			boolean buffer_frame, int n_subsamples, int dens_est_n_subsamples, int dens_est_n_pts_used, boolean fully_nonparamteric, 
			int max_iterations, DE_Mode den_est_mode, double cluster_epsilon, int items_per_row)
		 */
		/*
		ExperimentHarness eh = new ExperimentHarness("abby_replication", AlgorithmType.MIXEDLICORS,
				new int[] { NUM_FRAMES_TO_USE, 440, 330, PIXEL_DEPTH }, NUM_FRAMES_TO_USE, 1, 1, NUM_OF_STATES, DELTA, true, folds, 1,
				false, true, false, true, false, NUM_LIGHTCONE_SAMPLES, 500, 500, true, MAX_ITERATIONS, DE_Mode.FULL, 1e-3,
				ITEMS_PER_OUTPUT_ROW, RESCALE_DATA, BANDWIDTH_ADJUSTMENT_FACTOR);
		 */
		DoubleEndedExperimentHarness eh = new DoubleEndedExperimentHarness("abby_replication", AlgorithmType.LR,
				new int[] { NUM_FRAMES_TO_USE, 440, 330, PIXEL_DEPTH }, NUM_FRAMES_TO_USE, C, H_P, NUM_OF_STATES, DELTA, true, folds, 1,
				false, true, false, true, false, NUM_LIGHTCONE_SAMPLES, 500, 500, true, MAX_ITERATIONS, DE_Mode.FULL, 1e-3,
				ITEMS_PER_OUTPUT_ROW, RESCALE_DATA, BANDWIDTH_ADJUSTMENT_FACTOR, DOUBLE_ENDED);
		for (int trial = 0; trial < 1; trial++)
		{
			//assume we will need to turn these on and off based upon what args are sent in...will need to make this an restful webservice
			eh.exp_trial(DATA_DIR, RESULTS_DIR, "_LR", trial, 1, NUM_FRAMES_TO_USE, NUM_LIGHTCONE_SAMPLES, "double-ended");
		}
		System.out.println("ALL DONE!");
	}

}
