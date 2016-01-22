package bluesky;

import java.io.File;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.Arrays;
import java.util.Scanner;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.RandomStringUtils;

import utils.FileUtility;
import utils.Utils;
import bluesky.enums.AlgorithmType;
import bluesky.enums.DE_Mode;

public class DoubleEndedExperimentHarness {

	private int num_frames = 0;
	private int num_folds = 0;
	private String exp_name = "";
	private String resultsdir = "";
	private int speed = 1;
	private int h_p = 1;
	private boolean buffer_frame = false;
	private boolean do_not_cache = true;
	private int[] dataGeometry = null;
	private AlgorithmType algoType;
	private int K_max = 100;
	private double delta = .0019;
	private boolean verbose = false;
	private int fold_number = 1;
	private boolean track_weights = false;
	private boolean fixed_bandwidth = false;
	private boolean kmeans_init = true;
	private int n_subsamples = -1;
	private int dens_est_n_subsamples = 1000;
	private int dens_est_n_pts_used = 500;
	private boolean fully_nonparamteric = true;
	private int max_iterations = 1000;
	private DE_Mode den_est_mode = DE_Mode.FULL;
	private double cluster_epsilon = 1e-10;
	private int items_per_output_row;
	private boolean rescale_data;
	private double bandwidth_adjustment_factor;
	private LightConeExtractor lce;
	private boolean double_ended = false;

	public DoubleEndedExperimentHarness(String experiment_name, AlgorithmType algoType, int[] dataGeometry, int num_frames, 
			int speed, int h_p, int K_max, double delta, boolean verbose, int num_folds, int fold_number,
			boolean track_weights, boolean fixed_bandwidth, boolean do_not_cache, boolean kmeans_init, 
			boolean buffer_frame, int n_subsamples, int dens_est_n_subsamples, int dens_est_n_pts_used, boolean fully_nonparamteric, 
			int max_iterations, DE_Mode den_est_mode, double cluster_epsilon, int items_per_row, boolean rescale_data_standard,
			double bandwidth_adjustment, boolean double_ended)
	{	
		this.exp_name = experiment_name;
		this.algoType = algoType;
		this.dataGeometry = ArrayUtils.clone(dataGeometry);
		this.speed = speed;
		this.h_p = h_p;
		this.num_frames = num_frames;
		this.K_max = K_max;
		this.delta = delta;
		this.verbose = verbose;
		this.num_folds = num_folds;
		this.fold_number = fold_number;
		this.track_weights = track_weights;
		this.fixed_bandwidth = fixed_bandwidth;
		this.do_not_cache = do_not_cache;
		this.kmeans_init = kmeans_init;
		this.buffer_frame = buffer_frame;
		this.n_subsamples = n_subsamples;
		this.dens_est_n_subsamples = dens_est_n_subsamples;
		this.dens_est_n_pts_used = dens_est_n_pts_used;
		this.fully_nonparamteric = fully_nonparamteric;
		this.max_iterations = max_iterations;
		this.den_est_mode = den_est_mode;
		this.cluster_epsilon = cluster_epsilon;
		this.items_per_output_row = items_per_row;
		this.rescale_data = rescale_data_standard;
		this.bandwidth_adjustment_factor = bandwidth_adjustment;
		this.double_ended = true;
	}

	public double[] get_data_from_path(String path, int num_of_frames) throws IOException
	{
		double[][] array_set = new double[][]{};
		File directory = new File(path);
		File[] files = directory.listFiles();

		if (files != null)
		{
			Arrays.sort(files);

			for (File f : files)
			{
				if (array_set.length >= num_of_frames + this.h_p) break;
				
				if (f.getName().endsWith(".txt") && !f.getName().startsWith("."))
				{
					double[] data = new double[FileUtility.count(f)];
					
					Scanner scan = new Scanner(f);
					int i = 0;
					while(scan.hasNextDouble())
				    {
						data[i++] = scan.nextDouble();
				    }
					scan.close();
					array_set = ArrayUtils.add(array_set, data);
					
				}
			}
		}

		return Utils.ravel(array_set);
	}
	
	public LightConesCollection fetch_light_cones(double[] raveled_data, int[] data_geometry, int num_samples)
	{
		if (this.lce == null) {
			this.lce = new LightConeExtractor(this.speed, this.h_p, this.rescale_data, true);
		}
		LightConesCollection lcc;
		lcc = this.lce.extract(raveled_data, data_geometry, num_samples);
		return lcc;
	}
	
	private double[] _extract_frames_data(double[] raveled_data, int[] data_geometry, int start, int end) {
		int frame_length = Utils.product(Utils.slice(data_geometry, 1, data_geometry.length));
		int raveled_start = start * frame_length;
		int raveled_end = end * frame_length;
		return Utils.slice(raveled_data, raveled_start, raveled_end);
	}
		
	public LightConesSet train_test_split(double[] raveled_data, int[] data_geometry, int test_start, int test_end, String results_path) throws IOException 
	{
		int slackened_start = test_start - this.h_p;
		int num_frames = data_geometry[0];
		int num_frames_test = test_end - slackened_start;
		int num_frames_before = slackened_start;
		int num_frames_after = num_frames - test_end;
		int num_samples = this.n_subsamples;
		int num_samples_before = (int) (num_samples * ((double) num_frames_before / (double) num_frames));
		int num_samples_after = num_samples - num_samples_before;
		
		int[] chunk_geometry = data_geometry.clone();
		chunk_geometry[0] = num_frames_test;
		LightConesCollection lcctest = fetch_light_cones(this._extract_frames_data(raveled_data, data_geometry, slackened_start, test_end), chunk_geometry, -1);
		// Save actual test frames
		int i = 0;
		int cutoff = Utils.cutoff(data_geometry, h_p, speed);
		while (i < lcctest.get_FLCs().length)
		{
			int slice_num = (i / cutoff) + 1;
			save_frames(results_path, lcctest.get_FLCs_range(i, i + cutoff), slice_num + "_0", this.items_per_output_row);
			i += cutoff;
		}
		chunk_geometry[0] = 1 + this.h_p * 2;
		double[] test_frames_data = this._extract_frames_data(raveled_data, data_geometry, slackened_start, test_end);
		lcctest = new LightConesCollection(new double[][]{}, new double[][]{}, new CoordTuple[]{});
		// extract inserted double ended light cone structures
		for (int index = this.h_p; index < num_frames_test - this.h_p; index++) {
			double[] tf_extracted = _extract_frames_data(test_frames_data, chunk_geometry, index - this.h_p, index + this.h_p);
			tf_extracted = Utils.add_pad_frame(tf_extracted, chunk_geometry, this.h_p);
			lcctest = LightConesCollection.merge(lcctest, fetch_light_cones(tf_extracted, chunk_geometry, -1));			
		}
		chunk_geometry[0] = num_frames_before;
		LightConesCollection lcctrainbefore = fetch_light_cones(this._extract_frames_data(raveled_data, data_geometry, 0, slackened_start), chunk_geometry, num_samples_before);
		chunk_geometry[0] = num_frames_after;
		LightConesCollection lcctrainafter = fetch_light_cones(this._extract_frames_data(raveled_data, data_geometry, test_end, data_geometry[0]), chunk_geometry, num_samples_after);
		LightConesCollection lcctrain = LightConesCollection.merge(lcctrainbefore, lcctrainafter);
		
		return new LightConesSet(lcctrain, lcctest);
	}

	public static void save_frames(String dir_path, double[][] frames, String name, int items_per_row) throws IOException
	{
		File f = FileUtility.createDirIfNotExists(dir_path + "/" + name + "_actual.txt");
		FileUtility.writeStringToFile(f, Utils.ArrayToRowString(Utils.ravel(frames), items_per_row));
	}
	
	public static void save_results(String dir_path, double[][] truth, double[][] predicted, double[][] error, int slice_num, int items_per_row) throws IOException
	{
		File f = FileUtility.createDirIfNotExists(dir_path + "/" + slice_num + "_actual.txt");
		FileUtility.writeStringToFile(f, Utils.ArrayToRowString(Utils.ravel(truth), items_per_row));
		f = FileUtility.createDirIfNotExists(dir_path + "/" + slice_num + "_predicted.txt");
		FileUtility.writeStringToFile(f, Utils.ArrayToRowString(Utils.ravel(predicted), items_per_row));
		f = FileUtility.createDirIfNotExists(dir_path + "/" + slice_num + "_error.txt");
		FileUtility.writeStringToFile(f, Utils.ArrayToRowString(Utils.ravel(error), items_per_row) );

	}
	
	public static void save_likelihoods(String dir_path, double[] likelihoods, int slice_num) throws IOException {
		File f = FileUtility.createDirIfNotExists(dir_path + "/" + slice_num + "_probs.txt");
		FileUtility.writeStringToFile(f, Utils.ArrayToRowString(likelihoods, 1));
	}

	
	public void do_fold(LightConesSet lcSet, int fold, String FOLD_RESULTS_DIR_PATH, int number_lightcone_samples, 
			int cutoff) throws IOException
	{	
		
		Algorithm algo = AlgorithmFactory.getAlgorithm(this.algoType, this.K_max, this.delta, this.verbose, fold, 
				this.track_weights, this.fixed_bandwidth, this.do_not_cache, this.kmeans_init, 
				number_lightcone_samples, this.dens_est_n_subsamples, this.dens_est_n_pts_used, this.fully_nonparamteric, 
				this.max_iterations, this.den_est_mode, this.cluster_epsilon, this.bandwidth_adjustment_factor);
		
		LightConesCollection train_coords_and_data = lcSet.getTrainCordAndData();
		algo.learn(train_coords_and_data.get_PLCs(), train_coords_and_data.get_FLCs());

		int i = 0;
		LightConesCollection test_coords_and_data = lcSet.getTestCordAndData();
		while (i < test_coords_and_data.get_FLCs().length)
		{
			String chunk_name = "CHUNK_" + fold + "_" + i;
			int slice_num = (i / cutoff) + 1;
			double[][] chunk = test_coords_and_data.get_PLCs_range(i, i + cutoff);
			double[][] truth = test_coords_and_data.get_FLCs_range(i, i + cutoff);
			double[][] preds = algo.predict_batch(chunk, chunk_name);
			// Descale if scaled
			if (this.rescale_data) {
				truth = this.lce.flc_scaler_obj.descale(truth);
				preds = this.lce.flc_scaler_obj.descale(preds);
			}
			
			double[][] error = Utils.getAbsForArrays(preds, truth);
			save_results(FOLD_RESULTS_DIR_PATH, truth, preds, error, slice_num, this.items_per_output_row);
			i = i + cutoff;
		}

		// Get rid of internally allocated variables
		algo.clear();
	}
	
	public void exp_trial(String data_dir, String results_root_dir, String results_suffix, int trial, int start_fold, 
			int num_of_frames, int number_lightcone_samples, String group_folder) throws IOException {
		
		resultsdir = results_root_dir + group_folder + results_suffix;
		int cutoff = Utils.cutoff(dataGeometry, h_p, speed);
		LightConesSet lcSet;
		double[] data = get_data_from_path(data_dir, num_of_frames);
		int FOLD_SIZE = this.num_frames / this.num_folds;

		// Folds
		for (int fold = start_fold; fold < this.num_folds + 1; fold++)
		{						
			System.out.println("Experiment FOLD=" + fold + " results_suffix=" + results_suffix);
			String FOLD_RESULTS_DIR_PATH = resultsdir + "_" + trial + "/" + fold;

			// Get light cones for training and testing
			// Account for light cone extra frames
			int start = (fold - 1) * FOLD_SIZE + this.h_p;
			int stop = Math.min(start + FOLD_SIZE, this.num_frames);
			lcSet = train_test_split(data, this.dataGeometry, start, stop, FOLD_RESULTS_DIR_PATH);
			
			// Process the fold
			this.do_fold(lcSet, fold, FOLD_RESULTS_DIR_PATH, number_lightcone_samples, cutoff);
			
			// Try to do garbage collection to free up heap space
			Runtime.getRuntime().gc();
		}
	}
}

