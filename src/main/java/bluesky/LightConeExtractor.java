package bluesky;

import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;

import utils.Scaler;
import utils.Utils;

public class LightConeExtractor {

	int c = 0;
	int h_p = 0;
	boolean scale_data;
	boolean double_ended;
	Scaler plc_scaler_obj;
	Scaler flc_scaler_obj;

	public LightConeExtractor(int speed, int h_p, boolean rescale, boolean double_ended_cones)
	{
		this.c = speed;
		this.h_p = h_p;
		this.scale_data = rescale;
		this.double_ended = double_ended_cones;
	}	

	private boolean at_least_one_light_cone(double[][] data) {
		int future_pad = double_ended ? this.h_p : 0;
		return (data.length >= 1 + this.h_p + future_pad);
	}
	private boolean at_least_one_light_cone(double[][][] data) {
		int future_pad = double_ended ? this.h_p : 0;
		return (data.length >= 1 + this.h_p + future_pad) && 
				(data[0].length >= 1 + 2 * this.h_p * c);
	}
	private boolean at_least_one_light_cone(double[][][][] data) {
		int future_pad = double_ended ? this.h_p : 0;
		return (data.length >= 1 + this.h_p + future_pad) &&
				(data[0].length >= 1 + 2 * this.h_p * c) &&
				(data[0][0].length >= 1 + 2 * this.h_p * c);
	}
	private boolean at_least_one_light_cone(double[][][][][] data) {
		int future_pad = double_ended ? this.h_p : 0;
		return (data.length >= 1 + this.h_p + future_pad) &&
				(data[0].length >= 1 + 2 * this.h_p * c) &&
				(data[0][0].length >= 1 + 2 * this.h_p * c) &&
				(data[0][0][0].length >= 1 + 2 * this.h_p * c);
	}
	
	private double[][] plc_scale(double[][] data) {
		if (this.scale_data) {
			if (this.plc_scaler_obj == null) {
				this.plc_scaler_obj = new Scaler();
			}
			return this.plc_scaler_obj.scale(data);
		} else {
			return data;
		}
	}
	
	private double[][] flc_scale(double[][] data) {
		if (this.scale_data) {
			if (this.flc_scaler_obj == null) {
				this.flc_scaler_obj = new Scaler();
			}
			return this.flc_scaler_obj.scale(data);
		} else {
			return data;
		}
	}
	
	public LightConesCollection extract(double[] raveled_data, int[] data_geometry, int number_of_samples) {
		assert(data_geometry.length >= 2);
		switch (data_geometry.length) {
			case 2: // Time series, multivariate time series
				return this._extract((double[][]) Utils.reshape(raveled_data, data_geometry), number_of_samples);
			case 3: // Cellular automata type data (grayscale or RGB)
				return this._extract((double[][][]) Utils.reshape(raveled_data, data_geometry), number_of_samples);
			case 4: // Video data (grayscale or RGB)
				return this._extract((double[][][][]) Utils.reshape(raveled_data, data_geometry), number_of_samples);
			case 5: // Swarm data, FMRI with color
				return this._extract((double[][][][][]) Utils.reshape(raveled_data, data_geometry), number_of_samples);
			default:
				System.out.print("ERROR: Does not currently support data with dimensions given by data_geometry. (i.e., with " + data_geometry.length + " entries.");
				System.exit(1);
				return null;
		}
	}
	
	//	Each extract method goes through data and for each point extracts 
	//  past light cone and future light cone point, 
	//  adding them to two lists. NOTE: each light cone
	//  is flattened to a one-dimensional array.
	//  input: data (numpy matrix or array)
	//  Assumes time dimension is always first dimension.
	
	private int calculate_num_plc_elements(int[] ns, int spatial_data_dim, int pixel_dim) {
		int total = 0;
		for (int i = 1; i < ns.length; i++) {
			total += Math.pow(ns[i], spatial_data_dim - 1);
		}
		return total * pixel_dim;
	}
	
	// Time series data. Could be multivariate or univariate.
	private LightConesCollection _extract(double[][] data, int number_of_samples)
	{
		// Check if too small to extract at least one light cone
		if (!at_least_one_light_cone(data)) 
			return new LightConesCollection(new double[][]{}, new double[][]{}, new CoordTuple[]{});
		
		int timesteps = data.length;
		int pixel_dim = data[0].length;
		int future_pad = this.double_ended ? this.h_p : 0;
		int num_of_lightcones = (timesteps - this.h_p - future_pad);
		
		if (number_of_samples == -1) {
			number_of_samples = num_of_lightcones;
		}
		int num_retained = Math.min(num_of_lightcones, number_of_samples);
		int[] retained_indices = Utils.slice(Utils.get_random_permutation(num_of_lightcones), 0, num_retained);
		Arrays.sort(retained_indices);
		CoordTuple[] coordinates = new CoordTuple[num_retained];
		double[][] PLCs = new double[num_retained][];
		double[][] FLCs = new double[num_retained][pixel_dim];
		int lc_count = 0;
		int lc_index = 0;

		for (int t = this.h_p; t < timesteps - future_pad; t++)
		{
			lc_index++;
			
			if (lc_count >= num_retained || retained_indices[lc_count] != lc_index-1) 
				continue;
				
			// ''' Get FLC point '''
			FLCs[lc_count] = data[t];
				
			// Get rest of past pyramid
			double[] plc = Utils.ravel(Utils.slice(data, t - this.h_p, t, 0));
			ArrayUtils.reverse(plc);

			// if double-ended, construct forward PLC
			if (this.double_ended) {
				double[] forward_plc = Utils.ravel(Utils.slice(data, t + this.h_p, t, 0));					
				ArrayUtils.reverse(forward_plc);
				plc = Utils.extend(plc, forward_plc);
			}
			
			// Save PLC
			PLCs[lc_count] = plc;
			
			// Save coordinates of FLC
			coordinates[lc_count] = new CoordTuple(new int[]{t});
			
			// Update light cones count
			lc_count++;
		}
		
		return new LightConesCollection(this.plc_scale(PLCs), this.flc_scale(FLCs), coordinates);
	}
	
	// Two dimensional cellular automata type data
	private LightConesCollection _extract(double[][][] data, int number_of_samples)
	{
		// Check if too small to extract at least one light cone
		if (!at_least_one_light_cone(data)) 
			return new LightConesCollection(new double[][]{}, new double[][]{}, new CoordTuple[]{});

		int timesteps = data.length;
		int shim_width = this.c * this.h_p;	
		int pixel_dim = data[0][0].length;
		int future_pad = this.double_ended ? this.h_p : 0;
		int num_of_lightcones = (timesteps - this.h_p - future_pad) * (data[0].length - 2 * shim_width);

		if (number_of_samples == -1) {
			number_of_samples = num_of_lightcones;
		}
		int num_retained = Math.min(num_of_lightcones, number_of_samples);
		int[] retained_indices = Utils.slice(Utils.get_random_permutation(num_of_lightcones), 0, num_retained);
		Arrays.sort(retained_indices);
		CoordTuple[] coordinates = new CoordTuple[num_retained];
		double[][] PLCs = new double[num_retained][];
		double[][] FLCs = new double[num_retained][pixel_dim];
		int[] ns = {1};
		int lc_count = 0;
		int lc_index = 0;
		
		for (int i = 0; i < this.h_p; i++)
		{
			//as we move back the triangle how many items are within the windows of the pyramid
			ns = ArrayUtils.add(ns, 2 * this.c + ns[ns.length - 1]);
		}

		for (int t = this.h_p; t < timesteps - future_pad; t++)
		{
			for (int i = shim_width; i < data[0].length - shim_width; i++)
			{
				lc_index++;
				
				if (lc_count >= num_retained || retained_indices[lc_count] != lc_index-1) 
					continue;
				
				// ''' Get FLC point '''
				FLCs[lc_count] = data[t][i];
				
				// Get rest of past pyramid
				int items_per_cone = this.calculate_num_plc_elements(ns, 2, pixel_dim);
				double[] plc = new double[items_per_cone];
				int plc_element_count = 0;
				for (int k = 1; k < this.h_p + 1; k++)
				{
					int r = ns[k] / 2;
					double[][] patch =  Utils.slice(data[t - k], i - r, i + r + 1, 0);
					for (double p : Utils.ravel(patch)) {
						plc[plc_element_count++] = p;
					}
				}

				ArrayUtils.reverse(plc);

				// if double-ended, construct forward PLC
				if (this.double_ended) {
					double[] forward_plc = new double[items_per_cone];
					plc_element_count = 0;
				
					for (int k = 1; k < this.h_p + 1; k++)
					{
						int r = ns[k] / 2;
						double[][] patch =  Utils.slice(data[t + k], i - r, i + r + 1, 0);
						for (double p : Utils.ravel(patch)) {
							forward_plc[plc_element_count++] = p;
						}
					}
					
					ArrayUtils.reverse(forward_plc);
					plc = Utils.extend(plc, forward_plc);
				}
				
				// Save PLC
				PLCs[lc_count] = plc;
				
				// Save coordinates of FLC
				coordinates[lc_count] = new CoordTuple(new int[]{t, i});
				
				// Update light cones count
				lc_count++;
			}
		}

		return new LightConesCollection(this.plc_scale(PLCs), this.flc_scale(FLCs), coordinates);
	}
	
	// RGB video Data
	private LightConesCollection _extract(double[][][][] data, int number_of_samples)
	{
		// Check if too small to extract at least one light cone
		if (!at_least_one_light_cone(data)) 
			return new LightConesCollection(new double[][]{}, new double[][]{}, new CoordTuple[]{});

		int timesteps = data.length;
		int shim_width = this.c * this.h_p;
		int pixel_dim = data[0][0][0].length;
		int future_pad = this.double_ended ? this.h_p : 0;
		int num_of_lightcones = (timesteps - this.h_p - future_pad) * (data[0].length - 2 * shim_width) * (data[0][0].length - 2 * shim_width);

		if (number_of_samples == -1) {
			number_of_samples = num_of_lightcones;
		}
		int num_retained = Math.min(num_of_lightcones, number_of_samples);
		int[] retained_indices = Utils.slice(Utils.get_random_permutation(num_of_lightcones), 0, num_retained);
		Arrays.sort(retained_indices);
		CoordTuple[] coordinates = new CoordTuple[num_retained];
		double[][] PLCs = new double[num_retained][];
		double[][] FLCs = new double[num_retained][];
		int[] ns = {1};
		int lc_count = 0;
		int lc_index = 0;
	
		for (int i = 0; i < this.h_p; i++)
		{
			//as we move back the triangle how many items are within the windows of the pyramid
			ns = ArrayUtils.add(ns, 2 * this.c + ns[ns.length - 1]);
		}
		
		for (int t = this.h_p; t < timesteps - future_pad; t++)
		{
			for (int i = shim_width; i < data[0].length - shim_width; i++)
			{
				for (int j = shim_width; j < data[0][0].length - shim_width; j++) 
				{
					lc_index++;
					
					if (lc_count >= num_retained || retained_indices[lc_count] != lc_index-1) 
						continue;
					
					// Get FLC point
					FLCs[lc_count] = data[t][i][j];
					
					// Get rest of past pyramid
					int items_per_cone = this.calculate_num_plc_elements(ns, 3, pixel_dim);
					double[] plc = new double[items_per_cone];
					int plc_element_count = 0;
					for (int k = 1; k < this.h_p + 1; k++)
					{
						int r = ns[k] / 2;
						double[][][] patch_level_1 = Utils.slice(data[t - k], i - r, i + r + 1, 0);
						double[][][] patch_level_2 = Utils.slice(patch_level_1, j - r, j + r + 1, 1);
						for (double p : Utils.ravel(patch_level_2)) {
							plc[plc_element_count++] = p;
						}
					}
					
					ArrayUtils.reverse(plc);
					
					// if double-ended, construct forward PLC
					if (this.double_ended) {
						double[] forward_plc = new double[items_per_cone];
						plc_element_count = 0;
					
						for (int k = 1; k < this.h_p + 1; k++)
						{
							int r = ns[k] / 2;
							double[][][] patch_level_1 = Utils.slice(data[t + k], i - r, i + r + 1, 0);
							double[][][] patch_level_2 = Utils.slice(patch_level_1, j - r, j + r + 1, 1);
							for (double p : Utils.ravel(patch_level_2)) {
								forward_plc[plc_element_count++] = p;
							}
						}
						
						ArrayUtils.reverse(forward_plc);
						plc = Utils.extend(plc, forward_plc);
					}
					
					// Save PLC
					PLCs[lc_count] = plc;
		
					// Save coordinates of FLC
					coordinates[lc_count] = new CoordTuple(new int[]{t, i, j});
					
					//Update count of light cones
					lc_count++;
				}
			}
		}
		
		return new LightConesCollection(this.plc_scale(PLCs), this.flc_scale(FLCs), coordinates);
	}
	
	// Five-dimensional color (x,y,z) or FMRI data-like.
	private LightConesCollection _extract(double[][][][][] data, int number_of_samples)
	{
		// Check if too small to extract at least one light cone
		if (!at_least_one_light_cone(data)) 
			return new LightConesCollection(new double[][]{}, new double[][]{}, new CoordTuple[]{});
			
			int timesteps = data.length;
			int shim_width = this.c * this.h_p;
			int pixel_dim = data[0][0][0][0].length;
			int future_pad = this.double_ended ? this.h_p : 0;
			int num_of_lightcones = (timesteps - this.h_p - future_pad) * 
									(data[0].length - 2 * shim_width) * 
									(data[0][0].length - 2 * shim_width) * 
									(data[0][0][0].length - 2 * shim_width);
			if (number_of_samples == -1) {
				number_of_samples = num_of_lightcones;
			}
			int num_retained = Math.min(num_of_lightcones, number_of_samples);
			int[] retained_indices = Utils.slice(Utils.get_random_permutation(num_of_lightcones), 0, num_retained);
			Arrays.sort(retained_indices);
			CoordTuple[] coordinates = new CoordTuple[num_retained];
			double[][] PLCs = new double[num_retained][];
			double[][] FLCs = new double[num_retained][pixel_dim];
			int[] ns = {1};
			int lc_count = 0;
			int lc_index = 0;
			
			for (int i = 0; i < this.h_p; i++)
			{
				//as we move back the triangle how many items are within the windows of the pyramid
				ns = ArrayUtils.add(ns, 2 * this.c + ns[ns.length - 1]);
			}

			for (int t = this.h_p; t < timesteps - future_pad; t++)
			{
				for (int i = shim_width; i < data[0].length - shim_width; i++)
				{
					for (int j = shim_width; j < data[0][0].length - shim_width; j++) 
					{
						for (int l = shim_width; l < data[0][0][0].length - shim_width; l++) 
						{
							lc_index++;
							
							if (lc_count >= num_retained || retained_indices[lc_count] != lc_index-1) 
								continue;
								
							// Get FLC point
							FLCs[lc_count] = data[t][i][j][l];
						
							// Get rest of past pyramid
							int items_per_cone = this.calculate_num_plc_elements(ns, 4, pixel_dim);
							double[] plc = new double[items_per_cone];
							int plc_element_count = 0;
							for (int k = 1; k < this.h_p + 1; k++)
							{
								int r = ns[k] / 2;
								double[][][][] patch_level_1 = Utils.slice(data[t - k], i - r, i + r + 1, 0);
								double[][][][] patch_level_2 = Utils.slice(patch_level_1, j - r, j + r + 1, 1);
								double[][][][] patch_level_3 = Utils.slice(patch_level_2, l - r, l + r + 1, 2);
								for (double p : Utils.ravel(patch_level_3)) {
									plc[plc_element_count++] = p;
								}
							}
						
							ArrayUtils.reverse(plc);
							
							// if double-ended, construct forward PLC
							if (this.double_ended) {
								double[] forward_plc = new double[items_per_cone];
								plc_element_count = 0;
							
								for (int k = 1; k < this.h_p + 1; k++)
								{
									int r = ns[k] / 2;
									double[][][][] patch_level_1 = Utils.slice(data[t + k], i - r, i + r + 1, 0);
									double[][][][] patch_level_2 = Utils.slice(patch_level_1, j - r, j + r + 1, 1);
									double[][][][] patch_level_3 = Utils.slice(patch_level_2, l - r, l + r + 1, 2);
									for (double p : Utils.ravel(patch_level_3)) {
										forward_plc[plc_element_count++] = p;
									}
								}
								
								ArrayUtils.reverse(forward_plc);
								plc = Utils.extend(plc, forward_plc);
							}
							
							// Save PLC
							PLCs[lc_count] = plc;
							
							// Save coordinates of FLC
							coordinates[lc_count] = new CoordTuple(new int[]{t, i, j, l});
							
							// Update light cone count
							lc_count++;
						}
					}
				}
			}
			
			return new LightConesCollection(this.plc_scale(PLCs), this.flc_scale(FLCs), coordinates);
		}
}