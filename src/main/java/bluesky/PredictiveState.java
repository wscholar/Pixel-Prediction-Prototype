package bluesky;

import utils.Utils;
import bluesky.enums.DE_Mode;

public class PredictiveState {
	public double[] mean_FLC;
	public double[] mean_PLC;
	public int total_points = 0;
	public int state_points = 0;
	public WeightedKDE FLC_DE;
	public WeightedKDE PLC_DE;
	
	public PredictiveState(double[][] PLCs, double[][] FLCs, int total_points, int num_pts_used, int num_of_samples, double bandwidth_adjustment_factor) {
		this.mean_FLC = Utils.mean(FLCs, 0);
		this.mean_PLC = Utils.mean(PLCs, 0);
		this.total_points = total_points;
		this.state_points = Utils.is_empty(PLCs) ? 0 : PLCs.length;
		this.FLC_DE = new WeightedKDE(FLCs, num_of_samples, num_pts_used, DE_Mode.FULL, false, bandwidth_adjustment_factor);
		this.PLC_DE = new WeightedKDE(PLCs, num_of_samples, num_pts_used, DE_Mode.FULL, false, bandwidth_adjustment_factor);
	}	
	
	public PredictiveState(double[][] PLCs, double[][] FLCs, int total_points, int num_pts_used, double bandwidth_adjustment_factor) {
		this.mean_FLC = Utils.mean(FLCs, 0);
		this.mean_PLC = Utils.mean(PLCs, 0);
		this.total_points = total_points;
		this.state_points = Utils.is_empty(PLCs) ? 0 : PLCs.length;
		this.FLC_DE = new WeightedKDE(FLCs, FLCs.length, num_pts_used, DE_Mode.FULL, false, bandwidth_adjustment_factor);
		this.PLC_DE = new WeightedKDE(PLCs, PLCs.length, num_pts_used, DE_Mode.FULL, false, bandwidth_adjustment_factor);
	}	
	
	public double distance_to_state(double[] point) {
		return Utils.calculateDistance(point, this.mean_PLC);
	}

	public double[] batch_PLC_densities(double[][] PLCs, String label) {
		return this.PLC_DE.pdf(PLCs, label);
	}
	
	public double[] batch_FLC_densities(double[][] FLCs, String label) {
		return this.FLC_DE.pdf(FLCs, label);
	}
	
	public double state_likelihood()
	{
		return this.state_points / this.total_points;
	}

	public void clear() {
		this.FLC_DE.clear();
		this.PLC_DE.clear();
	}

}
