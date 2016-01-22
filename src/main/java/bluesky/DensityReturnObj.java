package bluesky;

public class DensityReturnObj {
	public double result;
	public double[] kernel_evals;
	
	public DensityReturnObj(double res, double[] kes) {
		this.result = res;
		this.kernel_evals = kes;
	}
}
