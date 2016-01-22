package bluesky;

public interface Algorithm {

	public void learn(double[][] histories, double[][] futures);

	public double[][] predict_batch(double[][] chunk, String chunkname);

	double[] compute_likelihoods(double[][] PLCs, double[][] FLCs, String label);
	
	public void clear();
}
