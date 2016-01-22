package bluesky;

public class FLTP implements Algorithm {
	public int fold_number = 1;

	public FLTP(int fold_number) {
		this.fold_number = fold_number;
	}

	public void learn(double[][] histories, double[][] futures) {
		// TODO Auto-generated method stub

	}

	public double[][] predict_batch(double[][] chunk, String chunkname) {
		// TODO Auto-generated method stub
		return null;
	}

	public double[] compute_likelihoods(double[][] PLCs, double[][] FLCs, String label) {
		// TODO Auto-generated method stub
		return null;
	}
	
	
	public void clear() {
		
	}
}
