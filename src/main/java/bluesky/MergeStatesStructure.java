package bluesky;

public class MergeStatesStructure {
	public double distance;
	public int state_i;
	public int state_j;

	public MergeStatesStructure(double dist, int i, int j) {
		distance = dist;
		state_i = i;
		state_j = j;
	}
}