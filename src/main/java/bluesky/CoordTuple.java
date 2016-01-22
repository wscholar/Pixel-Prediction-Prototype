package bluesky;

public class CoordTuple {
	public int[] coords;
	
	public CoordTuple(int[] coords) {
		this.coords = new int[coords.length];
		for (int i = 0; i < this.coords.length; i++) {
			this.coords[i] = coords[i];
		}
	}
	
	public CoordTuple copy() {
		return new CoordTuple(this.coords);
	}
	
	public boolean is_match(int[] coords) {
		boolean match = true;
		for (int i = 0; i < this.coords.length; i++) {
			match = match && this.coords[i] == coords[i];
		}
		return match;
	}
}
