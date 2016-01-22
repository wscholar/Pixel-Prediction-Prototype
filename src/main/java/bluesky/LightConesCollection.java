package bluesky;

import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;

import utils.Utils;

public class LightConesCollection {
	private double[][] _PLCs;
	private double[][] _FLCs;
	private CoordTuple[] _FLC_coordinates;

	public LightConesCollection(double[][] PLCs, double[][] FLCs, CoordTuple[] FLC_coordinates) {
		this._PLCs = Utils.deep_copy(PLCs);
		this._FLCs = Utils.deep_copy(FLCs);
		this._FLC_coordinates = Utils.deep_copy(FLC_coordinates);
	}

	public double[][] get_PLCs() {
		return this._PLCs;
	}

	public double[][] get_FLCs() {
		return this._FLCs;
	}
	
	public double[][] get_PLCs_range(int start, int end) {
		return Utils.slice(this._PLCs, start, end);
	}

	public double[][] get_FLCs_range(int start, int end) {
		return Utils.slice(this._FLCs, start, end);
	}
	
	public CoordTuple[] get_coordinates() {
		return this._FLC_coordinates;
	}
	
	public static LightConesCollection merge(LightConesCollection LCC_1, LightConesCollection LCC_2) {
		double[][] PLCs_1 = LCC_1.get_PLCs();
		double[][] PLCs_2 = LCC_2.get_PLCs();
		double[][] FLCs_1 = LCC_1.get_FLCs();
		double[][] FLCs_2 = LCC_2.get_FLCs();
		CoordTuple[] Coords_1 = LCC_1.get_coordinates();
		CoordTuple[] Coords_2 = LCC_2.get_coordinates();
		return new LightConesCollection(Utils.vstack(PLCs_1, PLCs_2), Utils.vstack(FLCs_1, FLCs_2), Utils.vstack(Coords_1, Coords_2));
	}
	
	public static LightConesCollection merge_by_coords(LightConesCollection LCC_1, LightConesCollection LCC_2) {
		double[][] PLCs_1 = LCC_1.get_PLCs();
		double[][] PLCs_2 = LCC_2.get_PLCs();
		double[][] FLCs_1 = LCC_1.get_FLCs();
		CoordTuple[] Coords_1 = LCC_1.get_coordinates();
		CoordTuple[] Coords_2 = LCC_2.get_coordinates();
		double[][] double_ended_PLCs = new double[][]{};
		double[][] matching_FLCs = new double[][]{};
		CoordTuple[] matching_coords = new CoordTuple[]{};
		// Match by coordinates
		for (int i = 0; i < Coords_1.length; i++) {
			for (int j = i; j < Coords_2.length; j++) {
				if (Arrays.equals(Coords_1[i].coords, Coords_2[j].coords)) {
					double_ended_PLCs = ArrayUtils.add(double_ended_PLCs, Utils.hstack(PLCs_1[i], PLCs_2[j]));
					matching_FLCs = ArrayUtils.add(matching_FLCs, FLCs_1[i]);
					matching_coords = ArrayUtils.add(matching_coords, Coords_1[i]);
					break;
				}
			}
		}
		return new LightConesCollection(double_ended_PLCs, matching_FLCs, matching_coords);
	}
}
