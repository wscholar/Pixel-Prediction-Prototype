package utils;

import java.lang.reflect.Array;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.lang3.ArrayUtils;

import bluesky.CoordTuple;

public class Utils {

	// TODO: Find faster alternatives to math.pow and our home grown dot product methods.

	public static int randInt(int min, int max) {
		// NOTE: Usually this should be a field rather than a method
		// variable so that it is not re-seeded every call.
		Random rand = new Random();

		// nextInt is normally exclusive of the top value,
		// so add 1 to make it inclusive
		int randomNum = rand.nextInt((max - min) + 1) + min;

		return randomNum;
	}

	public static int[] randPermutation(int num_of_items) {
		int[] out = new int[num_of_items];
		for (int i = 0; i < num_of_items; i++) {
			out[i] = i;
		}

		List<Integer> outL = new ArrayList<Integer>();
		for (int index = 0; index < out.length; index++) {
			outL.add(out[index]);
		}

		Collections.shuffle(outL);

		for (int i = 0; i < num_of_items; i++) {
			out[i] = (int) outL.get(i);
		}

		return out;
	}

	public static int[] get_random_permutation(int length) {
		int[] array = new int[length];
		for (int i = 0; i < array.length; i++)
			array[i] = i;

		for (int i = 0; i < length; i++) {
			Random r = new Random();
			int ran = i + r.nextInt(length - i);
			int temp = array[i];
			array[i] = array[ran];
			array[ran] = temp;
		}
		return array;
	}

	public static double[] ravel(double[][] arr) {
		if (is_empty(arr))
			return new double[] {};

		double[] flat = new double[arr.length * arr[0].length];
		int counter = 0;
		for (int i = 0; i < arr.length; i++)
		{
			for (int j = 0; j < arr[i].length; j++)
			{
				flat[counter] = arr[i][j];
				counter++;
			}

		}
		return flat;
	}

	public static double[] ravel(double[][][] arr) {

		if (is_empty(arr))
			return new double[] {};

		double[] flat = new double[arr.length * arr[0].length * arr[0][0].length];
		int counter = 0;
		for (int i = 0; i < arr.length; i++)
		{
			for (int j = 0; j < arr[i].length; j++)

			{
				for (int k = 0; k < arr[i][j].length; k++)
				{
					flat[counter] = arr[i][j][k];
					counter++;
				}
			}
		}
		return flat;
	}

	public static double[] ravel(double[][][][] arr) {

		if (is_empty(arr))
			return new double[] {};

		double[] flat = new double[arr.length * arr[0].length * arr[0][0].length * arr[0][0][0].length];
		int counter = 0;
		for (int i = 0; i < arr.length; i++)
		{
			for (int j = 0; j < arr[i].length; j++)
			{
				for (int k = 0; k < arr[i][j].length; k++)
				{
					for (int l = 0; l < arr[i][j][k].length; l++)
					{
						flat[counter] = arr[i][j][k][l];
						counter++;
					}
				}
			}
		}
		return flat;
	}

	//WMS TODO check these
	public static double calculateDistance(double[][] array1, double[][] array2)
	{
		if (is_empty(array1))
			return -1;

		if (is_empty(array2))
			return -1;

		if (is_empty(array1[0]))
			return -1;

		double Sum = 0.0;
		for (int i = 0; i < array1.length; i++) {
			for (int j = 0; j < array1[0].length; j++) {
				Sum = Sum + Math.pow((array1[i][j] - array2[i][j]), 2.0);
			}
		}
		return Math.sqrt(Sum);
	}

	public static double calculateDistance(double[] array1, double[] array2)
	{
		if (is_empty(array1))
			return -1;

		if (is_empty(array2))
			return -1;

		double Sum = 0.0;
		for (int i = 0; i < array1.length; i++) {
			Sum = Sum + Math.pow((array1[i] - array2[i]), 2.0);
		}
		return Math.sqrt(Sum);
	}

	public static int number_of_dimensions(Object array) {
		Object tmp = array;
		int nDimensions = 0;
		while (true) {
			if (array instanceof Object[]) {
				tmp = ((Object[]) array)[0];
			}
			else if (tmp.getClass().isArray()) {
				return nDimensions + 1;
			}
			nDimensions++;
		}
	}

	public static double[] deep_copy(double[] arr) {
		return arr.clone();
	}

	public static int[] deep_copy(int[] arr) {
		return arr.clone();
	}

	public static double[][] deep_copy(double[][] arr) {
		int N = arr.length;
		double[][] out = new double[N][];
		for (int i = 0; i < N; i++) {
			out[i] = deep_copy(arr[i]);
		}
		return out;
	}

	public static int[][] deep_copy(int[][] arr) {
		int N = arr.length;
		int[][] out = new int[N][];
		for (int i = 0; i < N; i++) {
			out[i] = deep_copy(arr[i]);
		}
		return out;
	}

	public static double[][][] deep_copy(double[][][] arr) {
		int N = arr.length;
		double[][][] out = new double[N][][];
		for (int i = 0; i < N; i++) {
			out[i] = deep_copy(arr[i]);
		}
		return out;
	}

	public static int[][][] deep_copy(int[][][] arr) {
		int N = arr.length;
		int[][][] out = new int[N][][];
		for (int i = 0; i < N; i++) {
			out[i] = deep_copy(arr[i]);
		}
		return out;
	}

	public static double[][][][] deep_copy(double[][][][] arr) {
		int N = arr.length;
		double[][][][] out = new double[N][][][];
		for (int i = 0; i < N; i++) {
			out[i] = deep_copy(arr[i]);
		}
		return out;
	}

	public static int[][][][] deep_copy(int[][][][] arr) {
		int N = arr.length;
		int[][][][] out = new int[N][][][];
		for (int i = 0; i < N; i++) {
			out[i] = deep_copy(arr[i]);
		}
		return out;
	}

	public static double[][][][][] deep_copy(double[][][][][] arr) {
		int N = arr.length;
		double[][][][][] out = new double[N][][][][];
		for (int i = 0; i < N; i++) {
			out[i] = deep_copy(arr[i]);
		}
		return out;
	}

	public static int[][][][][] deep_copy(int[][][][][] arr) {
		int N = arr.length;
		int[][][][][] out = new int[N][][][][];
		for (int i = 0; i < N; i++) {
			out[i] = deep_copy(arr[i]);
		}
		return out;
	}

	public static CoordTuple[] deep_copy(CoordTuple[] arr) {
		int N = arr.length;
		CoordTuple[] out = new CoordTuple[N];
		for (int i = 0; i < N; i++) {
			out[i] = arr[i].copy();
		}
		return out;
	}

	@SuppressWarnings("unchecked")
	public static <T> T[] deepCopyOf(T[] array) {

		if (array == null || 0 >= array.length)
			return array;

		return (T[]) deepCopyOf(
				array,
				Array.newInstance(array[0].getClass(), array.length),
				0);
	}

	private static Object deepCopyOf(Object array, Object copiedArray, int index) {

		if (index >= Array.getLength(array))
			return copiedArray;

		Object element = Array.get(array, index);

		if (element.getClass().isArray()) {

			Array.set(copiedArray, index, deepCopyOf(
					element,
					Array.newInstance(
							element.getClass().getComponentType(),
							Array.getLength(element)),
					0));

		} else {

			Array.set(copiedArray, index, element);
		}

		return deepCopyOf(array, copiedArray, ++index);
	}

	public static int getDimensionCount(Object array) {
		int count = 0;

		if ((array == null))
			return -1;

		@SuppressWarnings("rawtypes")
		Class arrayClass = array.getClass();
		while (arrayClass.isArray()) {
			count++;
			arrayClass = arrayClass.getComponentType();
		}
		return count;
	}

	public static double[] get_column_slice(double[][] arr, int col) {
		if (col < 0)
			return new double[] {};

		if (is_empty(arr))
			return new double[] {};

		double[] out = new double[arr.length];
		for (int i = 0; i < arr.length; i++) {
			out[i] = arr[i][col];
		}
		return out;
	}

	public static double[][] normalize_rows(double[][] arr) {
		if (is_empty(arr))
			return new double[][] {};

		double[][] out = Utils.deep_copy(arr);

		if (is_empty(out))
			return null;

		for (int i = 0; i < arr.length; i++) {
			out[i] = Utils.normalize(arr[i]);
		}
		return out;
	}

	public static double[] normalize(double[] arr) {

		if (is_empty(arr))
			return new double[] {};

		int N = arr.length;
		double[] out = new double[N];
		double sum = Utils.sum(arr);

		if (sum < 1e-200) {
			// Adding small fudge amount helps with case where all 
			// are close to 0, causing numerical instability.
			double fudge = 1e-30;
			for (int i = 0; i < N; i++) {
				arr[i] += fudge;
			}
			sum = Utils.sum(arr);
		}

		for (int i = 0; i < N; i++) {
			out[i] = arr[i] / sum;
		}
		return out;
	}

	public static double[] weighted_mean(double[] weights, double[][] arr) {

		if (is_empty(weights))
			return new double[] {};

		if (is_empty(arr))
			return new double[] {};

		int d = arr[0].length;
		double[] normweights;
		if (Math.abs(Utils.sum(weights) - 1.0) > .01) {
			normweights = Utils.normalize(weights);
		} else {
			normweights = weights;
		}
		double[] out_mean = new double[d];
		for (int i = 0; i < normweights.length; i++)
			for (int j = 0; j < d; j++)
				out_mean[j] += (normweights[i] * arr[i][j]);
		return out_mean;
	}

	public static double[] mean(double[][] arr, int axis) {
		if (is_empty(arr))
			return new double[] {};

		if (is_empty(arr[0]))
			return new double[] {};

		int N = arr.length;
		int d = arr[0].length;
		if (axis == 1) {
			double[] out_mean = new double[N];
			for (int i = 0; i < N; i++)
				for (int j = 0; j < d; j++)
					out_mean[i] += arr[i][j];
			for (int i = 0; i < N; i++)
				out_mean[i] /= (double) d;
			return out_mean;
		} else {
			double[] out_mean = new double[d];
			for (int j = 0; j < d; j++)
				for (int i = 0; i < N; i++)
					out_mean[j] += arr[i][j];
			for (int j = 0; j < d; j++)
				out_mean[j] /= (double) N;
			return out_mean;
		}
	}

	public static double[] standard_deviation(double[][] arr) {

		if (is_empty(arr))
			return new double[] {};

		if (is_empty(arr[0]))
			return new double[] {};

		int N = arr.length;
		int d = arr[0].length;
		double[] mean = Utils.mean(arr, 0);
		double[] out_std = new double[d];
		for (int j = 0; j < d; j++) {
			for (int i = 0; i < N; i++)
				out_std[j] += Math.pow((arr[i][j] - mean[j]), 2);
			out_std[j] = Math.sqrt((double) out_std[j] / (N - 1));
		}
		return out_std;
	}

	public static double[][] weighted_covariance(double[] weights, double[][] arr) {

		if (is_empty(arr))
			return new double[][] {};

		if (is_empty(arr[0]))
			return new double[][] {};

		if (is_empty(weights))
			return new double[][] {};

		int d = arr[0].length;
		double sum_weights = 0.0;
		double sum_sq_weights = 0.0;
		double[] normalized_weights = normalize(weights);
		double[][] cov = new double[d][d];

		for (double w : normalized_weights) {
			sum_weights += w;
			sum_sq_weights += Math.pow(w, 2);
		}

		double normalizing_const = (double) sum_weights / (Math.pow(sum_weights, 2) - sum_sq_weights);
		double[][] demeaned_arr = weighted_demean(arr, normalized_weights);

		// Compute weighted covariance matrix
		for (int j = 0; j < d; j++) {
			for (int k = 0; k < d; k++) {
				double total = 0.0;
				for (int i = 0; i < arr.length; i++) {
					total += normalized_weights[i] * demeaned_arr[i][j] * demeaned_arr[i][k];
				}
				cov[j][k] = normalizing_const * total;
			}
		}

		return cov;
	}

	public static double[][] weighted_demean(double[][] arr, double[] weights) {

		if (is_empty(arr))
			return new double[][] {};

		if (is_empty(arr[0]))
			return new double[][] {};

		if (is_empty(weights))
			return new double[][] {};

		int n = arr.length;
		int d = arr[0].length;
		double[] mean = Utils.weighted_mean(weights, arr);
		double[][] out = new double[n][d];

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < d; j++) {
				out[i][j] = arr[i][j] - mean[j];
			}
		}

		return out;
	}

	public static double[][] demean(double[][] arr) {

		if (is_empty(arr))
			return new double[][] {};

		if (is_empty(arr[0]))
			return new double[][] {};

		int n = arr.length;
		int d = arr[0].length;
		double[] mean = Utils.mean(arr, 0);
		double[][] out = new double[n][d];

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < d; j++) {
				out[i][j] = arr[i][j] - mean[j];
			}
		}

		return out;
	}


	public static boolean is_empty(int[] arr) {
		return (arr == null || arr.length == 0);
	}

	public static boolean is_empty(double[] arr) {
		return (arr == null || arr.length == 0);
	}

	public static boolean is_empty(double[][] arr) {
		return (arr == null || arr.length == 0);
	}

	public static boolean is_empty(double[][][] arr) {
		return (arr == null || arr.length == 0);
	}

	public static boolean is_empty(double[][][][] arr) {
		return (arr == null || arr.length == 0);
	}

	public static boolean is_empty(CoordTuple[] arr) {
		return (arr == null || arr.length == 0);
	}

	public static double[] vstack(double[] arr_1, double[] arr_2) {
		if (Utils.is_empty(arr_1) && !Utils.is_empty(arr_2)) {
			return arr_2.clone();
		}
		if (!Utils.is_empty(arr_1) && Utils.is_empty(arr_2)) {
			return arr_1.clone();
		}
		if (Utils.is_empty(arr_1) && Utils.is_empty(arr_2)) {
			return null;
		}

		double[] out = new double[arr_1.length + arr_2.length];
		int index = 0;
		for (; index < arr_1.length; index++) {
			out[index] = arr_1[index];
		}
		for (; index < out.length; index++) {
			out[index] = arr_2[index - arr_1.length];
		}
		return out;
	}

	public static double[][] vstack(double[][] arr_1, double[][] arr_2) {
		if (Utils.is_empty(arr_1) && !Utils.is_empty(arr_2)) {
			return Utils.deep_copy(arr_2);
		}
		if (!Utils.is_empty(arr_1) && Utils.is_empty(arr_2)) {
			return Utils.deep_copy(arr_1);
		}
		if (Utils.is_empty(arr_1) && Utils.is_empty(arr_2)) {
			return null;
		}

		int d = arr_1[0].length;
		double[][] out = new double[arr_1.length + arr_2.length][d];
		int index = 0;
		for (; index < arr_1.length; index++) {
			for (int j = 0; j < d; j++) {
				out[index][j] = arr_1[index][j];
			}
		}
		for (; index < out.length; index++) {
			for (int j = 0; j < d; j++) {
				out[index][j] = arr_2[index - arr_1.length][j];
			}
		}
		return out;
	}

	public static CoordTuple[] vstack(CoordTuple[] arr_1, CoordTuple[] arr_2) {
		if (Utils.is_empty(arr_1) && !Utils.is_empty(arr_2)) {
			return Utils.deep_copy(arr_2);
		}
		if (!Utils.is_empty(arr_1) && Utils.is_empty(arr_2)) {
			return Utils.deep_copy(arr_1);
		}
		if (Utils.is_empty(arr_1) && Utils.is_empty(arr_2)) {
			return new CoordTuple[] {};
		}
		CoordTuple[] out = new CoordTuple[arr_1.length + arr_2.length];
		for (int i = 0; i < arr_1.length; i++) {
			out[i] = arr_1[i];
		}
		for (int j = 0; j < arr_2.length; j++) {
			out[arr_1.length + j] = arr_2[j];
		}

		return out;
	}

	public static double[] hstack(double[] arr_1, double[] arr_2) {
		return Utils.vstack(arr_1, arr_2);
	}

	public static double[][] hstack(double[][] arr_1, double[][] arr_2) {
		if (Utils.is_empty(arr_1) && !Utils.is_empty(arr_2)) {
			return Utils.deep_copy(arr_2);
		}
		if (!Utils.is_empty(arr_1) && Utils.is_empty(arr_2)) {
			return Utils.deep_copy(arr_1);
		}
		if (Utils.is_empty(arr_1) && Utils.is_empty(arr_2)) {
			return null;
		}

		assert (arr_1.length == arr_2.length);

		double[][] out = new double[arr_1.length][arr_1[0].length + arr_2[0].length];
		for (int index = 0; index < arr_1.length; index++) {
			for (int j = 0; j < arr_1[index].length; j++) {
				out[index][j] = arr_1[index][j];
			}
			for (int j = 0; j < arr_2[index].length; j++) {
				out[index][arr_1[index].length + j] = arr_2[index][j];
			}
		}
		return out;
	}

	public static int[] find_matching_rows(double[] arr, double val) {
		if (Utils.is_empty(arr)) {
			return new int[] {};
		}
		int[] out = new int[] {};
		for (int i = 0; i < arr.length; i++) {
			if (Double.compare(arr[i], val) == 0)
				out = ArrayUtils.add(out, i);
		}
		return out;
	}

	public static int[] find_matching_rows(int[] arr, int val) {
		if (Utils.is_empty(arr)) {
			return new int[] {};
		}
		int[] out = new int[] {};
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == val)
				out = ArrayUtils.add(out, i);
		}
		return out;
	}

	public static int[] matching_argmax_indices(double[][] arr, int j) {
		// This methods takes in a matrix and returns the indices
		// where column j has the largest value in that row.
		if (Utils.is_empty(arr)) {
			return new int[] {};
		}
		int[] out = new int[] {};
		int d = arr[0].length;
		for (int i = 0; i < arr.length; i++) {
			boolean is_max = true;
			for (int k = 0; k < d; k++) {
				is_max = is_max && (arr[i][k] <= arr[i][j]);
			}
			if (is_max)
				out = ArrayUtils.add(out, i);
		}
		return out;
	}

	public static int[] rowwise_argmax(double[][] arr) {
		// This methods takes in a row-normalized matrix and returns 
		// the column indices of the largest value in each row.
		if (Utils.is_empty(arr)) {
			return new int[] {};
		}
		int[] out = new int[arr.length];
		int d = arr[0].length;
		for (int i = 0; i < arr.length; i++) {
			int max_index = 0;
			for (int k = 0; k < d; k++) {
				if (arr[i][k] > arr[i][max_index])
					max_index = k;
			}
			out[i] = max_index;
		}
		return out;
	}

	public static double distributional_distance(double[] A, double[] B) {
		assert (A != null);
		assert (B != null);
		assert (A.length == B.length);
		int N = A.length;
		double[] differences = new double[N];
		for (int i = 0; i < N; i++) {
			differences[i] = Math.abs(A[i] - B[i]);
		}
		return Utils.median(differences);
	}

	public static double median(double[] arr) {
		if (Utils.is_empty(arr))
			throw new NullPointerException("arr in media method is null or empty");

		int N = arr.length;
		if (N == 1)
			return arr[0];
		Arrays.sort(arr);
		double med;
		if (arr.length % 2 == 0)
			med = ((double) arr[N / 2] + (double) arr[N / 2 - 1]) / 2.0;
		else
			med = (double) arr[N / 2];
		return med;
	}

	public static double[][] getAbsForArrays(double[][] array1, double[][] array2)
	{
		if (is_empty(array1))
			return new double[][] {};

		if (is_empty(array1[0]))
			return new double[][] {};

		if (is_empty(array2))
			return new double[][] {};

		if (is_empty(array2[0]))
			return new double[][] {};

		double[][] absArray = new double[array1.length][array1[0].length];
		for (int i = 0; i < array1.length; i++) {
			for (int k = 0; k < array1[i].length; k++) {
				absArray[i][k] = Math.abs(array1[i][k] - array2[i][k]);
			}
		}
		return absArray;
	}

	public static double[][] select_rows(int[] indices, double[][] arr) {
		if (Utils.is_empty(arr) || Utils.is_empty(indices)) {
			return new double[][] {};
		}
		int d = arr[0].length;
		double[][] out = new double[indices.length][d];
		for (int i = 0; i < indices.length; i++) {
			for (int j = 0; j < d; j++)
				out[i][j] = arr[indices[i]][j];
		}
		return out;
	}

	public static double[] reshape1D(double[] raveled_data, int[] data_geometry) {
		return reshape_recursive(raveled_data, data_geometry);
	}

	public static double[][] reshape2D(double[] raveled_data, int[] data_geometry) {
		return reshape_recursive(raveled_data, data_geometry);
	}

	public static double[][][] reshape3D(double[] raveled_data, int[] data_geometry) {
		return reshape_recursive(raveled_data, data_geometry);
	}

	public static double[][][][] reshape4D(double[] raveled_data, int[] data_geometry) {
		return reshape_recursive(raveled_data, data_geometry);
	}

	public static double[][][][][] reshape5D(double[] raveled_data, int[] data_geometry) {
		return reshape_recursive(raveled_data, data_geometry);
	}

	public static Object reshape(double[] raveled_data, int[] data_geometry) {
		// Make sure the number of elements in array remains unchanged
		assert (raveled_data.length == Utils.product(data_geometry));
		if (Utils.is_empty(raveled_data) || Utils.is_empty(data_geometry)) {
			switch (data_geometry.length) {
				case 1:
					return new double[] {};
				case 2:
					return new double[][] {};
				case 3:
					return new double[][][] {};
				case 4:
					return new double[][][][] {};
				case 5:
					return new double[][][][][] {};
				default:
					return null;
			}
		} else {
			switch (data_geometry.length) {
				case 1:
					double[] result1D = reshape_recursive(raveled_data, data_geometry);
					return result1D;
				case 2:
					double[][] result2D = reshape_recursive(raveled_data, data_geometry);
					return result2D;
				case 3:
					double[][][] result3D = reshape_recursive(raveled_data, data_geometry);
					return result3D;
				case 4:
					double[][][][] result4D = reshape_recursive(raveled_data, data_geometry);
					return result4D;
				case 5:
					double[][][][][] result5D = reshape_recursive(raveled_data, data_geometry);
					return result5D;
				default:
					return null;
			}
		}
	}

	@SuppressWarnings("unchecked")
	public static <T> T reshape_recursive(double[] data, int[] data_geometry) {
		assert (!Utils.is_empty(data_geometry));

		int num_rows = data_geometry[0];

		if (data_geometry.length == 1) {
			return (T) data;
		} else {
			int[] reduced_geometry = Utils.slice(data_geometry, 1, data_geometry.length);
			int chunk_length = product(reduced_geometry);

			switch (reduced_geometry.length) {
				case 1:
					double[][] results2D = new double[num_rows][];
					for (int row = 0; row < num_rows; row++) {
						double[] plane_chunk = slice(data, row * chunk_length, (row + 1) * chunk_length);
						results2D[row] = plane_chunk.clone();
					}
					return (T) results2D;
				case 2:
					double[][][] results3D = new double[num_rows][][];
					for (int row = 0; row < num_rows; row++) {
						double[] plane_chunk = slice(data, row * chunk_length, (row + 1) * chunk_length);
						results3D[row] = deep_copy((double[][]) reshape_recursive(plane_chunk, reduced_geometry));
					}
					return (T) results3D;
				case 3:
					double[][][][] results4D = new double[num_rows][][][];
					for (int row = 0; row < num_rows; row++) {
						double[] plane_chunk = slice(data, row * chunk_length, (row + 1) * chunk_length);
						results4D[row] = deep_copy((double[][][]) reshape_recursive(plane_chunk, reduced_geometry));
					}
					return (T) results4D;
				case 4:
					double[][][][][] results5D = new double[num_rows][][][][];
					for (int row = 0; row < num_rows; row++) {
						double[] plane_chunk = slice(data, row * chunk_length, (row + 1) * chunk_length);
						results5D[row] = deep_copy((double[][][][]) reshape_recursive(plane_chunk, reduced_geometry));
					}
					return (T) results5D;
				default:
					return null;
			}
		}
	}

	public static double[] slice(double[] arr, int start, int end) {
		if (arr.length == 0)
			return arr;
		if (start == end)
			return new double[] {};
		end = Math.min(end, arr.length);
		int num_elements = end - start;
		double[] result = new double[num_elements];
		for (int i = 0; i < num_elements; i++) {
			result[i] = arr[start + i];
		}
		return result;
	}

	public static int[] slice(int[] arr, int start, int end) {
		if (arr.length == 0)
			return arr;
		end = Math.min(end, arr.length);
		int num_elements = end - start;
		int[] result = new int[num_elements];
		for (int i = 0; i < num_elements; i++) {
			result[i] = arr[start + i];
		}
		return result;
	}

	public static double[][] slice(double[][] arr, int start, int end) {
		if (arr.length == 0)
			return arr;
		end = Math.min(end, arr.length);
		int num_elements = end - start;
		double[][] result = new double[num_elements][];
		for (int i = 0; i < num_elements; i++) {
			result[i] = arr[start + i];
		}
		return result;
	}

	public static double[][] slice(double[][] arr, int start, int end, int axis) {
		if (axis < 0 || axis > 1) {
			System.out.print("ERROR: Trying to slice array along nonexistent axis.");
			System.exit(1);
		}
		int num_elements = 0;
		double[][] result;
		switch (axis) {
			case 0:
				if (arr.length == 0)
					return arr;
				end = Math.min(end, arr.length);
				num_elements = end - start;
				result = new double[num_elements][];
				for (int i = 0; i < num_elements; i++) {
					result[i] = arr[start + i];
				}
				return result;
			case 1:
				if (arr.length == 0)
					return arr;
				if (arr[0].length == 0)
					return arr;
				end = Math.min(end, arr[0].length);
				num_elements = end - start;
				result = new double[arr.length][num_elements];
				for (int i = 0; i < arr.length; i++) {
					for (int j = 0; j < num_elements; j++) {
						result[i][j] = arr[i][start + j];
					}
				}
				return result;
			default:
				return null;
		}
	}

	public static double[][][] slice(double[][][] arr, int start, int end, int axis) {
		if (axis < 0 || axis > 2) {
			System.out.print("ERROR: Trying to slice array along nonexistent axis.");
			System.exit(1);
		}
		int num_elements = 0;
		double[][][] result;
		switch (axis) {
			case 0:
				if (arr.length == 0)
					return arr;
				end = Math.min(end, arr.length);
				num_elements = end - start;
				result = new double[num_elements][][];
				for (int i = 0; i < num_elements; i++) {
					result[i] = arr[start + i];
				}
				return result;
			case 1:
				if (arr.length == 0)
					return arr;
				if (arr[0].length == 0)
					return arr;
				end = Math.min(end, arr[0].length);
				num_elements = end - start;
				result = new double[arr.length][num_elements][];
				for (int i = 0; i < arr.length; i++) {
					for (int j = 0; j < num_elements; j++) {
						result[i][j] = arr[i][start + j];
					}
				}
				return result;
			case 2:
				if (arr.length == 0)
					return arr;
				if (arr[0].length == 0)
					return arr;
				if (arr[0][0].length == 0)
					return arr;
				end = Math.min(end, arr[0][0].length);
				num_elements = end - start;
				result = new double[arr.length][arr[0].length][num_elements];
				for (int i = 0; i < arr.length; i++) {
					for (int j = 0; j < arr[0].length; j++) {
						for (int k = 0; k < num_elements; k++) {
							result[i][j][k] = arr[i][j][start + k];
						}
					}
				}
				return result;
			default:
				return null;
		}
	}

	public static double[][][][] slice(double[][][][] arr, int start, int end, int axis) {
		if (axis < 0 || axis > 3) {
			System.out.print("ERROR: Trying to slice array along nonexistent axis.");
			System.exit(1);
		}
		int num_elements = 0;
		double[][][][] result;
		switch (axis) {
			case 0:
				if (arr.length == 0)
					return arr;
				end = Math.min(end, arr.length);
				num_elements = end - start;
				result = new double[num_elements][][][];
				for (int i = 0; i < num_elements; i++) {
					result[i] = arr[start + i];
				}
				return result;
			case 1:
				if (arr.length == 0)
					return arr;
				if (arr[0].length == 0)
					return arr;
				end = Math.min(end, arr[0].length);
				num_elements = end - start;
				result = new double[arr.length][num_elements][][];
				for (int i = 0; i < arr.length; i++) {
					for (int j = 0; j < num_elements; j++) {
						result[i][j] = arr[i][start + j];
					}
				}
				return result;
			case 2:
				if (arr.length == 0)
					return arr;
				if (arr[0].length == 0)
					return arr;
				if (arr[0][0].length == 0)
					return arr;
				end = Math.min(end, arr[0][0].length);
				num_elements = end - start;
				result = new double[arr.length][arr[0].length][num_elements][];
				for (int i = 0; i < arr.length; i++) {
					for (int j = 0; j < arr[0].length; j++) {
						for (int k = 0; k < num_elements; k++) {
							result[i][j][k] = arr[i][j][start + k];
						}
					}
				}
				return result;
			case 3:
				if (arr.length == 0)
					return arr;
				if (arr[0].length == 0)
					return arr;
				if (arr[0][0].length == 0)
					return arr;
				if (arr[0][0][0].length == 0)
					return arr;
				end = Math.min(end, arr[0][0][0].length);
				num_elements = end - start;
				result = new double[arr.length][arr[0].length][arr[0][0].length][num_elements];
				for (int i = 0; i < arr.length; i++) {
					for (int j = 0; j < arr[0].length; j++) {
						for (int k = 0; k < arr[0][0].length; k++) {
							for (int l = 0; l < num_elements; l++) {
								result[i][j][k][l] = arr[i][j][k][start + l];
							}
						}
					}
				}
				return result;
			default:
				return null;
		}
	}

	public static int[][] slice(int[][] arr, int start, int end) {
		if (arr.length == 0)
			return arr;
		end = Math.min(end, arr.length);
		int num_elements = end - start;
		int[][] result = new int[num_elements][];
		for (int i = 0; i < num_elements; i++) {
			result[i] = arr[start + i];
		}
		return result;
	}

	public static double[][] delete_column(double[][] arr, int col) {
		assert (!Utils.is_empty(arr));
		int d = arr[0].length;
		double[][] result = new double[arr.length][d - 1];
		for (int i = 0; i < arr.length; i++) {
			int offset = 0;
			for (int j = 0; j < d; j++) {
				if (j != col) {
					result[i][offset] = arr[i][j];
					offset++;
				}

			}
		}
		return result;
	}

	public static double product(double[] arr) {
		double total = 1;
		for (double v : arr) {
			total *= v;
		}
		return total;
	}

	public static int product(int[] arr) {
		int total = 1;
		for (int v : arr) {
			total *= v;
		}
		return total;
	}

	public static double sum(double[] arr) {
		double total = 0.0;
		for (double v : arr) {
			total += v;
		}
		return total;
	}

	public static double[] axis_sum(double[][] arr, int axis) {
		if (Utils.is_empty(arr)) {
			return new double[] {};
		}
		int N = arr.length;
		int d = arr[0].length;
		double[] total = new double[axis == 1 ? N : d];
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < d; j++) {
				total[axis == 1 ? i : j] += arr[i][j];
			}
		}
		return total;
	}

	/**
	 * Returns the maximum of an array.
	 */
	public static double max(double[] v) {
		double max = v[0];
		for (int i = 1; i < v.length; i++) {
			if (max < v[i]) {
				max = v[i];
			}
		}
		return (max);
	}

	/**
	 * Returns the maximum of an array.
	 */
	public static double max(double[][] v) {
		double max = max(v[0]);
		for (int i = 1; i < v.length; i++) {
			if (max < max(v[i])) {
				max = max(v[i]);
			}
		}
		return (max);
	}

	/**
	 * Returns the minimum of an array.
	 */
	public static double min(double[][] v) {
		double min = min(v[0]);
		for (int i = 1; i < v.length; i++) {
			if (min > min(v[i])) {
				min = min(v[i]);
			}
		}
		return (min);
	}

	/**
	 * Returns the minimum of an array.
	 */
	public static double min(double[] v) {
		double min = v[0];
		for (int i = 1; i < v.length; i++) {
			if (min > v[i]) {
				min = v[i];
			}
		}
		return (min);
	}

	static public double[][] matrix_multiply(double[][] A, double[][] B) {
		assert (!Utils.is_empty(A));
		assert (!Utils.is_empty(B));
		//WMS changed this shoudl be not equal i believe they have to be square????
		assert (A[0].length != B.length) : "Matrix Multiply: Dimensions must be commensurate";
		int N = A.length;
		int J = A[0].length;
		int D = B[0].length;
		double[][] out = new double[N][D];
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < J; j++) {
				for (int d = 0; d < D; d++) {
					out[i][d] += A[i][j] * B[j][d];
				}
			}
		}
		return out;
	}

	static public double[][] elementwise_multiply(double[][] A, double[][] B) {
		assert (!Utils.is_empty(A));
		assert (!Utils.is_empty(B));
		int N = A.length;
		int K = A[0].length;
		double[][] out = new double[N][K];
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < K; j++) {
				out[i][j] = A[i][j] * B[i][j];
			}
		}
		return out;
	}

	static public double[][] transpose(double[][] A) {
		if (Utils.is_empty(A))
			return A;
		int N = A.length;
		int K = A[0].length;
		double[][] out = new double[K][N];
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < K; j++) {
				out[j][i] = A[i][j];
			}
		}
		return out;
	}

	static public double[][][] transpose(double[][][] A) {
		if (Utils.is_empty(A))
			return A;
		int N = A.length;
		int K = A[0].length;
		double[][][] out = new double[K][N][];
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < K; j++) {
				out[j][i] = A[i][j];
			}
		}
		return out;
	}

	static public double dot_product(double[] a, double[] b) {
		assert (a.length == b.length);
		double out = 0.0;
		for (int i = 0; i < a.length; i++) {
			out += a[i] * b[i];
		}
		return out;
	}

	// TODO: find faster optimized alternatives for this
	static public double[] dot_product(double[][] a, double[] b) {
		assert (!Utils.is_empty(a));
		assert (!Utils.is_empty(b));
		assert (a[0].length == b.length);
		double[] out = new double[a.length];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < b.length; j++) {
				out[i] += a[i][j] * b[j];
			}
		}
		return out;
	}

	static public double[] dot_product(double[] a, double[][] b) {
		return Utils.dot_product(b, a);
	}

	static public int cutoff(int[] data_geometry, int h_p, int c) {
		// NOTE: data geometry is always going to have the first element as the number of frames,
		// and the last element will be the number of pixel elements (such as 1 for grayscale,
		// 3 for RGB, etc). Thus, we need to ignore first and last element.		
		int margin = 2 * c * h_p;
		int[] margin_free_geometry = new int[data_geometry.length - 2];
		for (int i = 1; i < data_geometry.length - 1; i++) {
			margin_free_geometry[i - 1] = data_geometry[i] - margin;
		}
		return Utils.product(margin_free_geometry);
	}

	static public double sum_log(double[] arr) {
		double total = 0;
		for (double v : arr) {
			total += Math.log(v);
		}
		return total;
	}

	public static int min(int a, int b) {
		return (a < b) ? a : b;
	}

	public static double[] difference(double[] a, double[] b) {
		assert (a.length == b.length);
		double[] out = new double[a.length];
		for (int i = 0; i < a.length; i++) {
			out[i] = a[i] - b[i];
		}
		return out;
	}

	public static String ArrayToRowString(double[] arr, int items_per_row) {
		StringBuilder builder = new StringBuilder();
		for (int i = 0; i < arr.length; i += items_per_row) {
			StringBuilder row_string = new StringBuilder();
			for (int j = 0; j < items_per_row; j++) {
				if (i + j >= arr.length) {
					break;
				} else {
					row_string.append(arr[i + j]);
					row_string.append(" ");
				}
			}
			builder.append(String.format("%s%n", row_string.toString().trim()));
		}
		return builder.toString();
	}

	public static boolean data_is_constant(double[][] data) {
		double[] maxes = Utils.axis_max(data, 1);
		double[] mins = Utils.axis_min(data, 1);
		for (int j = 0; j < maxes.length; j++) {
			if (maxes[j] - mins[j] < 1e-50) {
				return true;
			}
		}
		return false;
	}

	public static double[] axis_max(double[][] data, int axis) {
		// NOTE: assumes each row has same number of columns (rectangular)		
		double[] out;
		if (axis == 0) {
			out = new double[data.length];
			for (int i = 0; i < out.length; i++) {
				out[i] = Utils.max(data[i]);
			}
		} else {
			out = new double[data[0].length];
			for (int j = 0; j < out.length; j++) {
				out[j] = Utils.max(Utils.get_column_slice(data, j));
			}
		}
		return out;
	}

	public static double[] axis_min(double[][] data, int axis) {
		// NOTE: assumes each row has same number of columns (rectangular)		
		double[] out;
		if (axis == 0) {
			out = new double[data.length];
			for (int i = 0; i < out.length; i++) {
				out[i] = Utils.min(data[i]);
			}
		} else {
			out = new double[data[0].length];
			for (int j = 0; j < out.length; j++) {
				out[j] = Utils.min(Utils.get_column_slice(data, j));
			}
		}
		return out;
	}

	public static double[][] add_random_noise(double[][] data) {
		double[][] out = Utils.deep_copy(data);
		for (int i = 0; i < out.length; i++) {
			for (int j = 0; j < out[i].length; j++) {
				out[i][j] += (Math.random() / 1e-20);
			}
		}
		return out;
	}

	public static double[][] get_unique(double[][] data) {
		Set<double[]> seen = new HashSet<double[]>();
		for (double[] d : data) {
			seen.add(d);
		}
		double[][] out = new double[seen.size()][];
		int index = 0;
		for (double[] s : seen) {
			out[index++] = s;
		}
		return out;
	}

	public static Set<String> get_redundant(double[][] data) {
		Set<String> seen = new HashSet<String>();
		Set<String> redundant = new HashSet<String>();
		for (double[] d : data) {
			String key = Utils.create_key(d);
			if (seen.contains(key)) {
				redundant.add(key);
			} else {
				seen.add(key);
			}
		}
		return redundant;
	}

	public static double[] mahalanobis_distance(double[] p, double[][] points, double[][] inv_cov) {
		double[] out = new double[points.length];
		for (int i = 0; i < points.length; i++) {
			double[] diff = Utils.difference(p, points[i]);
			double[][] left = Utils.matrix_multiply(new double[][] { diff }, inv_cov);
			double res = 0;
			for (int j = 0; j < diff.length; j++) {
				res += left[0][j] * diff[j];
			}
			out[i] = Math.sqrt(res);
		}
		return out;
	}

	public static String create_key(double[][] arr) {
		return Utils.create_key(Utils.ravel(arr));
	}

	public static String create_key(double[] arr) {
		StringBuilder sb = new StringBuilder();
		NumberFormat formatter = new DecimalFormat("#0.000000");
		for (int r = 0; r < arr.length; r++) {
			sb.append(formatter.format(arr[r]));
			sb.append(",");
		}
		return sb.toString();
	}
	
	public static double[] extend(double[] a, double[] b) {
		double[] out = new double[a.length + b.length];
		for (int i = 0; i < a.length; i++) {
			out[i] = a[i];
		}
		for (int j = 0; j < b.length; j++) {
			out[a.length + j] = b[j];
		}
		return out;
	}
	
	public static double[] add_pad_frame(double[] data, int[] data_geometry, int blank_frame_location) {
		int frame_length = Utils.product(Utils.slice(data_geometry, 1, data_geometry.length));
		int data_index = 0;
		
		double[] out = new double[data.length + frame_length];
		for (int t = 0; t < out.length; t++) {
			if (t / frame_length != blank_frame_location) {
				out[t] = data[data_index];
				data_index += 1;
			}
		}
		return out; 
	}
	
	public static double[][] add_pad_frame(double[][] data, int blank_frame_location) {
		int frame_length = data[0].length;
		int data_index = 0;
		
		double[][] out = new double[data.length + 1][frame_length];
		for (int t = 0; t < out.length; t++) {
			if (blank_frame_location != t) {
				out[t] = data[data_index];
				data_index += 1;
			}
		}
		
		return out; 
	}
}
