package bluesky;

import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;

import utils.Utils;

public class DataObj {

	private Object multiDimArray = null;
	private Object testData = null;
	private Object trainDataBefore = null;
	private Object trainDataAfter = null;
	private Class classOfObject = Double.class;

	public <T> DataObj(T[] multiDimArray) {

	}

	public <T> DataObj(T[] multiDimArray, Class classOfObject) {

		if (Utils.getDimensionCount(multiDimArray) == 1)
		{
			this.multiDimArray = ArrayUtils.clone(multiDimArray);
		}
		else if (Utils.getDimensionCount(this.multiDimArray) == 2)
		{
			this.multiDimArray = ArrayUtils.clone(multiDimArray);
		}
		else if (Utils.getDimensionCount(this.multiDimArray) == 3)
		{

		}
		else if (Utils.getDimensionCount(this.multiDimArray) == 4)
		{

		}
		//this.multiDimArray = Utils.deepCopyOf(multiDimArray);
	}

	public DataObj(double[] multiDimArray) {
		this.multiDimArray = ArrayUtils.clone(multiDimArray);
	}

	public DataObj(double[][] multiDimArray) {
		this.multiDimArray = Utils.deep_copy(multiDimArray);
	}

	public DataObj(double[][][] multiDimArray) {
		this.multiDimArray = Utils.deep_copy(multiDimArray);
	}

	public DataObj(double[][][][] multiDimArray) {
		this.multiDimArray = Utils.deep_copy(multiDimArray);
	}

	public DataObj(double[][][][][] multiDimArray) {
		this.multiDimArray = Utils.deep_copy(multiDimArray);
	}
	
	public int getDimensions()
	{
		return Utils.getDimensionCount(this.multiDimArray);
	}

	@SuppressWarnings("unchecked")
	public <T> double[] getTrainDataBefore(int test_start)
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 1)
		{
			trainDataBefore = (Arrays.copyOfRange((T[]) multiDimArray, 0, test_start));
			return (double[]) trainDataBefore;
		}
		return null;
	}

	@SuppressWarnings("unchecked")
	public <T> double[][] getTrainDataBefore2D(int test_start)
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 2)
		{
			//	trainDataBefore = Utils.deepCopyOf(Arrays.copyOfRange((T[]) multiDimArray, 0, test_start));
			return (double[][]) trainDataBefore;
		}
		return null;
	}

	@SuppressWarnings("unchecked")
	public <T> double[][][] getTrainDataBefore3D(int test_start)
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 3)
		{
			//	trainDataBefore = Utils.deepCopyOf(Arrays.copyOfRange((T[]) multiDimArray, 0, test_start));
			return (double[][][]) trainDataBefore;
		}
		return null;
	}

	@SuppressWarnings("unchecked")
	public <T> double[][][][] getTrainDataBefore4D(int test_start)
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 4)
		{
			//	trainDataBefore = Utils.deepCopyOf(Arrays.copyOfRange((T[]) multiDimArray, 0, test_start));
			return (double[][][][]) trainDataBefore;
		}
		return null;
	}
	
	@SuppressWarnings("unchecked")
	public <T> double[][][][][] getTrainDataBefore5D(int test_start)
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 5)
		{
			//	trainDataBefore = Utils.deepCopyOf(Arrays.copyOfRange((T[]) multiDimArray, 0, test_start));
			return (double[][][][][]) trainDataBefore;
		}
		return null;
	}

	@SuppressWarnings("unchecked")
	public <T> double[] getTrainDataAfter(int test_end)
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 1)
		{
			trainDataAfter = (Arrays.copyOfRange((T[]) multiDimArray, test_end, ArrayUtils.getLength(multiDimArray) - 1));
			return (double[]) trainDataAfter;
		}
		return null;
	}

	@SuppressWarnings("unchecked")
	public <T> double[][] getTrainDataAfter2D(int test_end)
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 2)
		{
			//	trainDataAfter = Utils.deepCopyOf(Arrays.copyOfRange((T[]) multiDimArray, test_end, ArrayUtils.getLength(multiDimArray) - 1));
			return (double[][]) trainDataAfter;
		}
		return null;
	}

	@SuppressWarnings("unchecked")
	public <T> double[][][] getTrainDataAfter3D(int test_end)
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 3)
		{
			//	trainDataAfter = Utils.deepCopyOf(Arrays.copyOfRange((T[]) multiDimArray, test_end, ArrayUtils.getLength(multiDimArray) - 1));
			return (double[][][]) trainDataAfter;
		}
		return null;
	}

	@SuppressWarnings("unchecked")
	public <T> double[][][][] getTrainDataAfter4D(int test_end)
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 4)
		{
			//	trainDataAfter = Utils.deepCopyOf(Arrays.copyOfRange((T[]) multiDimArray, test_end, ArrayUtils.getLength(multiDimArray) - 1));
			return (double[][][][]) trainDataAfter;
		}
		return null;
	}

	@SuppressWarnings("unchecked")
	public <T> double[][][][][] getTrainDataAfter5D(int test_end)
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 5)
		{
			//	trainDataAfter = Utils.deepCopyOf(Arrays.copyOfRange((T[]) multiDimArray, test_end, ArrayUtils.getLength(multiDimArray) - 1));
			return (double[][][][][]) trainDataAfter;
		}
		return null;
	}
	
	@SuppressWarnings("unchecked")
	public <T> double[] getTestData(int test_start, int test_end)
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 1)
		{
			testData = (Arrays.copyOfRange((T[]) multiDimArray, test_start - 1, test_end));
			return (double[]) testData;
		}
		return null;
	}

	@SuppressWarnings("unchecked")
	public <T> double[][] getTestData2D(int test_start, int test_end)
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 2)
		{
			//testData = Utils.deepCopyOf(Arrays.copyOfRange((T[]) multiDimArray, test_start - 1, test_end));
			return (double[][]) testData;
		}
		return null;
	}

	@SuppressWarnings("unchecked")
	public <T> double[][][] getTestData3D(int test_start, int test_end)
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 3)
		{
			//testData = Utils.deepCopyOf(Arrays.copyOfRange((T[]) multiDimArray, test_start - 1, test_end));
			return (double[][][]) testData;
		}
		return null;
	}

	@SuppressWarnings("unchecked")
	public <T> double[][][][] getTestData4D(int test_start, int test_end)
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 4)
		{
			//testData = Utils.deepCopyOf(Arrays.copyOfRange((T[]) multiDimArray, test_start - 1, test_end));
			return (double[][][][]) testData;
		}
		return null;
	}

	@SuppressWarnings("unchecked")
	public <T> double[][][][][] getTestData5D(int test_start, int test_end)
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 5)
		{
			//testData = Utils.deepCopyOf(Arrays.copyOfRange((T[]) multiDimArray, test_start - 1, test_end));
			return (double[][][][][]) testData;
		}
		return null;
	}

	
	public double[] getData()
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 1)
		{
			return (double[]) this.multiDimArray;
		}
		return null;
	}

	public double[][] getData2D()
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 2)
		{
			return (double[][]) this.multiDimArray;
		}
		return null;
	}

	public double[][][] getData3D()
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 3)
		{
			return (double[][][]) this.multiDimArray;
		}
		return null;
	}

	public double[][][][] getData4D()
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 4)
		{
			return (double[][][][]) this.multiDimArray;
		}
		return null;
	}
	
	public double[][][][][] getData5D()
	{
		if (Utils.getDimensionCount(this.multiDimArray) == 5)
		{
			return (double[][][][][]) this.multiDimArray;
		}
		return null;
	}

}
