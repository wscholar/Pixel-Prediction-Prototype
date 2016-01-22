package bluesky;

import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import utils.Utils;
import weka.core.matrix.Matrix;

public class UtilsTest {

	double[][] arr2d;
	double[][] arr2db;
	double[][] arr2dbmultiply;
	double[][] arr2dnull = null;
	double[][][] arr3d;
	double[][][] arr3dnull = null;
	double[] arrdist;
	double[] arrdist2;
	double[][] arrdist2D;
	double[][] arrdist2D2;
	CoordTuple[] ct;
	CoordTuple[] ct2;
	int[] intarray;
	int[] intarray2;
	double[][] matrixmath;

	@Before
	public void setup() {

		matrixmath = new double[2][2];
		for (int i = 0; i < matrixmath.length; i++)
		{
			for (int j = 0; j < matrixmath[i].length; j++)
			{
				matrixmath[i][j] = Utils.randInt(1, 100);
			}
		}

		arr2d = new double[3][6];
		for (int i = 0; i < arr2d.length; i++)
		{

			for (int j = 0; j < arr2d[i].length; j++)
			{
				arr2d[i][j] = Utils.randInt(1, 100);
			}
		}

		arr2db = new double[3][6];
		for (int i = 0; i < arr2db.length; i++)
		{

			for (int j = 0; j < arr2db[i].length; j++)
			{
				arr2db[i][j] = Utils.randInt(1, 100);
			}
		}

		arr2dbmultiply = new double[6][6];
		for (int i = 0; i < arr2dbmultiply.length; i++)
		{

			for (int j = 0; j < arr2dbmultiply[i].length; j++)
			{
				arr2dbmultiply[i][j] = Utils.randInt(1, 100);
			}
		}

		arr3d = new double[2][3][4];
		for (int i = 0; i < arr3d.length; i++)
		{
			for (int j = 0; j < arr3d[i].length; j++)
			{
				for (int k = 0; k < arr3d[i][j].length; k++)
				{
					arr3d[i][j][k] = Utils.randInt(1, 100);
				}

			}
		}

		arrdist = new double[3];
		arrdist2 = new double[3];
		arrdist[0] = 2;
		arrdist[1] = 4;
		arrdist[2] = 6;
		arrdist2[0] = 5;
		arrdist2[1] = 7;
		arrdist2[2] = 9;

		arrdist2D = new double[3][4];
		arrdist2D2 = new double[3][4];
		for (int i = 0; i < arrdist2D.length; i++)
		{
			for (int j = 0; j < arrdist2D[i].length; j++)
			{
				arrdist2D[i][j] = Utils.randInt(1, 6);
			}
		}

		for (int i = 0; i < arrdist2D2.length; i++)
		{
			for (int j = 0; j < arrdist2D2[i].length; j++)
			{
				arrdist2D2[i][j] = Utils.randInt(1, 6);
			}
		}

		intarray = new int[4];
		intarray2 = new int[4];

		ct = new CoordTuple[4];
		ct2 = new CoordTuple[4];

		for (int z = 0; z < ct.length; z++)
		{
			for (int i = 0; i < intarray.length; i++)
			{
				intarray[i] = Utils.randInt(1, 100);
			}

			for (int i = 0; i < intarray2.length; i++)
			{
				intarray2[i] = Utils.randInt(1, 100);
			}

			ct[z] = new CoordTuple(intarray);
			ct2[z] = new CoordTuple(intarray2);
		}

	}

	@Test
	public void testSqRootMatrix()
	{
		System.out.println("matrixmath=" + Arrays.deepToString(matrixmath));
		Matrix newmatrix = Matrix.constructWithCopy(matrixmath);
		System.out.println("newmatrix=" + newmatrix.toString());
		Matrix squarematrix = newmatrix.sqrt();
		System.out.println("squarematrix=" + squarematrix.toString());
		System.out.println("squareArray=" + Arrays.deepToString(squarematrix.getArrayCopy()));

	}

	@Test
	public void testMatrixInversion()
	{
		System.out.println("matrixmath=" + Arrays.deepToString(matrixmath));
		Matrix newmatrix = Matrix.constructWithCopy(matrixmath);
		System.out.println("newmatrix=" + newmatrix.toString());
		Matrix inverseMatrix = newmatrix.inverse();
		System.out.println("inverseMatrix=" + inverseMatrix.toString());
		System.out.println("inverseArray=" + Arrays.deepToString(inverseMatrix.getArrayCopy()));
	}

	@Test
	public void testrandInt()
	{
		int rannum = Utils.randInt(1, 10);
		System.out.println("testrandInt=" + rannum);
		Assert.assertTrue(rannum <= 10);
		Assert.assertTrue(rannum >= 1);
	}

	@Test
	public void testrandPermutation()
	{

		int[] randArray = Utils.randPermutation(8);
		System.out.println("randArray=" + Arrays.toString(randArray));
		Assert.assertTrue(randArray.length == 8);
	}

	@Test
	public void testravel()
	{
		double[] flatarray = Utils.ravel(arr2d);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("flatarray=" + Arrays.toString(flatarray));
	}

	@Test
	public void testravel3D()
	{
		double[] flatarray = Utils.ravel(arr3d);
		System.out.println("arr3d=" + Arrays.deepToString(arr3d));
		System.out.println("flatarray=" + Arrays.toString(flatarray));
	}

	@Test
	public void testcalculateDistance()
	{
		System.out.println("arrdist=" + Arrays.toString(arrdist));
		System.out.println("arrdist2=" + Arrays.toString(arrdist2));
		double retval = Utils.calculateDistance(arrdist, arrdist2);
		System.out.println("testcalculateDistance retval=" + retval);
		Assert.assertTrue(retval == 5.196152422706632);
	}

	@Test
	public void testcalculateDistance2D()
	{
		System.out.println("arrdist2D=" + Arrays.deepToString(arrdist2D));
		System.out.println("arrdist2D2=" + Arrays.deepToString(arrdist2D2));
		double retval = Utils.calculateDistance(arrdist2D, arrdist2D2);
		System.out.println("testcalculateDistance2D retval=" + retval);
	}

	@Test
	public void testdeepCopyOf()
	{

		double[][] testArry = Utils.deep_copy(arr2d);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("testArry=" + Arrays.deepToString(testArry));

		Assert.assertArrayEquals(testArry, arr2d);

	}

	@Test
	public void testgetDimensionCount()
	{
		int dim = Utils.getDimensionCount(arr3d);
		System.out.println("testgetDimensionCount dim3=" + dim);
		Assert.assertTrue(dim == 3);
		dim = Utils.getDimensionCount(arr2d);
		System.out.println("testgetDimensionCount dim2=" + dim);
		Assert.assertTrue(dim == 2);
	}

	@Test
	public void testget_column_slice()
	{
		double[] daSlice = Utils.get_column_slice(arr2d, 4);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("daSlice=" + Arrays.toString(daSlice));
	}

	@Test
	public void testnormalize_rows()
	{
		double[][] normarray = Utils.normalize_rows(arr2d);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("normarray=" + Arrays.deepToString(normarray));

		double[] axis_sum = Utils.axis_sum(normarray, 1);
		Assert.assertTrue(axis_sum.length == normarray.length);
		System.out.println("axis_sum=" + Arrays.toString(axis_sum));

		for (int i = 0; i < axis_sum.length; i++)
		{
			System.out.println("axis_sum[i]=" + axis_sum[i]);
			Assert.assertTrue(axis_sum[i] >= .9999 && axis_sum[i] <= 1.0001);
		}

	}

	@Test
	public void testweighted_mean()
	{
		double[] wm = Utils.weighted_mean(arrdist, arr2d);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("arrdist=" + Arrays.toString(arrdist));
		System.out.println("wm=" + Arrays.toString(wm));
	}

	@Test
	public void testmean()
	{
		double[] mean = Utils.mean(arr2d, 0);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("mean=" + Arrays.toString(mean));
	}

	@Test
	public void teststandard_deviation()
	{
		double[] sd = Utils.standard_deviation(arr2d);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("sd=" + Arrays.toString(sd));
	}

	@Test
	public void testweighted_covariance()
	{
		double[][] wc = Utils.weighted_covariance(arrdist, arr2d);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("weights=" + Arrays.toString(arrdist));
		System.out.println("weightedcovar=" + Arrays.deepToString(wc));
	}

	@Test
	public void testweighted_demean()
	{
		double[][] wdemean = Utils.weighted_demean(arr2d, arrdist);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("weights=" + Arrays.toString(arrdist));
		System.out.println("wdemean=" + Arrays.deepToString(wdemean));
	}

	@Test
	public void testdemean()
	{

		double[][] demean = Utils.demean(arr2d);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("demean=" + Arrays.deepToString(demean));

	}

	@Test
	public void testvstack()
	{
		double[][] vstack = Utils.vstack(arr2d, arr2db);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("arr2db=" + Arrays.deepToString(arr2db));
		System.out.println("vstack=" + Arrays.deepToString(vstack));

	}

	@Test
	public void testvstackCT()
	{
		CoordTuple[] ctresult = Utils.vstack(ct, ct2);
		for (int i = 0; i < ctresult.length; i++)
		{
			System.out.println("CoordTuple[" + i + "] = " + Arrays.toString(ctresult[i].coords));

		}

	}

	@Test
	public void testmatching_argmax_indices()
	{
		//make row 2 have large number in it
		double[][] copyofarr2d = Utils.deep_copy(arr2d);
		copyofarr2d[0][2] = 1000;
		copyofarr2d[2][2] = 6500;
		double[][] arr2dnorm = Utils.normalize_rows(copyofarr2d);
		System.out.println("arr2d=" + Arrays.deepToString(copyofarr2d));
		System.out.println("arr2dnorm=" + Arrays.deepToString(arr2dnorm));
		int[] results = Utils.matching_argmax_indices(arr2dnorm, 2);
		System.out.println("results=" + Arrays.toString(results));

	}

	@Test
	public void testrowwise_argmax()
	{

		int[] awise = Utils.rowwise_argmax(arr2d);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("results=" + Arrays.toString(awise));

	}

	@Test
	public void testdistributional_distance()
	{
		System.out.println("testdistributional_distance arrdist=" + Arrays.toString(arrdist));
		System.out.println("testdistributional_distance arrdist2=" + Arrays.toString(arrdist2));
		double dd = Utils.distributional_distance(arrdist, arrdist2);
		System.out.println("testdistributional_distance dd=" + dd);
	}

	@Test
	public void testmedian()
	{
		double[] arrdistmed = new double[3];
		arrdistmed[0] = 2;
		arrdistmed[1] = 4;
		arrdistmed[2] = 6;
		System.out.println("testmedianarrdist=" + Arrays.toString(arrdistmed));
		System.out.println("testmedianarrdist len=" + arrdistmed.length);
		double dd = Utils.median(arrdistmed);
		System.out.println("median dd=" + dd);
	}

	@Test
	public void testgetabsforarray()
	{
		double[][] absarray = Utils.getAbsForArrays(arr2d, arr2db);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("arr2d=" + Arrays.deepToString(arr2db));
		System.out.println("absarray=" + Arrays.deepToString(absarray));
	}

	@Test
	public void testselect_rows()
	{
		int[] indicies = new int[1];
		indicies[0] = 1;
		double[][] selectedRows = Utils.select_rows(indicies, arr2d);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("selectedRows=" + Arrays.deepToString(selectedRows));

	}

	@Test
	public void testreshape()
	{
		int[] geo = new int[1];
		geo[0] = 18;
		Object dObj = Utils.reshape(Utils.ravel(arr2d), geo);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("datageo=" + Arrays.toString(geo));

		if (geo.length == 1)
			System.out.println("dObj=" + Arrays.toString((double[]) dObj));

		geo = new int[] { 3, 2, 3 };
		dObj = Utils.reshape(Utils.ravel(arr2d), geo);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("datageo=" + Arrays.toString(geo));

		if (geo.length == 2)
			System.out.println("dObj2D=" + Arrays.deepToString((double[][]) dObj));

		if (geo.length == 3)
			System.out.println("dObj3D=" + Arrays.deepToString((double[][][]) dObj));

		if (geo.length == 4)
			System.out.println("dObj4D=" + Arrays.deepToString((double[][][][]) dObj));

	}

	@Test
	public void testslice()
	{
		double[] myslice = Utils.slice(arrdist, 0, 2);
		System.out.println("arrdist=" + Arrays.toString(arrdist));
		System.out.println("myslice=" + Arrays.toString(myslice));
	}

	@Test
	public void testsliceint()
	{
		int[] myslice = Utils.slice(intarray, 2, 3);
		System.out.println("intarray=" + Arrays.toString(intarray));
		System.out.println("myslice=" + Arrays.toString(myslice));
	}

	@Test
	public void testdeletecol()
	{
		double[][] newarray = Utils.delete_column(arr2d, 2);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("newarray=" + Arrays.deepToString(newarray));
	}

	@Test
	public void testproduct()
	{
		double product = Utils.product(arrdist);
		System.out.println("arrdist=" + Arrays.toString(arrdist));
		System.out.println("product=" + product);
	}

	@Test
	public void testproduct2()
	{
		int product = Utils.product(intarray);
		System.out.println("intarray=" + Arrays.toString(intarray));
		System.out.println("product=" + product);
	}

	@Test
	public void testsum()
	{
		double sum = Utils.sum(arrdist);
		System.out.println("arrdist=" + Arrays.toString(arrdist));
		System.out.println("sum=" + sum);

	}

	@Test
	public void testaxis_sum()
	{
		double[] as = Utils.axis_sum(arr2d, 0);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("axissum=" + Arrays.toString(as));

	}

	@Test
	public void testaxis_sum2()
	{
		double[] as = Utils.axis_sum(arr2d, 1);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("axissum2=" + Arrays.toString(as));

	}

	@Test
	public void testmax()
	{
		double max = Utils.max(arrdist);
		System.out.println("arrdist=" + Arrays.toString(arrdist));
		System.out.println("max=" + max);

	}

	@Test
	public void testmax2d()
	{
		double max = Utils.max(arr2d);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("max=" + max);

	}

	@Test
	public void testmin()
	{
		double min = Utils.min(arrdist);
		System.out.println("arrdist=" + Arrays.toString(arrdist));
		System.out.println("min=" + min);
	}

	@Test
	public void testmin2d()
	{
		double min = Utils.min(arr2d);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("min=" + min);
	}

	//TODO:need to add a check for the 2nd matrix side and make sure it works
	@Test(expected = ArrayIndexOutOfBoundsException.class)
	public void testmatrix_multiply()
	{
		System.out.println("testmatrix_multiply arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("testmatrix_multiply arr2db=" + Arrays.deepToString(arr2db));
		double[][] mm = Utils.matrix_multiply(arr2d, arr2db);

		System.out.println("mm=" + Arrays.deepToString(mm));
	}

	@Test
	public void testmatrix_multiplyGood()
	{
		double[][] mm = Utils.matrix_multiply(arr2d, arr2dbmultiply);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("arr2dbmultiply=" + Arrays.deepToString(arr2dbmultiply));
		System.out.println("mm=" + Arrays.deepToString(mm));
	}

	@Test
	public void testelementwise_multiply()
	{
		double[][] mm = Utils.elementwise_multiply(arr2d, arr2db);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("arr2db=" + Arrays.deepToString(arr2db));
		System.out.println("mm=" + Arrays.deepToString(mm));
	}

	@Test
	public void testtranspose()
	{
		double[][] transposed = Utils.transpose(arr2d);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("transposed=" + Arrays.deepToString(transposed));

	}

	@Test
	public void testdot_product()
	{
		double dotprod = Utils.dot_product(arrdist, arrdist2);
		System.out.println("arrdist=" + Arrays.toString(arrdist));
		System.out.println("arrdist2=" + Arrays.toString(arrdist2));
		System.out.println("dotprod=" + dotprod);

	}

	@Test
	public void testcutoff()
	{
		int[] data_geometry = { 10, 100, 100, 1 };
		int h_p = 1;
		int speed = 1;
		int cutoff = Utils.cutoff(data_geometry, h_p, speed);
		System.out.println("data_geo=" + Arrays.toString(data_geometry));
		System.out.println("h_p=" + h_p);
		System.out.println("speed=" + speed);
		System.out.println("cutoff=" + cutoff);

	}

	@Test
	public void testsum_log()
	{
		double sumLog = Utils.sum_log(arrdist);
		System.out.println("arrdist=" + Arrays.toString(arrdist));
		System.out.println("sumLog=" + sumLog);

	}

	@Test
	public void testminint()
	{
		int minme = Utils.min(4, 10);
		System.out.println("a=" + 4);
		System.out.println("b=" + 10);
		System.out.println("minme=" + minme);
	}

	@Test
	public void testdifference()
	{
		double[] dif = Utils.difference(arrdist, arrdist2);
		System.out.println("arrdist=" + Arrays.toString(arrdist));
		System.out.println("arrdist2=" + Arrays.toString(arrdist2));
		System.out.println("dif=" + Arrays.toString(dif));
	}

	@Test
	public void testdeep_copy() {
		double[][] copied = Utils.deep_copy(arr2d);
		System.out.println("arr2d=" + Arrays.deepToString(arr2d));
		System.out.println("copied=" + Arrays.deepToString(copied));
		copied[0][1] = 14.92;
		System.out.println("arr2d (unmodified) =" + Arrays.deepToString(arr2d));
		System.out.println("copied (modified) =" + Arrays.deepToString(copied));
	}

	@Ignore
	@Test
	public void testMahalanobisDist()
	{

		//		// the covariance matrix
		//	    private double[][] S;
		//	    
		//	    public static void main(String[] args) {
		//	        
		//	        double[] x = {2000};
		//	        double[] y = {1999};
		//	        
		//	        Mahalanobis mah = new Mahalanobis(x.length);
		//	        System.out.println(mah.getSimilarity(x, y));
		//	        
		//	    }
		//	    
		//	    public Mahalanobis(int dim) {
		//	        S = new double[dim][dim];
		//	        for(int i=0; i<dim; i++)
		//	            for(int j=0; j<dim; j++)
		//	                if(i == j)
		//	                    S[i][j] = 1.0;
		//	                else
		//	                    S[i][j] = 0.0;
		//	    }
		//	    
		//	    public double getDistance(double[] x, double[] y) {
		//	        double[][] diff = new double[1][x.length];
		//	        for(int i=0; i<x.length; i++)
		//	            diff[0][i] = x[i] - y[i];
		//	        double result[][] = LinearAlgebra.times( diff, LinearAlgebra.inverse(S) );
		//	        result = LinearAlgebra.times( result, LinearAlgebra.transpose(diff) );
		//	        return Math.sqrt( result[0][0] );
		//	    }
		//	    
		//	    public double getSimilarity(double[] x, double[] y) {
		//	        return 1.0 / (1.0 + getDistance(x, y));
		//	    }
		// 

		arrdist[0] = 2;
		arrdist[1] = 4;
		arrdist[2] = 6;
		arrdist2[0] = 8;
		arrdist2[1] = 16;
		arrdist2[2] = 32;

		double[] x = ArrayUtils.clone(arrdist);
		double[] y = ArrayUtils.clone(arrdist2);

		double[][] S = new double[x.length][x.length];
		for (int i = 0; i < x.length; i++)
		{
			for (int j = 0; j < x.length; j++)
			{
				if (i == j)
				{
					S[i][j] = 1.0;
				}
				else
				{
					S[i][j] = 0.0;
				}
			}
		}

		System.out.println("S=" + Arrays.deepToString(S));

		double[][] diff = new double[1][x.length];
		for (int i = 0; i < x.length; i++)
		{
			diff[0][i] = x[i] - y[i];
		}
		System.out.println("diff=" + Arrays.deepToString(diff));

		Matrix matrixS = Matrix.constructWithCopy(S);
		System.out.println("matrixS=" + matrixS.toString());

		Matrix diffMatrix = Matrix.constructWithCopy(diff);
		System.out.println("diffMatrix=" + diffMatrix.toString());

		Matrix resultMatrix = diffMatrix.times(matrixS.inverse());
		System.out.println("resultMatrix=" + resultMatrix.toString());

		Matrix finalResultMatix = resultMatrix.times(diffMatrix.transpose());
		System.out.println("finalResultMatix=" + finalResultMatix.toString());

		double result[][] = finalResultMatix.getArrayCopy();
		System.out.println("result[][]=" + Arrays.deepToString(result));

		double mahalanobisDist = 1.0 / (1.0 + Math.sqrt(result[0][0]));
		System.out.println("mahalanobisDist=" + mahalanobisDist);
	}
}
