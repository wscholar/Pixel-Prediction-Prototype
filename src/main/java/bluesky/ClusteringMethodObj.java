package bluesky;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;

public class ClusteringMethodObj {
	public static int[] kmeanspp(double[][] points, int K) {
		int N = points.length;
		List<LCWrapper> clusterInput = new ArrayList<LCWrapper>(N);
		for (int i = 0; i < points.length; i++) {
			clusterInput.add(new LCWrapper(points[i], i));
		}

		// we use KMeans++ with K clusters and 10000 iterations maximum.
		KMeansPlusPlusClusterer<LCWrapper> clusterer = new KMeansPlusPlusClusterer<LCWrapper>(K, 10000);
		List<CentroidCluster<LCWrapper>> clusterResults = clusterer.cluster(clusterInput);

		int[] assignments = new int[N];
		for (int k = 0; k < clusterResults.size(); k++) {
			for (LCWrapper lc : clusterResults.get(k).getPoints()) {
				assignments[lc.getIndex()] = k;
			}
		}

		return assignments;
	}

	public static int[] dbscan(double[][] points, double eps) {
		int N = points.length;
		List<LCWrapper> clusterInput = new ArrayList<LCWrapper>(N);
		for (int i = 0; i < points.length; i++) {
			clusterInput.add(new LCWrapper(points[i], i));
		}

		DBSCANClusterer<LCWrapper> clusterer = new DBSCANClusterer<LCWrapper>(eps, 3);
		List<Cluster<LCWrapper>> clusterResults = clusterer.cluster(clusterInput);

		int[] assignments = new int[N];
		for (int k = 0; k < clusterResults.size(); k++) {
			for (LCWrapper lc : clusterResults.get(k).getPoints()) {
				assignments[lc.getIndex()] = k;
			}
		}

		return assignments;
	}
}
