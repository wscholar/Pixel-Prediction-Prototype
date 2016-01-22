package bluesky;

import java.util.ArrayList;

public class Moonshine implements Algorithm {

	public int K_max;
	public ArrayList<Integer> cluster_assignments;
	public double cluster_epsilon = .05;
	public int fold_number = 1;
	public ArrayList<PredictiveState> states;
	private double bandwidth_adjustment_factor;

	public Moonshine(int K_max, double cluster_epsilon, int fold_number, double bandwidth_adjustment_factor)
	{
		this.K_max = K_max;
		this.cluster_assignments = new ArrayList<Integer>();
		this.cluster_epsilon = cluster_epsilon;
		this.fold_number = fold_number;
		this.states = new ArrayList<PredictiveState>();
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
	
	//	public void learn(double[][] histories, double[][] futures)
	//	{
	//int N = histories.size(0);
	//TODO no need to flatten histories or future cause it's already a matrix
	//	        flat_histories = multi_flatten(histories)
	//	        flat_futures = multi_flatten(futures)
	//	        ''' Cluster histories '''
	//ArrayList<Integer> cluster_assignments = Cluster.clustpoints(histories);

	//histories.min(dimension);
	//histories.max(dimension);

	//Nd4j.min(histories, dimension)
	//	        ''' Create 2k + 1 random evaluation points '''
	//	        history_low, history_high = np.amin(flat_histories, axis=0), np.amax(flat_histories, axis=0)
	//	        eval_points = np.random.uniform(low=history_low, high=history_high, size=(2 * self.K_max + 1, flat_histories.shape[1]))

	//	        ''' Evaluate points on each cluster using KDE '''
	//	        vectors = {}
	//	        print "%d clusters found." % len(set(cluster_assignments))
	//	        for c in set(cluster_assignments):
	//	            kde_evals = np.nextafter(DenEst(flat_histories[np.equal(cluster_assignments, c)])(eval_points), 1.)
	//	            vectors[c] = np.log(kde_evals[0]) - np.log(kde_evals[1:])
	//	        ''' Assign vectors to predictive states '''
	//	        if len(vectors) <= self.K_max:
	//	            ''' Few vectors, no need to cluster '''
	//	            print "Few vectors, no need to group."
	//	            for c in set(cluster_assignments):
	//	                bit_mask = np.equal(cluster_assignments, c)
	//	                ps = self.predictive_state(flat_histories[bit_mask], flat_futures[bit_mask], N)
	//	                print "State %d has %d points" % (c, bit_mask.sum())
	//	                self.states.append(ps)
	//	            self.cluster_assignments = cluster_assignments
	//	        else:
	//	            ''' Cluster vectors to assign histories to predictive states '''
	//	            ordered_vectors = np.array([vectors[c] for c in sorted(vectors)]).reshape((len(vectors), -1))   
	//	            print "self.K_max", self.K_max
	//	            ''' Use k-means++ for clustering, based on reduced dim vectors '''
	//	            mbkm = MiniBatchKMeans(n_clusters=self.K_max, init='k-means++', compute_labels=True)
	//	            mbkm.fit(ordered_vectors)
	//	            labels = mbkm.labels_
	//	            self.cluster_assignments = np.ones(len(flat_histories)) * -1
	//	            ''' Build states from cluster assignments '''
	//	            for label in set(labels):
	//	                vectors_in_state = [i for i, is_in in enumerate(np.equal(labels, label)) if is_in]
	//	                indices_in_vectors = np.repeat(False, cluster_assignments.shape[0])
	//	                for v in vectors_in_state:
	//	                    indices_in_vectors = np.logical_or(indices_in_vectors, np.equal(cluster_assignments, v))
	//	                ps = self.predictive_state(flat_histories[indices_in_vectors], flat_futures[indices_in_vectors], N)
	//	                self.cluster_assignments[indices_in_vectors] = label
	//	                self.states.append(ps)
	//	        ''' Compute mean future and mean history for each state '''                
	//	        self.state_mean_futures_array = np.array([s.mean_future for s in self.states])
	//	        self.state_mean_histories_array = np.array([s.mean_history for s in self.states])      
	//}
	//	from __future__ import division
	//	import numpy as np
	//	from collections import defaultdict
	//	from sklearn.cluster import MiniBatchKMeans
	//	from scipy.spatial.distance import cdist
	//	from time import time
	//	from multi_flatten import multi_flatten
	//	from cluster import *
	//	from WeightedKDE import WeightedKDE as DenEst
	//	from scipy import stats
	//	from sklearn.neighbors import KernelDensity
	//	from sklearn.grid_search import GridSearchCV
	//
	//	class Moonshine(object):
	//	    class predictive_state(object):        
	//	        def __init__(self, histories, futures, N):
	//	            ''' Assumes histories and futures already flattened '''
	//	            self.num_of_subsamples = 500
	//	            self.mean_future = np.mean(futures, axis=0)
	//	            self.mean_history = np.mean(histories, axis=0)
	//	            self.total_points = N
	//	            self.state_points = histories.shape[0]
	//	            self.FLC_DE = DenEst(futures, num_subsamples=self.num_of_subsamples)
	//	            self.PLC_DE = DenEst(histories, num_subsamples=self.num_of_subsamples)
	//
	//	        def distance_to_state(self, point):
	//	            point = point.reshape((1,-1))
	//	            return cdist(point, self.mean_history, 'euclidean').item()
	//	           
	//	        def batch_lc_likelihood_given_state(self, points):
	//	            return self.PLC_DE(points)
	//
	//	        def batch_emission_likelihood_given_state(self, futures):
	//	            return self.FLC_DE(futures)
	//
	//	        def state_likelihood(self):
	//	            return self.state_points / self.total_points
	//
	//	    def __init__(self, K_max, cluster_epsilon=0.5, fold_number=1):
	//	        self.K_max = K_max
	//	        self.state_weights = []
	//	        self.states = []
	//	        self.cluster_assignments = []
	//	        self.cluster_epsilon = cluster_epsilon
	//	        self.fold_number = fold_number
	//
	//	    def learn(self, histories, futures):
	//	        N = histories.shape[0]
	//	        ''' Flatten out multidimensional X into 2D array'''
	//	        flat_histories = multi_flatten(histories)
	//	        flat_futures = multi_flatten(futures)
	//	        ''' Cluster histories '''
	//	        cluster_assignments = cluster(flat_histories, 'dbscan', assign_to_nearest_center=True, cluster_eps=self.cluster_epsilon)
	//	        ''' Create 2k + 1 random evaluation points '''
	//	        history_low, history_high = np.amin(flat_histories, axis=0), np.amax(flat_histories, axis=0)
	//	        eval_points = np.random.uniform(low=history_low, high=history_high, size=(2 * self.K_max + 1, flat_histories.shape[1]))
	//	        ''' Evaluate points on each cluster using KDE '''
	//	        vectors = {}
	//	        print "%d clusters found." % len(set(cluster_assignments))
	//	        for c in set(cluster_assignments):
	//	            kde_evals = np.nextafter(DenEst(flat_histories[np.equal(cluster_assignments, c)])(eval_points), 1.)
	//	            vectors[c] = np.log(kde_evals[0]) - np.log(kde_evals[1:])
	//	        ''' Assign vectors to predictive states '''
	//	        if len(vectors) <= self.K_max:
	//	            ''' Few vectors, no need to cluster '''
	//	            print "Few vectors, no need to group."
	//	            for c in set(cluster_assignments):
	//	                bit_mask = np.equal(cluster_assignments, c)
	//	                ps = self.predictive_state(flat_histories[bit_mask], flat_futures[bit_mask], N)
	//	                print "State %d has %d points" % (c, bit_mask.sum())
	//	                self.states.append(ps)
	//	            self.cluster_assignments = cluster_assignments
	//	        else:
	//	            ''' Cluster vectors to assign histories to predictive states '''
	//	            ordered_vectors = np.array([vectors[c] for c in sorted(vectors)]).reshape((len(vectors), -1))   
	//	            print "self.K_max", self.K_max
	//	            ''' Use k-means++ for clustering, based on reduced dim vectors '''
	//	            mbkm = MiniBatchKMeans(n_clusters=self.K_max, init='k-means++', compute_labels=True)
	//	            mbkm.fit(ordered_vectors)
	//	            labels = mbkm.labels_
	//	            self.cluster_assignments = np.ones(len(flat_histories)) * -1
	//	            ''' Build states from cluster assignments '''
	//	            for label in set(labels):
	//	                vectors_in_state = [i for i, is_in in enumerate(np.equal(labels, label)) if is_in]
	//	                indices_in_vectors = np.repeat(False, cluster_assignments.shape[0])
	//	                for v in vectors_in_state:
	//	                    indices_in_vectors = np.logical_or(indices_in_vectors, np.equal(cluster_assignments, v))
	//	                ps = self.predictive_state(flat_histories[indices_in_vectors], flat_futures[indices_in_vectors], N)
	//	                self.cluster_assignments[indices_in_vectors] = label
	//	                self.states.append(ps)
	//	        ''' Compute mean future and mean history for each state '''                
	//	        self.state_mean_futures_array = np.array([s.mean_future for s in self.states])
	//	        self.state_mean_histories_array = np.array([s.mean_history for s in self.states])        
	//
	//	    def predict_batch_nearest(self, pasts, points_label=None, do_not_cache=False):
	//	        ''' Use mean future for most likely state '''
	//	        state_given_past_probs = []
	//	        for k, s in enumerate(self.states):
	//	            print "Evaluating state", k
	//	            state_given_past_probs.append(s.batch_lc_likelihood_given_state(pasts))
	//	        state_given_past_probs = np.nextafter(state_given_past_probs, 1.).T
	//	        state_given_past_probs /= np.expand_dims(np.sum(state_given_past_probs, axis=1), axis=1)
	//	        assignment_indices = np.argmax(state_given_past_probs, axis=1)
	//	        super_future_matrix = np.tile(self.state_mean_futures_array, [pasts.shape[0],1,1])
	//	        return np.expand_dims(super_future_matrix[np.arange(len(assignment_indices)), assignment_indices], axis=2)
	//
	//	    def predict_batch_avg(self, pasts, points_label=None, do_not_cache=False):
	//	        ''' Take weighted average from states for prediction, weighted by likelihood '''
	//	        state_given_past_probs = []
	//	        for k, s in enumerate(self.states):
	//	            print "Evaluating state", k
	//	            state_given_past_probs.append(s.batch_lc_likelihood_given_state(pasts))
	//	        state_given_past_probs = np.nextafter(state_given_past_probs, 1.).T
	//	        ''' Weight by state likelihood '''
	//	        likelihoods = np.array([s.state_likelihood() for s in self.states])
	//	        state_given_past_probs *= likelihoods
	//	        state_given_past_probs = np.nextafter(state_given_past_probs, 1.)
	//	        ''' Normalize '''
	//	        state_given_past_probs /= np.expand_dims(np.sum(state_given_past_probs, axis=1), axis=1)        
	//	        ''' Return weighted average '''
	//	        return np.dot(state_given_past_probs, self.state_mean_futures_array)
	//
	//	    def predict_batch(self, pasts, points_label=None, do_not_cache=False):
	//	        return self.predict_batch_avg(pasts, points_label, do_not_cache)
	//
	//	    def predict_states(self, pasts):
	//	        ''' Predict most likely state given past '''
	//	        state_given_past_probs = []
	//	        for k, s in enumerate(self.states):
	//	            print "Evaluating state", k
	//	            state_given_past_probs.append(s.batch_lc_likelihood_given_state(pasts))
	//	        state_given_past_probs = np.nextafter(state_given_past_probs, 1.).T
	//	        state_given_past_probs /= np.expand_dims(np.sum(state_given_past_probs, axis=1), axis=1)
	//	        state_assignments = np.argmax(state_given_past_probs, axis=1)       
	//	        return state_assignments    
	//
	//	    def total_prediction_MSE(self, pasts, true_futures):
	//	        pasts = multi_flatten(pasts)
	//	        predictions = self.predict_batch(pasts)
	//	        truth = np.array(true_futures)
	//	        return np.sum((predictions - truth)**2) / len(true_futures)
	//
	//	    def compute_likelihoods(self, pasts, futures):
	//	        state_given_past_probs = []
	//	        future_given_state_probs = []
	//	        for k, s in enumerate(self.states):
	//	            print "Evaluating state", k
	//	            state_given_past_probs.append(s.batch_lc_likelihood_given_state(pasts))
	//	            future_given_state_probs.append(s.batch_emission_likelihood_given_state(futures))
	//	        state_given_past_probs = np.nextafter(state_given_past_probs, 1.).T
	//	        future_given_state_probs = np.nextafter(future_given_state_probs, 1.).T
	//	        ''' Weight by state likelihood '''
	//	        likelihoods = np.array([s.state_likelihood() for s in self.states])
	//	        state_given_past_probs *= likelihoods
	//	        state_given_past_probs = np.nextafter(state_given_past_probs, 1.)
	//	        ''' Normalize '''
	//	        state_given_past_probs /= np.expand_dims(np.sum(state_given_past_probs, axis=1), axis=1)        
	//	        ''' Return mixed likelihoods '''
	//	        return np.nextafter(np.sum(np.multiply(state_given_past_probs, future_given_state_probs), axis=1), 1.)
	//
	//	    def save_params(self, fold):
	//	        pass
	//
	//	    def print_state_info(self):
	//	        for s in self.states:
	//	            print s.mean_future
	//
	//
	//	def main():
	//	    p1 = np.random.random((800000,4,3))
	//	    p2 = np.random.random((200000,4,3)) + 8
	//	    points = np.vstack((p1, p2))
	//	    np.random.shuffle(points)
	//	    ms = Moonshine(K_max=15)
	//	    t1 = time()    
	//	    ms.learn(points[:,:-1], points[:,-1])      
	//	    print "Total Prediction MSE:", ms.total_prediction_MSE(points[:10,:-1], points[:10,-1])
	//	    print "All done %0.2f minutes" % ((time() - t1) / 60)
	//
	//	if __name__ == "__main__":
	//	    main()

}
