package bluesky;

public class LightConesSet {
	private LightConesCollection trainCordAndData;
	private LightConesCollection testCordAndData;

	public LightConesSet(LightConesCollection trainCordAndData, LightConesCollection testCordAndData) {
		this.trainCordAndData = trainCordAndData;
		this.testCordAndData = testCordAndData;
	}

	public LightConesCollection getTrainCordAndData() {
		return trainCordAndData;
	}

	public LightConesCollection getTestCordAndData() {
		return testCordAndData;
	}

}
