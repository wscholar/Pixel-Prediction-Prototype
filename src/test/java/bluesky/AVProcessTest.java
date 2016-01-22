package bluesky;

import java.io.IOException;

import org.junit.Ignore;
import org.junit.Test;

import utils.AVProcessing;

public class AVProcessTest {

	@Ignore
	@Test
	public void testreadInJson() throws IOException {

		String jsonInputPath = "/Users/waynescholar/Downloads/phone_samples/";
		String mp4InputPath = "/Users/waynescholar/Downloads/mp4_samples/";
		String phonemeAudioOutPath = "";
		String phonemeJpegOutPath = "";
		String textInputPath = "/Users/waynescholar/Downloads/text_samples/";
		AVProcessing avp = new AVProcessing(jsonInputPath, textInputPath, mp4InputPath, phonemeAudioOutPath, phonemeJpegOutPath);
		avp.process();
		System.out.println("all done");
	}
}
