package utils;

import java.io.File;
import java.io.IOException;

public class AVProcessing {

	private String jsonInputPath = "";
	private String textInputPath = "";
	private String mp4InputPath = "";
	private String phonemeAudioOutPath = "";
	private String phonemeJpegOutPath = "";

	public AVProcessing(String jsonInputPath, String textInputPath, String mp4InputPath, String phonemeAudioOutPath, String phonemeJpegOutPath) {
		this.jsonInputPath = jsonInputPath;
		this.textInputPath = textInputPath;
		this.mp4InputPath = mp4InputPath;
		this.phonemeAudioOutPath = phonemeAudioOutPath;
		this.phonemeJpegOutPath = phonemeJpegOutPath;
	}

	public void process() throws IOException
	{
		//		Configuration configuration = new Configuration();
		//		// Set path to acoustic model.
		//		configuration.setAcousticModelPath("resource:/edu/cmu/sphinx/models/en-us/en-us");
		//		// Set path to dictionary.
		//		configuration.setDictionaryPath("resource:/edu/cmu/sphinx/models/en-us/cmudict-en-us.dict");
		//		// Set language model.
		//		configuration.setLanguageModelPath("resource:/edu/cmu/sphinx/models/en-us/en-us.lm.bin");
		//
		//		configuration.setGrammarPath(textInputPath);

		File directory = new File(this.mp4InputPath);
		File[] files = directory.listFiles();
		for (File f : files)
		{
			//for each mp4 file get the phoneme file
			if (f.getName().contains(".mp4"))
			{

				//				String transcription = FileUtility.readFileToStringFromFilepath(textInputPath + f.getName().replace(".wav", ".txt"));
				//				transcription = transcription.replaceAll("[^a-zA-Z\\s]", "").toLowerCase();
				//
				//				// A grammar name corresponding to a file music.jsgf
				//				configuration.setGrammarName(f.getName().replace(".wav", ".txt"));
				//				configuration.setUseGrammar(true);
				//
				//				StreamSpeechRecognizer recognizer = new StreamSpeechRecognizer(configuration);
				//
				//				recognizer.startRecognition(new FileInputStream(f.getAbsolutePath()));
				//				SpeechResult result = recognizer.getResult();
				//				recognizer.stopRecognition();
				//
				//				System.out.println("result no filler is=" + result.getResult().getBestResultNoFiller());
				//				System.out.println("best token is=" + result.getResult().getBestFinalToken());
				//				System.out.println("pronouce for word 0 is=" + result.getResult().getBestFinalToken().getData());
				//				//	StreamSpeechRecognizer recognizer = new StreamSpeechRecognizer(configuration);
				//
				//				//loop over results and make sure words are correct and then get timings and timings for phonemes.
				//				//results.get(0).getWord().

				WordPhonemeObj wpObj = readWordPhoneJsonFile(this.jsonInputPath + f.getName().replace(".mp4", ".json"));
				System.out.println("Phone 0 for file = " + this.jsonInputPath + f.getName().replace(".mp4", ".json")
						+ " is " + wpObj.phones.get(0).getPhone());

				//grab images for phone times and write to output dir
				//Runtime.getRuntime().exec("ffmpeg");

				//grap wav files for phone times and write to output dir
				//Runtime.getRuntime().exec("ffmpeg");
			}

		}

	}

	public WordPhonemeObj readWordPhoneJsonFile(String phonemeFile) throws IOException
	{
		return ReadWordPhoneJson.readInJson(phonemeFile);
	}

	public String getJsonInputPath() {
		return jsonInputPath;
	}

	public String getMp4InputPath() {
		return mp4InputPath;
	}

	public String getPhonemeAudioOutPath() {
		return phonemeAudioOutPath;
	}

	public String getPhonemeJpegOutPath() {
		return phonemeJpegOutPath;
	}

}
