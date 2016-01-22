package utils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import com.fasterxml.jackson.databind.ObjectMapper;

public class ReadWordPhoneJson {

	public static WordPhonemeObj readInJson(String filepath) throws IOException
	{
		byte[] jsonData = Files.readAllBytes(Paths.get(filepath));
		ObjectMapper objectMapper = new ObjectMapper();
		return objectMapper.readValue(jsonData, WordPhonemeObj.class);

	}
}
