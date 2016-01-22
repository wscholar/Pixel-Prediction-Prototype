package utils;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import org.apache.commons.io.FileUtils;

public class FileUtility extends FileUtils {

	public static String readFileToStringFromFilepath(String filepath) throws IOException {
		if (filepath == null || filepath.length() == 0 || filepath.equals("/")) {
			return "";
		}
		return FileUtils.readFileToString(new File(filepath));

	}

	public static String readFileToStringFromFilepath(File file) throws IOException {
		if (file == null || file.getName() == null || file.getName().length() == 0) {
			return "";
		}
		return FileUtils.readFileToString(file);

	}

	public static File createDirIfNotExists(String filepath) throws IOException
	{
		File f = new File(filepath);
		if (!f.exists())
		{
			if (f.getParentFile() != null)
			{
				f.getParentFile().mkdirs();
			}

			f.createNewFile();
		}
		return f;
	}
	
	public static int count(File f) throws IOException {
	    InputStream is = new BufferedInputStream(new FileInputStream(f));
	    try {
	        byte[] c = new byte[1024];
	        int count = 0;
	        int readChars = 0;
	        boolean endsWithoutNewLine = false;
	        while ((readChars = is.read(c)) != -1) {
	            for (int i = 0; i < readChars; ++i) {
	                if (c[i] == '\n')
	                    ++count;
	            }
	            endsWithoutNewLine = (c[readChars - 1] != '\n');
	        }
	        if(endsWithoutNewLine) {
	            ++count;
	        } 
	        return count;
	    } finally {
	        is.close();
	    }
	}
}