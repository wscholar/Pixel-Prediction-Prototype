package utils;

import java.util.ArrayList;

public class WordPhonemeObj {
	ArrayList<Words> words = new ArrayList<Words>();
	ArrayList<Phones> phones = new ArrayList<Phones>();

	public ArrayList<Words> getWords() {
		return words;
	}

	public void setWords(ArrayList<Words> words) {
		this.words = words;
	}

	public ArrayList<Phones> getPhones() {
		return phones;
	}

	public void setPhones(ArrayList<Phones> phones) {
		this.phones = phones;
	}

}
