import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.Normalizer;
import java.util.ArrayList;


public class FileSplit {

	private static BufferedReader br;

    /*We create a method the will "normalize" our words : remove accents, remove big spaces, special characters...*/
	public static String normalizar(String s) {
		String str;
		
			str = Normalizer.normalize(s, Normalizer.Form.NFD);
			str = str.replaceAll(" ", "%20");
			return str.toLowerCase().replaceAll("[^\\[a-z]", "");
	}
	
	/* Method that while clean the entry file word by word and will split it by words in function of the number 
	 * of machines we have.
	 *  */
	public static int fileSplitting(String fichier, String dossSortie, int nbMachines) throws FileNotFoundException, IOException{
		
		//Creation of a bufferedReader of a given file
		br = new BufferedReader(new FileReader(new File(fichier)));
		
		//Creation of an arraylist of Strings that will contain all our words 
		ArrayList<String> mots = new ArrayList<String>();
		try {
			java.lang.String line = br.readLine();
			
			/* We clean each word : replace some word-linkers by " " and normalizing using our function
			 * We add our cleaned words to a dataframe "mots"*/
			while (line != null) {
				if(!line.isEmpty()){
					for (String words: line.replace("'", " ").replace("-", " ").split(" ")){
						if (!words.matches(".*\\d.*"))
							if(normalizar(words).length()>1)
								mots.add(normalizar(words));
					}
				}
				line = br.readLine();
			}
			//the number of words each file should have
			int sizeOfFile = mots.size()/nbMachines;
		
		int i=0;
		
		// We add the words to our file 
		for (int j=1; j<nbMachines; j++){
			BufferedWriter writer = new BufferedWriter(new FileWriter(new File(
					dossSortie+"S"+j)));
			while(i<j*sizeOfFile){
				writer.write(mots.get(i)+" "); 
				i++;}
			writer.close();
		}
		
		// Adding the remaining words
		BufferedWriter writer = new BufferedWriter(new FileWriter(new File(
				dossSortie+"S"+sizeOfFile)));
		while(i<mots.size()){
			writer.write(mots.get(i)+" ");
			System.out.println(mots.get(i));
			i++;
		}
		
		writer.close();
		} catch (IOException exception) {
			System.out.println("Erreur lors de la lecture : "
					+ exception.getMessage());
		}
		return nbMachines;
	
}}
