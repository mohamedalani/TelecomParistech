import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;


public class Test {
	
	
	
  public static void main(String[] args) throws IOException {
	  String chemin = args[0];
	  
	  //We verify the mode : SX to UMX (mapping) or UMX to SMX (shuffling + reducing)
	  if(args[1].toString().equals("modeSXUMX")){
		  HashMap<String,Integer> hm = new HashMap<String, Integer>();
		  try {
			  //reader for SX
			  BufferedReader br = new BufferedReader(new FileReader(new File(chemin+args[2])));
			  
			  try {
				  java.lang.String line = br.readLine();
				  
				  String[] lineSplit = line.split(" ");
				  //We create the writer for the file UMX
				  BufferedWriter writer = new BufferedWriter(new FileWriter(new File(
							chemin+"UM"+args[2].substring(1))));
			
				  // we write each word in the file
				  for (int i = 0; i<lineSplit.length; i++){
					  String word = lineSplit[i];
					  
						writer.write(word+" 1\n");
						// We test if the word is unique
						if(!hm.containsKey(word)){
							hm.put(word, 1);
							System.out.println(word);
						}
				  }
				  
				  writer.close();
				  br.close();
		  	} catch (IOException exception) {
		  		System.out.println(exception.getMessage());
			}
			}catch (FileNotFoundException exception) {
				System.out.println("file"+args[2]+"not found");
			}
	  }
	  
	  // We test if the argument passed is for the mode UMX to SMX/RMX
	  else if(args[1].toString().equals("modeUMXSMX")){
		  // writer for SMX
		  BufferedWriter writer = new BufferedWriter(new FileWriter(new File(
					chemin+args[3])));
		  //writer for RMX
		  BufferedWriter writerRM = new BufferedWriter(new FileWriter(new File(
					chemin+"RM"+args[3].substring(2))));
		  
		  //itérateur pour le nombre d'occurence des mots
		  int nbOccurence = 0;

		  try{
			  // On itére sur nos UMX
			  for(int i=4;i<args.length;++i){

				  BufferedReader br = new BufferedReader(new FileReader(new File(chemin+args[i])));
				  
				  try {
					  String line = br.readLine();
					  while (line != null) {
						  String mot = line.split(" ")[0];
						  if(mot != null)
							  //On vérifie l'égalité avec notre mot en entrée, si c'est le cas, on l'écrit dans le fichier
							  if(args[2].equals(mot)){
								  nbOccurence ++;
								  writer.write(mot+"1\n");
							  }
							  line = br.readLine(); 
				  }
				  } catch (IOException e1) {
						e1.printStackTrace();
					}
				  
				  br.close();

		  }
		  writer.close();
		  
		  writerRM.write((args[2])+" "+nbOccurence);
		  System.out.println((args[2])+" "+nbOccurence);
		  
		  writerRM.close();
		
  	} catch (IOException exception) {
	}

	}
  }
}