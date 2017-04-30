import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeMap;

public class Main {
	
	@SuppressWarnings("static-access")
	public static void main(String[] args) throws InterruptedException,
	IOException {
		
		String fileName = "";
	    String chemin = "";
	    
	    //On vérifie s'il y a des arguments en entrée
		if (args.length == 1){
			chemin = System.getProperty("user.dir")+"/";
			fileName = args[0];
		}
			//Sinon on utilise des chemins par défaut 
		else {
			fileName = "domaine_public_fluvial.txt";
			chemin = "/cal/homes/malani/systèmes_répartis/";
		}
				
		// Dico Machines - UM
		HashMap<String,String> hm = new HashMap<String, String>();
		// Dico mots - UMX
		HashMap<String,ArrayList<String>> algoMaster3 = new HashMap<String, ArrayList<String>>();
	    
	    
		//Pour chaque étape, on calcule un temps d'exécution qu'on affichera à la fin
		
		
	    //-----------------------------------------------------------------------------------------//
	    //--------- Démarrage  --------------------------------------------------------------------// 
	    //-----------------------------------------------------------------------------------------//
	    long startTime = System.currentTimeMillis();
	    
	    
	    // On ping les machines et on récupère les machines connectées
		ArrayList<String> machines = ConnectedMachines.connectedMachines(
				chemin+"machines.txt", chemin+"machinesConnect.txt");
	
		
		
		//On load un fichier de pronoms et de mots pas utiles à l'analyse
		ArrayList<String> stopWords = new ArrayList<String>();
		BufferedReader fileStopWords = new BufferedReader(new FileReader(new File(chemin+"stopWords.txt")));
		java.lang.String line = fileStopWords.readLine();
		// On normalise la liste des mots
		while (line!=null){
			stopWords.add(FileSplit.normalizar(line).trim());
			line = fileStopWords.readLine();
		}
		fileStopWords.close();
		
		
	    long endTime   = System.currentTimeMillis();
	    long totalTimeDemarrage = endTime - startTime; 
	    
	    
	    
	    
	    
	    //-----------------------------------------------------------------------------------------//
	    //--------- File Splitting ----------------------------------------------------------------//
	    //-----------------------------------------------------------------------------------------//
	    startTime = System.currentTimeMillis();

	    //On split notre fichier en utilisant la fonction fileSplitting 
	    int nbfichiersS = FileSplit.fileSplitting(chemin+fileName, chemin, machines.size());
		System.out.println(nbfichiersS);

	    endTime   = System.currentTimeMillis();
	    long totalTimeSplitting = endTime - startTime;

	    
	    
	    
	    
	    //-----------------------------------------------------------------------------------------//
		//--------- Mapping -----------------------------------------------------------------------//
	    //-----------------------------------------------------------------------------------------//
		startTime = System.currentTimeMillis();
		
		
		ArrayList<LaunchSlave> listeThread = new ArrayList<LaunchSlave>();
		
		for (int i = 0; i < nbfichiersS; i++) {
			int k = i+1;
			// On lance le slave en nous disposant au chemin et en indiquant le chemin au slave
			LaunchSlave t1 = new LaunchSlave(machines.get(i),"cd /; cd "+chemin+";java -jar SLAVESHAVADOOP.jar "+chemin+" modeSXUMX S"+k, 20);
			t1.start();
			//On ajoute les machines et les UM
			hm.put(machines.get(i), "UM"+k);
			listeThread.add(t1);
		}
		
		
		//Pour chaque thread, on ajoute nos éléments à notre dictionnaire 
		for (LaunchSlave thread : listeThread){
			//thread.start();
			thread.join();
			//On boucle sur les outputs
			for (String item : thread.getOutput()){
				//On vérifie si la clé est déjà dans le dico
				if (algoMaster3.containsKey(item)){		
					algoMaster3.get(item).add(hm.get(thread.getMachine()));
					}
				//Sinon on crée la clé
				else {
					ArrayList<String> tempo = new ArrayList<String>();
					tempo.add(hm.get(thread.getMachine()));
					algoMaster3.put(item, tempo);
				}
			}
		}
		
		System.out.println(hm.toString());
		System.out.println(algoMaster3.toString());
		
	    endTime   = System.currentTimeMillis();
	    long totalTimeMapping = endTime - startTime;
	    
	    
	    
	    
	    
	    //-----------------------------------------------------------------------------------------//
		//--------- Shuffling + Reducing ----------------------------------------------------------//
	    //-----------------------------------------------------------------------------------------//
	    
	    startTime = System.currentTimeMillis();

	    
		//Un itérateur pour obtenir l'index du HashMap algoMaster3
		int i=0;
		int k=0;
		ArrayList<LaunchSlave> listeThread2 = new ArrayList<LaunchSlave>();
		
		// On itère sur le dictionnaire mots / UM pour lancer la phase de shuffling, mapping
		for(String key : algoMaster3.keySet()){
			//On ajoute le chemin pour indiquer au slave ou stocker les fichiers créés + le mode + le mot + SM
			String arguments = chemin+" modeUMXSMX";
			arguments+=" "+key;
			arguments+=" "+"SM"+i;
			
			// On crée un Hashet pour ne pas avoir des UM dupliqués dans les valeurs du dictionnaire
			HashSet<String> uniqueMachines = new HashSet<>(algoMaster3.get(key));
			for(String elt : uniqueMachines)
				arguments=arguments+" "+elt;

		
			//Si l'itérateur k dépasse le nombre de machines, on le remet à 0
			if(k==machines.size())
				k=0;
			
			String mach = machines.get(k);
			
			//On lance le slave en nous positionnant dans le répertoire choisi et en utilisant la commande arguments
			LaunchSlave t2 = new LaunchSlave(mach,"cd /; cd "+chemin+"; java -jar SLAVESHAVADOOP.jar "+arguments , 30);
			// On pause le thread 40 millisecondes pour ne pas "bombarder" les machines
			t2.sleep(40);
			
			t2.start();

			listeThread2.add(t2);
		
			k++;
			i++;
		}
	    
	    //On crée notre fichier de sortie au chemin indiqué
		BufferedWriter writer = new BufferedWriter(new FileWriter(new File(
				chemin+"output_"+fileName)));
		
		ArrayList<String> rmList = new ArrayList<String>();

		//On join nos threads et on ajoute les output dans une arraylist
		for (LaunchSlave thread : listeThread2){
			//thread.start();
			thread.join();
			for(String rm : thread.getOutput())
				rmList.add(rm);
			}
		
		//On pop-up le résultat avec gedit
		LaunchSlave t7 = new LaunchSlave(machines.get(4),"cd /; cd "+chemin+"; gedit output_"+fileName , 20);
		t7.start();
		
		//On supprime nos fichiers générés pour ne pas saturer le répértoire 
		LaunchSlave t3 = new LaunchSlave(machines.get(0),"cd /; cd "+chemin+"; rm -rf RM*[0-9]" , 20);
		t3.start();
		LaunchSlave t4 = new LaunchSlave(machines.get(1),"cd /; cd "+chemin+"; rm -rf SM*[0-9]" , 20);
		t4.start();
		LaunchSlave t5 = new LaunchSlave(machines.get(2),"cd /; cd "+chemin+"; rm -rf UM*[0-9]" , 20);
		t5.start();
		LaunchSlave t6 = new LaunchSlave(machines.get(3),"cd /; cd "+chemin+"; rm -rf S*[0-9]" , 20);
		t6.start();
		
	    endTime   = System.currentTimeMillis();
	    long totalTimeReducing = endTime - startTime;
	    
	    
	    
	    
	    
	    
	    //-------------------------------------------------------------------------------------------//
		//--------- Assembling ----------------------------------------------------------------------//
	    //-------------------------------------------------------------------------------------------//
	    startTime = System.currentTimeMillis();
		
	    //On crée un HashMap des mots et du nombre d'occurence
		HashMap<String,Integer> wordCount = new HashMap<String, Integer>();
		
		int count = 0;
		for (String elem : rmList){
			String word = elem.split(" ")[0];
			if(!stopWords.contains(word.trim())){
				wordCount.put(elem,Integer.parseInt(elem.split(" ")[1]));
				count++;
			}
			
		}
		// On trie notre HashMap par valeurs en utilisant la classe ValueComparator
		ValueComparator comparateur = new ValueComparator(wordCount);
		TreeMap<String,Integer> mapTriee = new TreeMap<String,Integer>(comparateur);
		mapTriee.putAll(wordCount);
		
		//On écrit en fichier de sortie notre HashMap trié
		for (String key: mapTriee.keySet())
			writer.write(key+"\n");
		
		writer.close();
		
	    endTime   = System.currentTimeMillis();
	    long totalTimeAssembling = endTime - startTime;
	    
	    
	    
	    //-----------------------------------------------------------------------------------------//
		//--------- Affichage des temps d'exécution -----------------------------------------------//
	    //-----------------------------------------------------------------------------------------//
	   
		System.out.println("\n Tout est fini ! Voici les temps d'exécution\n");
		System.out.println("**---------------**----------------**");
		System.out.println("Temps de Démarrage : "+totalTimeDemarrage/1000);
		System.out.println("Temps de Splitting : "+totalTimeSplitting/1000);
		System.out.println("Temps de Mapping : "+totalTimeMapping/1000);
		System.out.println("Temps de Shuffling + Reducing : "+totalTimeReducing/1000);
		System.out.println("Temps de Assembling : "+totalTimeAssembling/1000);
		System.out.println("Temps total : "+(totalTimeDemarrage+totalTimeSplitting
				+totalTimeMapping+totalTimeReducing+totalTimeAssembling)/1000+" secondes");
		System.out.println("nombre total de mots : " +count);
		System.out.println("**---------------**----------------**");



	}
	
	
}


