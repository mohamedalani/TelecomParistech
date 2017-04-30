import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;


public class ConnectedMachines {

	public static ArrayList<String> connectedMachines(String nomFichier,
			String nomSortie) throws IOException, InterruptedException {

		ArrayList<String> machines = new ArrayList<String>();

		try {
			File f = new File(nomFichier);
			FileReader fr = new FileReader(f);
			BufferedReader br = new BufferedReader(fr);

			BufferedWriter writer = new BufferedWriter(new FileWriter(new File(
					nomSortie)));

			try {
				java.lang.String line = br.readLine();
				
				ArrayList<TestConnectionSSH> testMachines = new ArrayList<TestConnectionSSH>();
				
				while (line != null) {
					
					TestConnectionSSH test = new TestConnectionSSH(line, 3);
					test.start();
					testMachines.add(test);
					
					line = br.readLine();
					}
				
				for (TestConnectionSSH test: testMachines){
					test.join();// on attend la fin du test

					if (test.isConnectionOK()) {
						machines.add(test.getMachine());
					}
				}
				
				
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}

				br.close();
				fr.close();
				writer.close();
			} catch (IOException exception) {
				System.out.println("Erreur lors de la lecture : "
						+ exception.getMessage());
			}
		

		return machines;
	}
}

			
