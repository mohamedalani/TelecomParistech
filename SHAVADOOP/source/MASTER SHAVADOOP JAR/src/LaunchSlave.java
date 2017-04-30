
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class LaunchSlave extends Thread{
	
	/*--------Attributs--------*/
	private String machine;
	private String calcul;
	private int timeout;
	private ArrayList<String> output;

	
	
	/*--------Constructeur--------*/
	LaunchSlave(String machine, String calcul, int timeout){
		this.machine = machine;
		this.calcul = calcul;
		this.timeout = timeout;
		this.output = new ArrayList<String>();
	}
	
	
	
	/*--------Méthodes--------*/
	public void affiche(String texte){
		System.out.println("[TestConnectionSSH "+this.machine+"] "+texte);
	}		
	
	public void launch() throws IOException{
        String[] commande = {"ssh","-o StrictHostKeyChecking=no", this.machine, this.calcul};
        ProcessBuilder pb = new ProcessBuilder(commande);
        pb = pb.redirectErrorStream(true);
        Process p = pb.start();
        
        BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
        String ligne; 
        
        while (( ligne = br.readLine()) != null) { 
               
               // S'il y a une erreur, on relance récursivement
               while (ligne.contains("ssh_exchange") || ligne.contains("timeout") || ligne.contains("X11")) 
        		{
            	   launch();
            	   return;
               }
               
           	   this.output.add(ligne);
           	   affiche(ligne);
               
        }
	}
	
	public void run() {
			try {
				launch();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		

	/*--------Getters & Setters--------*/

	public String getMachine() {
		return this.machine;
	}

	public void setMachine(String machine) {
		this.machine = machine;
	}

	public int getTimeout() {
		return timeout;
	}

	public void setTimeout(int timeout) {
		this.timeout = timeout;
	}
	
	public ArrayList<String> getOutput() {
		return output;
	}

	public void setOutput(ArrayList<String> output) {
		this.output = output;
	}


}

