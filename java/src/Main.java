import java.util.Random;
import java.util.ArrayList;
import common.IAgent;
import common.ITest;
import common.Matrix;
import agents.manic.AgentManic;
import agents.randy.AgentRandy;
import tests.DriftingPlatform;

public class Main {

	static void gauntlet(ArrayList<IAgent> agents, ArrayList<ITest> tests) {

		Matrix results = new Matrix(tests.size(), agents.size());

		// Evaluate every agent against every test
		for(int i = 0; i < tests.size(); i++) {
			ITest challenge = tests.get(i);
			for(int j = 0; j < agents.size(); j++) {
				IAgent agent = agents.get(j);
				double result = challenge.test(agent);
				results.row(i)[j] = result;
			}
		}

		System.out.println("\n\n");
		System.out.println("-------------");
		System.out.println("Final results");
		System.out.println("-------------");
		System.out.print("[" + agents.get(0).getName());
		for(int i = 1; i < agents.size(); i++) {
			System.out.print("," + agents.get(i).getName());
		}
		System.out.println("]");
		results.print();
	}

	public static void main(String[] args) throws Exception {

		Random r = new Random(0);
		
		// Make a list of agents
		ArrayList<IAgent> agents = new ArrayList<IAgent>();
		agents.add(new AgentRandy(r));
		agents.add(new AgentManic(r));
		
		// Make a list of tests
		ArrayList<ITest> tests = new ArrayList<ITest>();
		tests.add(new DriftingPlatform(r));

		// Run the agents through the gauntlet
		gauntlet(agents, tests);
	}
}
