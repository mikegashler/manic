import java.util.Random;
import common.IAgent;
import agents.manic.AgentManic;
import agents.randy.AgentRandy;


class Main {

	static void printScore(String testName, double score) {
		System.out.print(testName);
		System.out.print(": ");
		System.out.println(Double.toString(score));
	}

	// Performs a series of tests to evaluate the general intelligence of the agent
	static void runGauntlet(Random rand, IAgent agent) {

		printScore("Drifting platform", tests.DriftingPlatform.test(rand, agent));
		//printScore("Some other test", tests.SomeOtherTest.test(rand, agent));
		//printScore("Yet another test", tests.YetAnotherTest.test(rand, agent));
		
		//
		// ...
		//
		// todo: add your own tests here
		//
		// ...
		//
	}

	public static void main(String[] args) throws Exception {
		Random r = new Random(1234);
		//runGauntlet(r, new AgentRandy(r));
		runGauntlet(r, new AgentManic(r));
	}
}
