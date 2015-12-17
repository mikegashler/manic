package tests;

import common.IMentor;
import common.ITutor;
import common.IAgent;
import common.ITest;
import common.Vec;
import java.util.Random;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;


class DriftingPlatformMentor implements IMentor {
	boolean active;

	DriftingPlatformMentor() {
		active = true;
	}

	// Prefer the fantasy that minimizes the magnitude of the observation vector
	public double evaluate(double[] anticipatedObservations) {
		if(!active)
			return NO_FEEDBACK;

		double sqMag = Vec.squaredMagnitude(anticipatedObservations);
		return Math.exp(-sqMag);
	}
}




class DriftingPlatformTutor implements ITutor {
	double controlOrigin;
	double stepSize;
	DriftingPlatformMentor mentor;

	DriftingPlatformTutor(DriftingPlatformMentor m)
	{
		controlOrigin = 0.0;
		stepSize = 0.05;
		mentor = m;
	}

	public double[] observationsToState(double[] observations) {
		return Vec.copy(observations);
	}

	public double[] stateToObservations(double[] state) {
		return Vec.copy(state);
	}

	public void transition(double[] current_state, double[] actions, double[] next_state) {
		Vec.copy(next_state, current_state);
		double angle = actions[0] * 2.0 * Math.PI + controlOrigin;
		next_state[0] += stepSize * Math.cos(angle);
		next_state[1] += stepSize * Math.sin(angle);
		Vec.clip(next_state, -1.0, 1.0);
	}

	public double evaluateState(double[] state) {
		double[] obs = stateToObservations(state);
		return mentor.evaluate(obs);
	}

	public void chooseActions(double[] state, double[] actions) {
		double theta = Math.atan2(state[1], state[0]);
		theta -= controlOrigin;
		theta += Math.PI;
		while(theta < 0.0)
			theta += 1.0;
		while(theta > 1.0)
			theta -= 1.0;
		theta /= (2.0 * Math.PI);
		actions[0] = theta;
	}
}




public class DriftingPlatform implements ITest {

	Random rand;


	public DriftingPlatform(Random r) {
		rand = r;
	}

/*
	/// Generates an image to visualize what's going on inside an AgentManic's artificial brain for debugging purposes
	static BufferedImage visualize(agents.manic.AgentManic agent, double[] state_orig, double[] state_drifted, double[] state) {
		if(agent.beliefs.length != 2)
			throw new IllegalArgumentException("Sorry, this method only works with 2D belief spaces");
	
		// Find the min and max locations
		double[] in = new double[2];
		double mi = Double.MAX_VALUE;
		double ma = -Double.MAX_VALUE;
		double[] min_loc = new double[2];
		double[] max_loc = new double[2];
		for(int y = 0; y < 1000; y++) {
			for(int x = 0; x < 1000; x++) {
				in[0] = ((double)x) / 1000.0 * 2.0 - 1.0;
				in[1] = ((double)y) / 1000.0 * 2.0 - 1.0;
				double out = agent.contentmentModel.evaluate(agent.observationModel.observationsToBeliefs(in));
				if(out < mi) {
					mi = out;
					min_loc[0] = in[0];
					min_loc[1] = in[1];
				}
				if(out > ma) {
					ma = out;
					max_loc[0] = in[0];
					max_loc[1] = in[1];
				}
			}
		}

		// Draw the contours of the contentment function
		BufferedImage image = new BufferedImage(1000, 1000, BufferedImage.TYPE_INT_ARGB);
		for(int y = 0; y < 1000; y++) {
			for(int x = 0; x < 1000; x++) {
				in[0] = ((double)x) * 0.002 - 1.0;
				in[1] = ((double)y) * 0.002 - 1.0;
				double out = (agent.contentmentModel.evaluate(agent.observationModel.observationsToBeliefs(in)) - mi) * 256.0 / (ma - mi);
				int g = Math.max(0, Math.min(255, (int)out));
				int gg = g;
				if(g % 5 == 0)
					gg = (128 - (int)(Math.tanh((double)(g - 128) * 0.03) * 127.0));
				image.setRGB(x, y, new Color(g, g, gg).getRGB());
			}
		}

		// Draw magenta dots at the sample locations for training the contentment function
		Graphics2D g = image.createGraphics();
		g.setColor(new Color(255, 0, 255));
		for(int i = 0; i < agent.contentmentModel.trainSize; i++) {
			double[] r = agent.observationModel.beliefsToObservations(agent.contentmentModel.samples.row(i));
			int x = (int)((r[0] + 1.0) * 500.0);
			int y = (int)((r[1] + 1.0) * 500.0);
			g.fillOval(x - 2, y - 2, 4, 4);
		}

		// Draw the circle of transitions. (If the agent has learned transitions well, these will form a circle around the point of beliefs.)
		double[] tmp_act = new double[1];
		for(double d = 0; d <= 1.0; d += 0.03125) {
			if(d == 0)
				g.setColor(new Color(255, 0, 0));
			else if(d == 0.25)
				g.setColor(new Color(255, 255, 0));
			else if(d == 0.5)
				g.setColor(new Color(0, 255, 0));
			else if(d == 0.75)
				g.setColor(new Color(0, 255, 255));
			tmp_act[0] = d;
			double[] next = agent.transitionModel.anticipateNextBeliefs(agent.beliefs, tmp_act);
			double[] next_obs = agent.observationModel.beliefsToObservations(next);
			g.fillOval((int)((next_obs[0] + 1.0) * 500.0) - 4, (int)((next_obs[1] + 1.0) * 500.0) - 4, 8, 8);
		}

		// Draw an orange circle to represent the agent's beliefs
		g.setColor(new Color(255, 128, 0));
		double[] exp_obs = agent.observationModel.beliefsToObservations(agent.beliefs);
		g.fillOval((int)((exp_obs[0] + 1.0) * 500.0) - 4, (int)((exp_obs[1] + 1.0) * 500.0) - 4, 8, 8);

		// Draw orange lines to represent the plans
		for(int i = 0; i < agent.planningSystem.plans.size(); i++) {
			agents.manic.Plan plan = agent.planningSystem.plans.get(i);
			double[] prev = agent.beliefs;
			double[] prev_obs = exp_obs;
			for(int j = 0; j < plan.steps.size(); j++) {
				double[] next = agent.transitionModel.anticipateNextBeliefs(prev, plan.steps.get(j));
				double[] next_obs = agent.observationModel.beliefsToObservations(next);
				g.drawLine((int)((prev_obs[0] + 1.0) * 500.0), (int)((prev_obs[1] + 1.0) * 500.0), (int)((next_obs[0] + 1.0) * 500.0), (int)((next_obs[1] + 1.0) * 500.0));
				prev = next;
				prev_obs = next_obs;
			}
		}

		// Draw the chosen action in dark green
		g.setColor(new Color(0, 128, 0));
		double[] ant_obs = agent.observationModel.beliefsToObservations(agent.anticipatedBeliefs);
		g.drawLine((int)((exp_obs[0] + 1.0) * 500.0), (int)((exp_obs[1] + 1.0) * 500.0), (int)((ant_obs[0] + 1.0) * 500.0), (int)((ant_obs[1] + 1.0) * 500.0));

		// Draw the actual action in bright green
		g.setColor(new Color(0, 255, 0));
		g.drawLine((int)((state_drifted[0] + 1.0) * 500.0), (int)((state_drifted[1] + 1.0) * 500.0), (int)((state[0] + 1.0) * 500.0), (int)((state[1] + 1.0) * 500.0));

		// Draw the drift in cyan
		g.setColor(new Color(0, 255, 255));
		g.drawLine((int)((state_orig[0] + 1.0) * 500.0), (int)((state_orig[1] + 1.0) * 500.0), (int)((state_drifted[0] + 1.0) * 500.0), (int)((state_drifted[1] + 1.0) * 500.0));

		return image;
	}

	static void makeVisualization(String suffix, agents.manic.AgentManic agent, double[] state_orig, double[] state_drifted, double[] state) {
		BufferedImage image = visualize(agent, state_orig, state_drifted, state);
		String filename = "viz" + suffix + ".png";
		try {
			ImageIO.write(image, "png", new File(filename));
		}catch(Exception e) {
			throw new IllegalArgumentException("got an exception while trying to write file " + filename);
		}
	}
*/


	public double test(IAgent agent) {

		System.out.println("----------------------");
		System.out.println("Drifting platform test       Agent: " + agent.getName());
		System.out.println("----------------------");
		System.out.println("In this test, the agent is placed on an imaginary 2D platform of infinite size. " +
				"The agent's objective is to stay near the origin. Each time-step, the platform " +
				"drifts a small amount in a random direction. The agent can step in any direction " +
				"(from 0 to 2*PI). Initially, a mentor will help it learn what to do.\n");

		// Define some constants for this test
		double driftSpeed = 0.1;

		// Set up the agent
		DriftingPlatformMentor mentor = new DriftingPlatformMentor();
		agent.reset(mentor, // This mentor prefers plans that lead closer to the origin
			2, // The agent observes its x,y position (which is the complete state of this world)
			2, // the agent models state with 2 dimensions because it cannot be simplified further
			1, // The agent chooses a direction for travel
			1); // The agent plans up to 1 time-steps into the future
		DriftingPlatformTutor tutor = new DriftingPlatformTutor(mentor);

		// To debug an agent that isn't working, uncomment the following line and verify that it works.
		// Then, set each "true" to "false" until you find the component that isn't doing its job properly.
		//agent.setTutor(tutor, true/*observation*/, true/*transition*/, true/*contentment*/, true/*planning*/);

		// Train with mentor
		System.out.println("Phase 1 of 3: Learn the objective from the mentor...");
		System.out.println("|------------------------------------------------|");
		double[] state = new double[2];
		double[] next_state = new double[2];
		double[] drift = new double[2];
		for(int i = 0; i < 2000; i++) {

			if(i % 40 == 0)
				System.out.print(">");

			// The platform drifts in a random direction
			drift[0] = rand.nextGaussian();
			drift[1] = rand.nextGaussian();
			Vec.normalize(drift);
			Vec.scale(drift, driftSpeed);
			Vec.add(state, drift);
			Vec.clip(state, -1.0, 1.0);

			// The agent takes a step in a direction of its choice
			double[] obs = tutor.stateToObservations(state);
			double[] act = agent.think(obs);
			tutor.transition(state, act, next_state);
			Vec.copy(state, next_state);
		}

		System.out.println("\n\nNow, the mentor is removed, so the agent is completely on its own.");
		mentor.active = false;

		System.out.println("Also, to make the problem more challenging, the agent's controls " +
				"are changed by 120 degrees. The agent will now have to figure out how to operate " +
				"the new controls without a mentor to help it.\n");
		tutor.controlOrigin += Math.PI * 2.0 / 3.0;

		// Train without mentor
		System.out.println("Phase 2 of 3: Figure out new controls (without mentor)...");
		System.out.println("|------------------------------------------------|");
		for(int i = 0; i < 2000; i++) {

			if(i % 40 == 0)
				System.out.print(">");
//			if(i % 80 == 0)
//				makeVisualization(Integer.toString(i), (agents.manic.AgentManic)agent, state_orig, state_drifted, state);

			// The platform drifts in a random direction
			drift[0] = rand.nextGaussian();
			drift[1] = rand.nextGaussian();
			Vec.normalize(drift);
			Vec.scale(drift, driftSpeed);
			Vec.add(state, drift);
			Vec.clip(state, -1.0, 1.0);

			// The agent takes a step in a direction of its choice
			double[] obs = tutor.stateToObservations(state);
			double[] act = agent.think(obs);
			tutor.transition(state, act, next_state);
			Vec.copy(state, next_state);
		}

		// Test
		System.out.println("\n\nThe agent has had enough time to figure out the new controls, so now we test the agent. " +
				"We will let the platform continue to drift randomly for 1000 iterations, and measure the average " +
				"distance between the origin and the agent. (If the agent is intelligent, it should achieve a low " +
				"average distance, such as 0.2. If it is unintelligent, it will achieve a higher average distance, " +
				"such as 0.7.\n");
		System.out.println("Phase 3 of 3: Testing (without mentor)...");
		System.out.println("|------------------------------------------------|");
		double sumSqMag = 0.0;
		for(int i = 0; i < 1000; i++) {

			if(i % 20 == 0)
				System.out.print(">");

// 			if(i % 100 == 0) 
// 				makeVisualization(Integer.toString(i), (agents.manic.AgentManic)agent, state_orig, state_drifted, state);

			// The platform drifts in a random direction
			drift[0] = rand.nextGaussian();
			drift[1] = rand.nextGaussian();
			Vec.normalize(drift);
			Vec.scale(drift, driftSpeed);
			Vec.add(state, drift);
			Vec.clip(state, -1.0, 1.0);

			// The agent takes a step in a direction of its choice
			double[] obs = tutor.stateToObservations(state);
			double[] act = agent.think(obs);
			tutor.transition(state, act, next_state);
			Vec.copy(state, next_state);

			// Sum up how far the agent ever drifts from the origin
			sumSqMag += Math.sqrt(Vec.squaredMagnitude(state));
		}

		double aveDist = sumSqMag / 1000.0;
		System.out.println("\n\nThe agent's average distance from the origin during the testing phase was " + Double.toString(aveDist));

		return -aveDist; // Bigger is supposed to be better, so we negate the average distance
	}
}
