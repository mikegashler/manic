package tests;

import common.ITeacher;
import common.IAgent;
import common.Vec;
import java.util.Random;
import java.awt.image.BufferedImage;
import java.io.File;
import agents.manic.AgentManic;
import javax.imageio.ImageIO;
import java.io.IOException;
import java.awt.Color;
import java.awt.Graphics2D;


class MinimizeMagnitudeTeacher implements ITeacher {
	boolean active;

	MinimizeMagnitudeTeacher() {
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



public class DriftingPlatform {

	public static double test(Random rand, IAgent agent) {

		System.out.println("-----------------");
		System.out.println("Drifting platform");
		System.out.println("-----------------");
		System.out.println("In this test, the agent is placed on a 2D platform. Each time-step, the platform " +
				"drifts a small amount in a random direction. The agent seeks to stay near the origin. " +
				"The agent can step in any angular direction, but the size of the step is fixed. " +
				"After 5000 training iterations, the teacher goes away.");

		// Define some constants for this test
		double driftSpeed = 0.1;
		double stepSize = 0.1;
		double controlOrigin = 0.0;

		// Make an agent
		MinimizeMagnitudeTeacher teacher = new MinimizeMagnitudeTeacher();
		agent.reset(teacher, // This teacher prefers plans that lead closer to the origin
			2, // The agent observes its x,y position (which is the complete state of this world)
			2, // the agent models state with 2 dimensions because it cannot be simplified further
			1, // The agent chooses a direction for travel
			10); // The agent plans up to 10 time-steps into the future

		// Train
		System.out.println("Training with the teacher. (The agent learns what it is supposed to do in this world.)");
		System.out.println("|------------------------------------------------|");
		double[] state = new double[2];
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
			double[] act = agent.think(state);
			double angle = act[0] * 2.0 * Math.PI + controlOrigin;
			state[0] += stepSize * Math.cos(angle);
			state[1] += stepSize * Math.sin(angle);
			Vec.clip(state, -1.0, 1.0);

// if(i % 40 == 0){
// System.out.print(Double.toString(Vec.squaredMagnitude(state)) + "	");
// Vec.println(state);
// }
		}

		System.out.println("\nThe teacher now leaves.");
		teacher.active = false;

		System.out.println("\nThe agents controls are adjusted by 120 degrees (to see if it can figure out how to compensate).");
		controlOrigin += Math.PI * 2.0 / 3.0;

		// Train
		System.out.println("Now training without a teacher. (This gives the agent a chance to figure out how to operate effectively with its new controls.)");
		System.out.println("|------------------------------------------------|");
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
			double[] act = agent.think(state);
			double angle = act[0] * 2.0 * Math.PI + controlOrigin;
			state[0] += stepSize * Math.cos(angle);
			state[1] += stepSize * Math.sin(angle);
			Vec.clip(state, -1.0, 1.0);

// if(i % 40 == 0){
// System.out.print(Double.toString(Vec.squaredMagnitude(state)) + "	");
// Vec.println(state);
// }
		}

// BufferedImage image = ((AgentManic)agent).visualizeSpace();
// try{
// ImageIO.write(image, "png", new File("viz.png"));
// }catch(Exception e) {
// throw new IllegalArgumentException("got an exception");
// }

		// Test
		System.out.println("Testing the agent. (If it possesses general intelligence, it should be able to stay near the origin, even though it has never been shown how to do so with the adjusted controls.)");
		System.out.println("|------------------------------------------------|");
		double sumSqMag = 0.0;
		for(int i = 0; i < 1000; i++) {

//			if(i % 20 == 0)
//				System.out.print(">");
double sbn0 = state[0];
double sbn1 = state[1];
			// The platform drifts in a random direction
			drift[0] = rand.nextGaussian();
			drift[1] = rand.nextGaussian();
			Vec.normalize(drift);
			Vec.scale(drift, driftSpeed);
			Vec.add(state, drift);
			Vec.clip(state, -1.0, 1.0);

			// The agent takes a step in a direction of its choice
			double[] act = agent.think(state);

if(i >= 18 && i <= 21) {

System.out.print(Integer.toString(i));
System.out.print("-bef(");
System.out.print(Double.toString(act[0]));
System.out.print(")");
Vec.print(state);
System.out.println(Double.toString(Math.sqrt(Vec.squaredMagnitude(state))));


BufferedImage image = ((AgentManic)agent).visualizeSpace();

double s0b = state[0];
double s1b = state[1];

double angle = act[0] * 2.0 * Math.PI + controlOrigin;
state[0] += stepSize * Math.cos(angle);
state[1] += stepSize * Math.sin(angle);
Vec.clip(state, -1.0, 1.0);

System.out.print(Integer.toString(i));
System.out.print("-aft(");
System.out.print(Double.toString(act[0]));
System.out.print(")");
Vec.print(state);
System.out.println(Double.toString(Math.sqrt(Vec.squaredMagnitude(state))));


Graphics2D g = image.createGraphics();
g.setColor(new Color(0, 255, 0));
g.drawLine((int)((s0b + 1.0) * 500.0), (int)((s1b + 1.0) * 500.0), (int)((state[0] + 1.0) * 500.0), (int)((state[1] + 1.0) * 500.0));

g.setColor(new Color(0, 255, 255));
g.drawLine((int)((sbn0 + 1.0) * 500.0), (int)((sbn1 + 1.0) * 500.0), (int)((s0b + 1.0) * 500.0), (int)((s1b + 1.0) * 500.0));


String filename = "viz" + Integer.toString(i) + ".png";
try{
ImageIO.write(image, "png", new File(filename));
}catch(Exception e) {
throw new IllegalArgumentException("got an exception");
}
}
else{			
			
			

			double angle = act[0] * 2.0 * Math.PI + controlOrigin;
			state[0] += stepSize * Math.cos(angle);
			state[1] += stepSize * Math.sin(angle);
			Vec.clip(state, -1.0, 1.0);
}
			// Track the farthest the agent ever drifts from the origin
			sumSqMag += Vec.squaredMagnitude(state);
// if(i % 20 == 0){
// System.out.print(Double.toString(maxSqMag) + "	");
// Vec.println(state);
// }
		}

		double aveDist = Math.sqrt(sumSqMag / 1000.0);
		System.out.println("Average distance from the origin: " + Double.toString(aveDist));
		return -aveDist; // Bigger is supposed to be better, so we negate the average distance
	}
}
