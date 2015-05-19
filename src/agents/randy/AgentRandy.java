package agents.randy;

import java.util.Random;
import common.IAgent;
import common.ITeacher;
import common.json.JSONObject;


// A poor agent that just picks random actions
public class AgentRandy implements IAgent {
	double[] actions;
	Random rand;


	// General-purpose constructor.
	public AgentRandy(Random r) {
		rand = r;
	}

	public String getName() { return "Randy"; }

	// This method is called to initialize the agent in a new world.
	// oracle is an object that helps the agent learn what to do in this world.
	// observationDims is the number of double values that the agent observes each time step.
	// beliefDims is the number of double values that the agent uses internally to model the state of the world. (It should generally be <= observationDims.)
	// actionDims is the number of double values the agent uses to specify an action.
	// maxPlanLength specifies the maximum number of time-steps into the future that the agent should attempt to plan.
	public void reset(ITeacher oracle, int observationDims, int beliefDims, int actionDims, int maxPlanLength) {
		actions = new double[actionDims];
	}


	/// Unmarshaling constructor
	public AgentRandy(JSONObject obj, Random r, ITeacher oracle) {
		rand = r;
		int actionDims = ((Long)obj.get("actionDims")).intValue();
		actions = new double[actionDims];
	}


	/// Marshals this agent to a JSON DOM.
	public JSONObject marshal() {
		JSONObject obj = new JSONObject();
		obj.put("actionDims", actions.length);
		return obj;
	}


	/// Replaces the teacher with the specified one
	public void setTeacher(ITeacher oracle) {
	}


	/// Ignores the observations and picks random actions.
	public double[] think(double[] observations) {

		for(int i = 0; i < actions.length; i++) {
			actions[i] = rand.nextDouble();
		}

		return actions;
	}
}
