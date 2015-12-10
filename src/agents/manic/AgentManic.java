package agents.manic;

import java.util.Random;
import common.IAgent;
import common.ITeacher;
import common.json.JSONObject;
import common.json.JSONArray;
import common.Vec;

/// Implements a weak artificial general intelligence.
public class AgentManic implements IAgent {
	public Random rand;
	public TransitionModel transitionModel;
	public ObservationModel observationModel;
	public ContentmentModel contentmentModel;
	public PlanningSystem planningSystem;
	public double[] actions;
	public double[] beliefs;
	public double[] anticipatedBeliefs;


	// General-purpose constructor.
	public AgentManic(Random r) {
		rand = r;
	}

	public String getName() { return "Manic"; }

	// This method is called to initialize the agent in a new world.
	// oracle is an object that helps the agent learn what to do in this world.
	// observationDims is the number of double values that the agent observes each time step.
	// beliefDims is the number of double values that the agent uses internally to model the state of the world. (It should generally be <= observationDims.)
	// actionDims is the number of double values the agent uses to specify an action.
	// maxPlanLength specifies the maximum number of time-steps into the future that the agent should attempt to plan.
	public void reset(ITeacher oracle, int observationDims, int beliefDims, int actionDims, int maxPlanLength) {
		transitionModel = new TransitionModel(
			actionDims + beliefDims,
			beliefDims,
			2, // number of layers in the transition model
			500, // size of short term memory for transitions
			1000, // number of training iterations to perform with each new sample
			rand);
		observationModel = new ObservationModel(
			transitionModel,
			observationDims,
			beliefDims,
			2, // number of layers in the decoder
			2, // number of layers in the encoder
			500, // size of short term memory for observations
			50, // number of training iterations to perform with each new sample
			500, // number of iterations to calibrate beliefs to correspond with observations
			rand);
		contentmentModel = new ContentmentModel(
			beliefDims,
			2, // number of layers in the contentment model
			500, // size of short term memory for feedback from the teacher
			50, // number of training iterations to perform with each new sample
			rand);
		planningSystem = new PlanningSystem(
			transitionModel,
			observationModel,
			contentmentModel,
			oracle,
			actionDims,
			30, // population size
			50, // number of iterations to refine each member of the population per time step
			500, // burn-in iterations (the number of times at the start to just pick a random action, so the transition function has a chance to explore its space)
			maxPlanLength,
			0.99, // discount factor (to make short plans be preferred over long plans that ultimately arrive at nearly the same state)
			0.0, // exploration rate (the probability that the agent will choose a random action, just to see what happens)
			rand);
		actions = new double[actionDims];
		beliefs = new double[beliefDims];
		anticipatedBeliefs = new double[beliefDims];
	}


	/// Unmarshaling constructor
	public AgentManic(JSONObject obj, Random r, ITeacher oracle) {
		rand = r;
		transitionModel = new TransitionModel((JSONObject)obj.get("transition"), r);
		observationModel = new ObservationModel(transitionModel, (JSONObject)obj.get("observation"), r);
		contentmentModel = new ContentmentModel((JSONObject)obj.get("contentment"), r);
		planningSystem = new PlanningSystem((JSONObject)obj.get("planning"), r, transitionModel, observationModel, contentmentModel, oracle);
		actions = new double[transitionModel.actionDims()];
		beliefs = Vec.unmarshal((JSONArray)obj.get("beliefs"));
		anticipatedBeliefs = new double[beliefs.length];
	}


	/// Marshals this agent to a JSON DOM.
	public JSONObject marshal() {
		JSONObject obj = new JSONObject();
		obj.put("transition", transitionModel.marshal());
		obj.put("observation", observationModel.marshal());
		obj.put("contentment", contentmentModel.marshal());
		obj.put("planning", planningSystem.marshal());
		obj.put("beliefs", Vec.marshal(beliefs));
		return obj;
	}


	/// Replaces the teacher with the specified one
	public void setTeacher(ITeacher oracle) {
		planningSystem.setTeacher(oracle);
	}


	/// Learns from observations
	void learnFromExperience(double[] observations) {

		// Learn to perceive the world a little better
		observationModel.trainIncremental(observations);

		// Refine beliefs to correspond with the new observations better
		observationModel.calibrateBeliefs(anticipatedBeliefs, observations);

		// Learn to anticipate consequences a little better
		transitionModel.trainIncremental(beliefs, actions, anticipatedBeliefs);
	}


	/// Returns an action vector
	double[] decideWhatToDo() {

		// Make the anticipated beliefs the new beliefs
		double[] tmp = beliefs;
		beliefs = anticipatedBeliefs;
		anticipatedBeliefs = tmp;

		// Drop the first action in every plan
		planningSystem.advanceTime();

		// Try to make the plans better
		planningSystem.refinePlans(beliefs);

		// Choose an action that is expected to maximize contentment (with the assistance of the teacher, if available)
		planningSystem.chooseNextActions(beliefs, actions);

		// Anticipate how the world will change with time
		transitionModel.anticipateNextBeliefsInPlace(beliefs, actions, anticipatedBeliefs);

		// Return the selected actions
		return actions;
	}


	/// A vector of observations goes in. All observed values may be expected to fall between -1 and 1.
	/// Returns a vector of chosen actions. All returned values should fall between 0 and 1.
	public double[] think(double[] observations) {

		// Check the observations
		for(int i = 0; i < observations.length; i++) {
			if(observations[i] < -1.0 || observations[i] > 1.0)
				throw new IllegalArgumentException("Observed values must be between -1 and 1.");
		}

		learnFromExperience(observations);
		return decideWhatToDo();
	}

/*
	public static void testMarshaling() throws Exception {
		// Make an agent
		AgentManic agent = new AgentManic(
			new Random(1234),
			new MyTeacher(),
			8, // observation dims
			3, // belief dims
			2, // action dims
			10); // max plan length

		// Write it to a file
		JSONObject obj = agent.marshal();
		FileWriter file = new FileWriter("test.json");
		file.write(obj.toJSONString());
		file.close();

		// Read it from a file
		JSONParser parser = new JSONParser();
		JSONObject obj2 = (JSONObject)parser.parse(new FileReader("test.json"));
		AgentManic agent2 = new AgentManic(obj2, new Random(1234), new MyTeacher());

		System.out.println("passed");
	}
*/
}
