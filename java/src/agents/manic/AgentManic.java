package agents.manic;

import java.util.Random;
import common.IAgent;
import common.IMentor;
import common.ITutor;
import common.json.JSONObject;
import common.json.JSONArray;
import common.Vec;
import common.Matrix;

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
	// mentor is an object that helps the agent learn what to do in this world.
	// observationDims is the number of double values that the agent observes each time step.
	// beliefDims is the number of double values that the agent uses internally to model the state of the world. (It should generally be <= observationDims.)
	// actionDims is the number of double values the agent uses to specify an action.
	// maxPlanLength specifies the maximum number of time-steps into the future that the agent should attempt to plan.
	public void reset(IMentor mentor, int observationDims, int beliefDims, int actionDims, int maxPlanLength) {
		if(beliefDims > observationDims)
			throw new IllegalArgumentException("Expected beliefDims to be <= observationDims");
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
			500, // size of short term memory for feedback from the mentor
			50, // number of training iterations to perform with each new sample
			rand);
		planningSystem = new PlanningSystem(
			this,
			transitionModel,
			observationModel,
			contentmentModel,
			mentor,
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
		teleport();
	}


	/// Unmarshaling constructor
	public AgentManic(JSONObject obj, Random r, IMentor mentor) {
		rand = r;
		transitionModel = new TransitionModel((JSONObject)obj.get("transition"), r);
		observationModel = new ObservationModel(transitionModel, (JSONObject)obj.get("observation"), r);
		contentmentModel = new ContentmentModel((JSONObject)obj.get("contentment"), r);
		planningSystem = new PlanningSystem((JSONObject)obj.get("planning"), this, r, transitionModel, observationModel, contentmentModel, mentor);
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


	/// Replaces the mentor with the specified one
	public void setMentor(IMentor mentor) {
		planningSystem.setMentor(mentor);
	}


	/// Sets the tutor to use with this agent
	public void setTutor(ITutor tutor, boolean helpObservationFunction, boolean helpTransitionFunction, boolean helpContentmentModel, boolean helpPlanningSystem) {
		observationModel.setTutor(helpObservationFunction ? tutor : null);
		transitionModel.setTutor(helpTransitionFunction ? tutor : null);
		contentmentModel.setTutor(helpContentmentModel ? tutor : null);
		planningSystem.setTutor(helpPlanningSystem ? tutor : null);
	}


	/// Tells the agent that the next observation passed to learnFromExperience does not follow
	/// from the previous one. This should be called when a game is reset, or when the state is
	/// adjusted in a manner that the agent is not expected to anticipate.
	public void teleport() {
		beliefs[0] = IMentor.NO_FEEDBACK;
	}


	/// Learns from observations
	void learnFromExperience(double[] observations) {

		// Learn to perceive the world a little better
		observationModel.trainIncremental(observations);

		// Refine beliefs to correspond with the new observations better
		observationModel.calibrateBeliefs(anticipatedBeliefs, observations);

		// Learn to anticipate consequences a little better
		if(beliefs[0] != IMentor.NO_FEEDBACK)
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

		// Choose an action that is expected to maximize contentment (with the assistance of the mentor, if available)
		planningSystem.chooseNextActions(beliefs, actions);

		// Anticipate how the world will change with time
		transitionModel.anticipateNextBeliefsInPlace(beliefs, actions, anticipatedBeliefs);

		// Return the selected actions
		return actions;
	}


	/// Anticipates what this agent will observe if the specified plan is performed.
	public double[] anticipateObservation(Matrix plan)
	{
		double[] anticipatedBeliefs = transitionModel.getFinalBeliefs(beliefs, plan);
		double[] anticipatedObs = observationModel.beliefsToObservations(anticipatedBeliefs);
		return anticipatedObs;
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
			new MyMentor(),
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
		AgentManic agent2 = new AgentManic(obj2, new Random(1234), new MyMentor());

		System.out.println("passed");
	}
*/
}
