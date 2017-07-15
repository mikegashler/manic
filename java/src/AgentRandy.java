import java.util.Random;


// A poor agent that just picks random actions
public class AgentRandy implements IAgent {
	double[] recentObservation;
	double[] actions;
	Random rand;


	// General-purpose constructor.
	public AgentRandy(Random r) {
		rand = r;
	}

	public String getName() { return "Randy"; }

	// This method is called to initialize the agent in a new world.
	// mentor is an object that helps the agent learn what to do in this world.
	// observationDims is the number of double values that the agent observes each time step.
	// beliefDims is the number of double values that the agent uses internally to model the state of the world. (It should generally be <= observationDims.)
	// actionDims is the number of double values the agent uses to specify an action.
	// maxPlanLength specifies the maximum number of time-steps into the future that the agent should attempt to plan.
	public void reset(IMentor mentor, int observationDims, int beliefDims, int actionDims, int maxPlanLength) {
		actions = new double[actionDims];
	}


	/// Unmarshaling constructor
	public AgentRandy(Json obj, Random r, IMentor mentor) {
		rand = r;
		int actionDims = (int)obj.getLong("actionDims");
		actions = new double[actionDims];
	}


	/// Marshals this agent to a JSON DOM.
	public Json marshal() {
		Json obj = Json.newObject();
		obj.add("actionDims", actions.length);
		return obj;
	}


	/// Replaces the mentor with the specified one
	public void setMentor(IMentor mentor) {
	}


	/// Sets the tutor to use with this agent
	public void setTutor(ITutor tutor, boolean helpObservationFunction, boolean helpTransitionFunction, boolean helpContentmentModel, boolean helpPlanningSystem) {
	}


	/// Does nothing, since this agent has no memory anyway
	public void teleport() {
	}


	/// Ignores the plan and anticipates that the most recent observation will occur again
	public double[] anticipateObservation(Matrix plan)
	{
		return recentObservation;
	}


	/// Ignores the observations and picks random actions
	public double[] think(double[] observations) {
		recentObservation = observations;
		for(int i = 0; i < actions.length; i++) {
			actions[i] = rand.nextDouble();
		}

		return actions;
	}
}
