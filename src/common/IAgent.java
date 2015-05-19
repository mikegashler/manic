package common;

public interface IAgent {

	/// Returns this agent's name
	String getName();

	/// This method is called to initialize the agent in a new world.
	/// oracle is an object that helps the agent learn what to do in this world.
	/// observationDims is the number of double values that the agent observes each time step.
	/// beliefDims is the number of double values that the agent uses internally to model the state of the world. (It should generally be <= observationDims.)
	/// actionDims is the number of double values the agent uses to specify an action.
	/// maxPlanLength specifies the maximum number of time-steps into the future that the agent should attempt to plan.
	void reset(ITeacher oracle, int observationDims, int beliefDims, int actionDims, int maxPlanLength);

	/// A vector of observations goes in. All observed values may be expected to fall between -1 and 1.
	/// Returns a vector of chosen actions. All returned values should fall between 0 and 1.
	double[] think(double[] observations);
}
