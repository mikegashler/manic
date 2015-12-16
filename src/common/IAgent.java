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
	void reset(IMentor oracle, int observationDims, int beliefDims, int actionDims, int maxPlanLength);

	/// Tells the agent that the next observation passed to learnFromExperience does not follow
	/// from the previous one. This should be called when a game is started over, or when the state is
	/// adjusted in a manner that the agent is not expected to anticipate.
	void teleport();

	/// Sets the mentor to use with this agent
	void setMentor(IMentor oracle);

	/// Sets the tutor to use with this agent.
	void setTutor(ITutor tutor, boolean helpObservationFunction, boolean helpTransitionFunction, boolean helpContentmentModel, boolean helpPlanningSystem);

	/// A vector of observations goes in. All observed values may be expected to fall between -1 and 1.
	/// Returns a vector of chosen actions. All returned values should fall between 0 and 1.
	double[] think(double[] observations);
}
