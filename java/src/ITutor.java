
/// A tutor helps the agent do certain parts of its job.
/// Using a tutor is typically considered cheating.
/// The main purpose of a tutor is to help debug an agent that is failing to learn some problem.
/// When you find the minimal subset of jobs that the tutor must perform to make the agent successful, you have isolated the bug.
public interface ITutor {

	/// Computes the state from the observations.
	double[] observationsToState(double[] observations);

	/// Computes the observations from the state.
	double[] stateToObservations(double[] state);

	/// Computes how actions will affect state.
	void transition(double[] current_state, double[] actions, double[] next_state);

	/// Computes a near-optimal evaluation of state.
	double evaluateState(double[] state);

	/// Chooses the best actions to perform in the given state.
	void chooseActions(double[] state, double[] actions);
}
