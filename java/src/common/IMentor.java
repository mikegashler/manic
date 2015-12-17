package common;

/// A mentor helps the agent learn what to do.
/// It does not help the agent learn how to do anything.
public interface IMentor {

	public static final double NO_FEEDBACK = -Double.MIN_VALUE;

	/// Implementations should evaluate the goodness of the anticipated observation.
	/// return 1 for the best possible observation.
	/// return 0 for the worst possible observation.
	/// return a value between 0 and 1 for observations that are neither the worst nor best.
	/// return NO_FEEDBACK if the mentor cannot determine the goodness of the anticpated observation,
	///         or if the mentor is not available, or if the mentor wants to test the agent by letting
	///         it decide for itself.
	double evaluate(double[] anticipatedObservations);
}
