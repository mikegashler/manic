package common;

public interface ITeacher {

	public static final double NO_FEEDBACK = -Double.MIN_VALUE;

	/// Implementations should evaluate the goodness of the anticipated observation.
	/// return 1 for the best possible observation.
	/// return 0 for the worst possible observation.
	/// return a value between 0 and 1 for observations that are neither the worst nor best.
	/// return NO_FEEDBACK if the teacher cannot determine the goodness of the anticpated observation,
	//         or if the teacher is not available, or if the teacher wants to test the agent by letting
	//         it decide for itself.
	double evaluate(double[] anticipatedObservations);
}
