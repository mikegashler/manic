
/// A mentor helps the agent learn what to do.
/// It does not help the agent learn how to do anything.
public interface IMentor {

	public static final double NO_FEEDBACK = -Double.MIN_VALUE;

	/// Implementations should evaluate the goodness of the plan.
	/// return 1 for the best possible plan.
	/// return 0 for the worst possible plan.
	/// return a value between 0 and 1 for plans that are neither the worst nor best.
	/// return NO_FEEDBACK if the mentor cannot determine the goodness of the plan,
	///         or if the mentor is not available, or if the mentor wants to test the
	///         agent by letting the agent decide for itself.
	double evaluatePlan(IAgent agent, Matrix plan);
}
