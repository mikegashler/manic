
public interface ITest {

	/// Evaluates the general intelligence of the agent with some task.
	/// Returns a number that represents the intelligence of the agent.
	/// (More intelligent agents should achieve a higher score.
	/// Less intelligent agents should achieve a lower score.
	/// The scores may be span any range, even negative values.)
	double test(IAgent agent);
}
