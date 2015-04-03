
interface ITeacher {

	/// Compares two candidate plans.
	/// Implementations should
	/// returns 1 if planA is preferred over planB,
	/// returns 0 if no preference can be determined (or if the teacher is not currently available), and
	/// returns -1 if planA is less preferred than planB.
	///
	/// Note that the task is only to compare the final states anticipated by these plans.
	/// However, in partially observable environments, observations do not fully identify the state,
	/// so it may be helpful for the teacher to view the entire plans in order to assess their final outcomes.
	/// Consequently, the information necessary to reconstruct all of the planned observations
	/// is provided. For example, the teacher may obtain a sequence of planned observations
	/// with code like this:
	///
	///   double[] observations = observationModel.beliefsToObservations(beliefs);
	///   ...
	///   for(int i = 0; i < planA.size(); i++) {
	///     beliefs = transitionModel.anticipateNextBeliefs(beliefs, planA.getActions(i));
	///     observations = observationModel.beliefsToObservations(beliefs);
	///     ...
	///   }
	///
	int compare(double[] beliefs, Plan planA, Plan planB, TransitionModel transitionModel, ObservationModel observationModel);
}
