#ifndef MENTOR_H
#define MENTOR_H

#include <GClasses/GVec.h>

class Agent;

/// Helps the agent learn what it should want to do.
/// Does not tell the agent how to do anything.
class Mentor
{
public:
	/// Implementations should evaluate the goodness of the plan.
	/// return 1 for the best possible plan.
	/// return 0 for the worst possible plan.
	/// return a value between 0 and 1 for observations that are neither the worst nor best.
	/// return UNKNOWN_REAL_VALUE if the mentor cannot determine the goodness of the anticpated observation,
	///         or if the mentor is not available, or if the mentor wants to test the agent by letting
	///         it decide for itself.
	virtual double evaluatePlan(Agent& agent, const GClasses::GMatrix& plan) = 0;
};


/// Helps the agent cheat at performing some of its expected abilities,
/// so you can debug which ones are giving it trouble.
/// A tutor is not typically given to the agent.
class Tutor
{
protected:
	/// The constructor is protected so that the user is forced to make a child class
	/// in order to instantiate a tutor.
	Tutor() {}

public:
	/// Computes state from observations.
	virtual void observations_to_state(const GClasses::GVec& observations, GClasses::GVec& state) = 0;

	/// Computes observations from state.
	virtual void state_to_observations(const GClasses::GVec& state, GClasses::GVec& observations) = 0;

	/// Computes how actions will affect state.
	virtual void transition(const GClasses::GVec& current_state, const GClasses::GVec& actions, GClasses::GVec& next_state) = 0;

	/// Evaluates the utility of a state.
	virtual double evaluate_state(const GClasses::GVec& state) = 0;

	/// Picks the best action for the given state.
	virtual void choose_actions(const GClasses::GVec& state, GClasses::GVec& actions) = 0;
};

#endif
