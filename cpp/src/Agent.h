#ifndef AGENT_H
#define AGENT_H

#include <string>
#include <GClasses/GVec.h>

using namespace GClasses;
using std::string;

class Mentor;
class Tutor;

class Agent
{
public:
	virtual ~Agent() {}

	/// Returns this agent's name
	virtual string getName() = 0;

	/// This method is called to initialize the agent in a new world.
	/// oracle is an object that helps the agent learn what to do in this world.
	/// observationDims is the number of double values that the agent observes each time step.
	/// beliefDims is the number of double values that the agent uses internally to model the state of the world. (It should generally be <= observationDims.)
	/// actionDims is the number of double values the agent uses to specify an action.
	/// maxPlanLength specifies the maximum number of time-steps into the future that the agent should attempt to plan.
	virtual void reset(Mentor& oracle, size_t observationDims, size_t beliefDims, size_t actionDims, size_t maxPlanLength) = 0;

	/// Tells the agent that the next observation passed to learnFromExperience does not follow
	/// from the previous one. This should be called when a game is started over, or when the state is
	/// adjusted in a manner that the agent is not expected to anticipate.
	virtual void teleport() = 0;

	/// Sets the mentor to use with this agent
	virtual void setMentor(Mentor* m) = 0;

	/// Sets the tutor to use with this agent.
	virtual void setTutor(Tutor* tutor, bool helpWithObservations, bool helpWithTransitions, bool helpWithContentment, bool helpWithPlanning) = 0;

	/// A vector of observations goes in. All observed values may be expected to fall between -1 and 1.
	/// Returns a vector of chosen actions. All returned values should fall between 0 and 1.
	virtual GVec& think(GVec& observations) = 0;
};

#endif
