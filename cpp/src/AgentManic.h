#ifndef AGENTMANIC_H
#define AGENTMANIC_H

#include "Agent.h"
#include <GClasses/GRand.h>
#include <GClasses/GVec.h>
#include <GClasses/GDom.h>
#include "TransitionModel.h"
#include "ObservationModel.h"
#include "ContentmentModel.h"
#include "PlanningSystem.h"
#include <string>

using std::string;

/// Implements a weak artificial general intelligence.
class AgentManic : public Agent
{
public:
	GRand& rand;
	TransitionModel* transitionModel;
	ObservationModel* observationModel;
	ContentmentModel* contentmentModel;
	PlanningSystem* planningSystem;
	GVec actions;
	GVec beliefs;
	GVec anticipatedBeliefs;
	GVec buf;


	// General-purpose constructor.
	AgentManic(GRand& r);

	virtual ~AgentManic();

	string getName() { return "Manic"; }

	virtual void reset(Mentor& oracle, size_t observationDims, size_t beliefDims, size_t actionDims, size_t maxPlanLength);

	/// Unmarshaling constructor
	AgentManic(GDomNode* pNode, GRand& r, Mentor& oracle);

	/// Marshals this agent to a JSON DOM.
	GDomNode* marshal(GDom* pDoc);

	/// Replaces the mentor with the specified one
	virtual void setMentor(Mentor* oracle);

	/// Sets a tutor. (Clears the tutor if tutor is nullptr.)
	virtual void setTutor(Tutor* tutor, bool helpObservationFunction, bool helpTransitionFunction, bool helpContentmentModel, bool helpPlanningSystem);

	/// Tells the agent that the next observation passed to learnFromExperience does not follow
	/// from the previous one. This should be called when a game is reset, or when the state is
	/// adjusted in a manner that the agent is not expected to anticipate.
	virtual void teleport();

	/// Learns from observations
	void learnFromExperience(GVec& observations);

	/// Returns an action vector
	GVec& decideWhatToDo();

	/// Returns the observation that would be expected after performing the plan.
	virtual void anticipateObservation(const GMatrix& plan, GVec& obs);

	/// A vector of observations goes in. All observed values may be expected to fall between -1 and 1.
	/// Returns a vector of chosen actions. All returned values should fall between 0 and 1.
	virtual GVec& think(GVec& observations);
};

#endif
