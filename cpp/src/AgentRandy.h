#ifndef AGENTRANDY_H
#define AGENTRANDY_H

#include "Agent.h"
#include <GClasses/GRand.h>
#include <GClasses/GVec.h>
#include <GClasses/GDom.h>
#include <string>

using std::string;

/// Implements a weak artificial general intelligence.
class AgentRandy : public Agent
{
public:
	GRand& rand;
	GVec actions;
	GVec recent_obs;


	// General-purpose constructor.
	AgentRandy(GRand& r);

	virtual ~AgentRandy();

	string getName() { return "Randy"; }

	virtual void reset(Mentor& oracle, size_t observationDims, size_t beliefDims, size_t actionDims, size_t maxPlanLength);

	/// Unmarshaling constructor
	AgentRandy(GDomNode* pNode, GRand& r, Mentor& oracle);

	/// Marshals this agent to a JSON DOM.
	GDomNode* marshal(GDom* pDoc);

	virtual void setMentor(Mentor* oracle);

	virtual void setTutor(Tutor* tutor, bool helpObservationFunction, bool helpTransitionFunction, bool helpContentmentModel, bool helpPlanningSystem);

	virtual void teleport();

	virtual void anticipateObservation(const GMatrix& plan, GVec& obs);

	virtual GVec& think(GVec& observations);
};

#endif
