#include "AgentRandy.h"

AgentRandy::AgentRandy(GRand& r)
: rand(r)
{
}

// virtual
AgentRandy::~AgentRandy()
{
}

// virtual
void AgentRandy::reset(Mentor& oracle, size_t observationDims, size_t beliefDims, size_t actionDims, size_t maxPlanLength)
{
	actions.resize(actionDims);
	actions.fill(0.0);
	teleport();
}


AgentRandy::AgentRandy(GDomNode* pNode, GRand& r, Mentor& oracle)
: rand(r)
{
	actions.resize(pNode->field("actions")->asInt());
}


GDomNode* AgentRandy::marshal(GDom* pDoc)
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "actions", pDoc->newInt(actions.size()));
	return pNode;
}


void AgentRandy::setMentor(Mentor* oracle)
{
}


void AgentRandy::setTutor(Tutor* tutor, bool helpObservationFunction, bool helpTransitionFunction, bool helpContentmentModel, bool helpPlanningSystem)
{
}


// virtual
void AgentRandy::teleport()
{
}


// virtual
void AgentRandy::anticipateObservation(const GMatrix& plan, GVec& obs)
{
	obs.copy(recent_obs);
}


// virtual
GVec& AgentRandy::think(GVec& observations)
{
	recent_obs.copy(observations);
	actions.fillUniform(rand);
	return actions;
}
