#include "AgentManic.h"

AgentManic::AgentManic(GRand& r)
: rand(r)
{
}

AgentManic::~AgentManic()
{
	delete(transitionModel);
	delete(observationModel);
	delete(contentmentModel);
	delete(planningSystem);
}

// virtual
void AgentManic::reset(Mentor& oracle, size_t observationDims, size_t beliefDims, size_t actionDims, size_t maxPlanLength)
{
	if(beliefDims > observationDims)
		throw Ex("Expected beliefDims to be <= observationDims");
	transitionModel = new TransitionModel(
		actionDims + beliefDims,
		beliefDims,
		2, // number of layers in the transition model
		500, // size of short term memory for transitions
		1000, // number of training iterations to perform with each new sample
		rand);
	observationModel = new ObservationModel(
		*transitionModel,
		observationDims,
		beliefDims,
		2, // number of layers in the decoder
		2, // number of layers in the encoder
		500, // size of short term memory for observations
		50, // number of training iterations to perform with each new sample
		500, // number of iterations to calibrate beliefs to correspond with observations
		rand);
	contentmentModel = new ContentmentModel(
		beliefDims,
		2, // number of layers in the contentment model
		500, // size of short term memory for feedback from the mentor
		50, // number of training iterations to perform with each new sample
		rand);
	planningSystem = new PlanningSystem(
		*transitionModel,
		*observationModel,
		*contentmentModel,
		&oracle,
		actionDims,
		30, // population size
		50, // number of iterations to refine each member of the population per time step
		500, // burn-in iterations (the number of times at the start to just pick a random action, so the transition function has a chance to explore its space)
		maxPlanLength,
		0.99, // discount factor (to make short plans be preferred over long plans that ultimately arrive at nearly the same state)
		0.0, // exploration rate (the probability that the agent will choose a random action, just to see what happens)
		rand);
	actions.resize(actionDims);
	actions.fill(0.0);
	beliefs.resize(beliefDims);
	beliefs.fill(0.0);
	anticipatedBeliefs.resize(beliefDims);
	anticipatedBeliefs.fill(0.0);
	teleport();
}


AgentManic::AgentManic(GDomNode* pNode, GRand& r, Mentor& oracle)
: rand(r)
{
	transitionModel = new TransitionModel(pNode->field("transition"), r);
	observationModel = new ObservationModel(*transitionModel, pNode->field("observation"), r);
	contentmentModel = new ContentmentModel(pNode->field("contentment"), r);
	planningSystem = new PlanningSystem(pNode->field("planning"), r, *transitionModel, *observationModel, *contentmentModel, &oracle);
	actions.resize(transitionModel->actionDims());
	beliefs.deserialize(pNode->field("beliefs"));
	anticipatedBeliefs.resize(beliefs.size());
}


GDomNode* AgentManic::marshal(GDom* pDoc)
{
	GDomNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "transition", transitionModel->marshal(pDoc));
	pNode->addField(pDoc, "observation", observationModel->marshal(pDoc));
	pNode->addField(pDoc, "contentment", contentmentModel->marshal(pDoc));
	pNode->addField(pDoc, "planning", planningSystem->marshal(pDoc));
	pNode->addField(pDoc, "beliefs", beliefs.serialize(pDoc));
	return pNode;
}


void AgentManic::setMentor(Mentor* oracle)
{
	planningSystem->setMentor(oracle);
}


void AgentManic::setTutor(Tutor* tutor, bool helpObservationFunction, bool helpTransitionFunction, bool helpContentmentModel, bool helpPlanningSystem)
{
	observationModel->setTutor(helpObservationFunction ? tutor : nullptr);
	transitionModel->setTutor(helpTransitionFunction ? tutor : nullptr);
	contentmentModel->setTutor(helpContentmentModel ? tutor : nullptr);
	planningSystem->setTutor(helpPlanningSystem ? tutor : nullptr);
}


// virtual
void AgentManic::teleport()
{
	beliefs[0] = UNKNOWN_REAL_VALUE;
}


void AgentManic::learnFromExperience(GVec& observations)
{
	// Learn to perceive the world a little better
	observationModel->trainIncremental(observations);

	// Refine beliefs to correspond with the new observations better
	observationModel->calibrateBeliefs(anticipatedBeliefs, observations);

	// Learn to anticipate consequences a little better
	if(beliefs[0] != UNKNOWN_REAL_VALUE)
		transitionModel->trainIncremental(beliefs, actions, anticipatedBeliefs);
}


GVec& AgentManic::decideWhatToDo()
{
	// Make the anticipated beliefs the new beliefs
	GVec& tmp = beliefs;
	beliefs = anticipatedBeliefs;
	anticipatedBeliefs = tmp;

	// Drop the first action in every plan
	planningSystem->advanceTime();

	// Try to make the plans better
	planningSystem->refinePlans(beliefs);

	// Choose an action that is expected to maximize contentment (with the assistance of the mentor, if available)
	planningSystem->chooseNextActions(beliefs, actions);

	// Anticipate how the world will change with time
	transitionModel->anticipateNextBeliefs(beliefs, actions, anticipatedBeliefs);

	// Return the selected actions
	return actions;
}


// virtual
GVec& AgentManic::think(GVec& observations)
{
	// Check the observations
	for(size_t i = 0; i < observations.size(); i++) {
		if(observations[i] < -1.0 || observations[i] > 1.0)
			throw Ex("Observed values must be between -1 and 1.");
	}

	learnFromExperience(observations);
	return decideWhatToDo();
}
