#include "PlanningSystem.h"
#include <iostream>

using std::cout;

// General-purpose constructor
PlanningSystem::PlanningSystem(TransitionModel& transition, ObservationModel& observation, ContentmentModel& contentment, Mentor* oracle,
	size_t actionDimensions, size_t populationSize, size_t planRefinementIters, size_t burnInIters, size_t maxPlanLen, double discount, double explore, GRand& r)
: transitionModel(transition),
observationModel(observation),
contentmentModel(contentment),
mentor(oracle),
tutor(nullptr),
actionDims(actionDimensions),
burnIn(burnInIters),
discountFactor(discount),
explorationRate(explore),
rand(r),
randomPlan(1, actionDims)
{
	GAssert(randomPlan[0].size() == actionDims);
	if(populationSize < 2)
		throw Ex("The population size must be at least 2");
	refinementIters = populationSize * planRefinementIters;
	maxPlanLength = maxPlanLen;
	for(size_t i = 0; i < populationSize; i++) {
		GMatrix* p = new GMatrix(0, actionDims);
		plans.push_back(p);
		for(size_t j = std::min(maxPlanLen, rand.next(maxPlanLen) + 2); j > 0; j--) {
			// Add a random action vector to the end
			GVec& newActions = p->newRow();
			newActions.fillUniform(rand);
		}
	}
}


/// Unmarshaling constructor
PlanningSystem::PlanningSystem(GDomNode* pNode, GRand& r, TransitionModel& transition, ObservationModel& observation, ContentmentModel& contentment, Mentor* oracle)
: transitionModel(transition),
observationModel(observation),
contentmentModel(contentment),
mentor(oracle),
tutor(nullptr),
maxPlanLength(pNode->field("maxPlanLength")->asInt()),
refinementIters(pNode->field("refinementIters")->asInt()),
actionDims(pNode->field("actionDims")->asInt()),
burnIn(pNode->field("burnIn")->asInt()),
discountFactor(pNode->field("discount")->asDouble()),
explorationRate(pNode->field("explore")->asDouble()),
rand(r),
randomPlan(1, actionDims)
{
	GDomListIterator it(pNode->field("plans"));
	plans.resize(it.remaining());
	for(size_t i = 0; it.current(); i++)
	{
		plans[i] = new GMatrix(it.current());
		it.advance();
	}
}

PlanningSystem::~PlanningSystem()
{
	for(size_t i = 0; i < plans.size(); i++)
		delete(plans[i]);
}

/// Marshals this model to a JSON DOM.
GDomNode* PlanningSystem::marshal(GDom* pDoc)
{
	GDomNode* pNode = pDoc->newObj();
	GDomNode* pPlans = pNode->addField(pDoc, "plans", pDoc->newList());
	for(size_t i = 0; i < plans.size(); i++)
		pPlans->addItem(pDoc, plans[i]->serialize(pDoc));
	pNode->addField(pDoc, "maxPlanLength", pDoc->newInt(maxPlanLength));
	pNode->addField(pDoc, "discount", pDoc->newDouble(discountFactor));
	pNode->addField(pDoc, "explore", pDoc->newDouble(explorationRate));
	pNode->addField(pDoc, "refinementIters", pDoc->newInt(refinementIters));
	pNode->addField(pDoc, "burnIn", pDoc->newInt(burnIn));
	pNode->addField(pDoc, "actionDims", pDoc->newInt(actionDims));
	return pNode;
}


/// Replaces the mentor with the specified one
void PlanningSystem::setMentor(Mentor* oracle)
{
	mentor = oracle;
}


/// Prints a representation of all the plans to stdout
void PlanningSystem::printPlans()
{
	for(size_t i = 0; i < plans.size(); i++)
		plans[i]->print(cout);
}


/// Perturbs a random plan
void PlanningSystem::mutate()
{
	double d = rand.uniform();
	GMatrix& p = *plans[rand.next(plans.size())];
	if(d < 0.1) { // lengthen the plan
		if(p.rows() < maxPlanLength) {
			GVec* newActions = new GVec(actionDims);
			newActions->fillUniform(rand);
			p.takeRow(newActions, rand.next(p.rows() + 1));
		}
	}
	else if(d < 0.2) { // shorten the plan
		if(p.rows() > 1) {
			p.deleteRow(rand.next(p.rows()));
		}
	}
	else if(d < 0.7) { // perturb a single element of an action vector
		GVec& actions = p[rand.next(p.rows())];
		size_t i = rand.next(actions.size());
			actions[i] = std::max(0.0, std::min(1.0, actions[i] + 0.03 * rand.normal()));
	}
	else if(d < 0.9) { // perturb a whole action vector
		GVec& actions = p[rand.next(p.rows())];
		for(size_t i = 0; i < actions.size(); i++) {
			actions[i] = std::max(0.0, std::min(1.0, actions[i] + 0.02 * rand.normal()));
		}
	}
	else { // perturb the whole plan
		for(size_t j = 0; j < p.rows(); j++) {
			GVec& actions = p[j];
			for(size_t i = 0; i < actions.size(); i++) {
				actions[i] = std::max(0.0, std::min(1.0, actions[i] + 0.01 * rand.normal()));
			}
		}
	}
}


/// Replaces the specified plan with a new one.
void PlanningSystem::replace(size_t childIndex)
{
	double d = rand.uniform();
	if(d < 0.2) {
		// Clone a random parent (asexual reproduction)
		size_t randomPlanIndex = rand.next(plans.size() - 1);
		if(randomPlanIndex >= childIndex)
			randomPlanIndex++;
		GMatrix& randPlan = *plans[randomPlanIndex];
		GMatrix* pPlanCopy = new GMatrix(randPlan);
		delete(plans[childIndex]);
		plans[childIndex] = pPlanCopy;
	} else if(d < 0.7) {
		// Cross-over (sexual reproduction)
		GMatrix& mother = *plans[rand.next(plans.size())];
		GMatrix& father = *plans[rand.next(plans.size())];
		size_t crossOverPoint = rand.next(mother.rows());
		GMatrix* pChild = new GMatrix(0, mother.cols());
		for(size_t i = 0; i < crossOverPoint; i++)
			pChild->newRow() = mother[i];
		for(size_t i = crossOverPoint; i < father.rows(); i++)
			pChild->newRow() = father[i];
		delete(plans[childIndex]);
		plans[childIndex] = pChild;
	} else {
		// Interpolation/extrapolation
		GMatrix& mother = *plans[rand.next(plans.size())];
		GMatrix& father = *plans[rand.next(plans.size())];
		size_t len = std::min(mother.rows(), father.rows());
		GMatrix* pChild = new GMatrix(len, mother.cols());
		double alpha = rand.uniform() * 2.0;
		for(size_t i = 0; i < len; i++)
		{
			GVec& a = mother[i];
			GVec& b = father[i];
			GVec& c = (*pChild)[i];
			for(size_t j = 0; j < c.size(); j++)
				c[j] = alpha * a[j] + (1.0 - alpha) * b[j];
			c.clip(0.0, 1.0);
		}
		delete(plans[childIndex]);
		plans[childIndex] = pChild;
	}
}


/// Returns the expected contentment at the end of the plan
double PlanningSystem::evaluatePlan(const GVec& beliefs, GMatrix& plan)
{
	transitionModel.getFinalBeliefs(beliefs, plan, buf);
	return contentmentModel.evaluate(buf) * std::pow(discountFactor, plan.rows());
}


/// Performs a tournament between two randomly-selected plans.
/// One of them, usually the winner, is replaced.
void PlanningSystem::tournament(const GVec& beliefs)
{
	size_t a = rand.next(plans.size());
	size_t b = rand.next(plans.size());
	bool a_prevails;
	if(rand.uniform() < 0.3)
		a_prevails = true; // Let a random plan prevail
	else {
		// Let the better plan prevail
		double fitnessA = evaluatePlan(beliefs, *plans[a]);
		double fitnessB = evaluatePlan(beliefs, *plans[b]);
		if(fitnessA >= fitnessB)
			a_prevails = true;
		else
			a_prevails = false;
	}
	replace(a_prevails ? b : a);
}


/// Performs several iterations of plan refinement
void PlanningSystem::refinePlans(const GVec& beliefs)
{

	// If we are still burning in, then the models are probably not even reliable enough to make refining plans worthwhile
	if(burnIn > 0)
		return;

	for(size_t i = 0; i < refinementIters; i++) {
		double d = rand.uniform();
		if(d < 0.65)
			mutate();
		else
			tournament(beliefs);
	}
}

/*
void PlanningSystem::checkPlans()
{
	for(size_t i = 0; i < plans.size(); i++)
	{
		GMatrix& p = *plans[i];
		for(size_t j = 0; j < p.rows(); j++)
		{
			if(p[j].size() != p.cols())
				throw Ex("found the problem");
		}
	}
}
*/

/// Drops the first action in every plan
void PlanningSystem::advanceTime()
{
	for(size_t i = 0; i < plans.size(); i++)
	{
		GMatrix& p = *plans[i];
		if(p.rows() > 0)
		{
			// Move the first action vector in each plan to the end
			GVec* tmp = p.releaseRowPreserveOrder(0);
			p.takeRow(tmp);
		}
	}
}


/// Asks the mentor to evaluate the plan, given our current beliefs, and learn from it
void PlanningSystem::askMentorToEvaluatePlan(const GVec& beliefs, GMatrix& plan)
{
	transitionModel.getFinalBeliefs(beliefs, plan, buf);
	observationModel.beliefsToObservations(buf, buf2);
	double feedback = mentor->evaluate(buf2);
	if(feedback != UNKNOWN_REAL_VALUE)
		contentmentModel.trainIncremental(buf, feedback);
}


/// Finds the best plan and copies its first step
void PlanningSystem::chooseNextActions(const GVec& beliefs, GVec& actions)
{
	if(tutor)
		tutor->choose_actions(beliefs, actions);
	else
	{
		// Find the best plan (according to the contentment model) and ask the mentor to evaluate it
		size_t planBestIndex = 0;
		double bestContentment = -1e300;
		for(size_t i = 0; i < plans.size(); i++)
		{
			double d = evaluatePlan(beliefs, *plans[i]);
			if(d > bestContentment)
			{
				bestContentment = d;
				planBestIndex = i;
			}
		}
		//std::cout << "Best contentment: " << to_str(bestContentment) << "\n";
		GMatrix& bestPlan = *plans[planBestIndex];
		askMentorToEvaluatePlan(beliefs, bestPlan);

		// Pick a random plan from the population and ask the mentor to evaluate it (for contrast)
		size_t planBindex = rand.next(plans.size() - 1);
		if(planBindex >= planBestIndex)
			planBindex++;
		askMentorToEvaluatePlan(beliefs, *plans[planBindex]);

		// Make a random one-step plan, and ask the mentor to evaluate it (for contrast)
		GVec& action = randomPlan[0];
		action.fillUniform(rand);
		askMentorToEvaluatePlan(beliefs, randomPlan);

		// Copy the first action vector of the best plan for our chosen action
		GVec* bestActions = &bestPlan[0];
		if(burnIn > 0 || rand.uniform() < explorationRate)
			bestActions = &randomPlan[0];
		if(burnIn > 0)
			burnIn--;
		GAssert(bestActions->size() == actionDims);
		actions = *bestActions;
	}
}
