package agents.manic;

import java.util.Random;
import java.util.ArrayList;
import java.util.Iterator;
import common.Vec;
import common.json.JSONObject;
import common.json.JSONArray;
import common.IMentor;
import common.ITutor;


/// A genetic algorithm that sequences actions to form a plan intended to maximize contentment.
public class PlanningSystem {
	Plan randomPlan;
	public ArrayList<Plan> plans;
	TransitionModel transitionModel;
	ObservationModel observationModel;
	ContentmentModel contentmentModel;
	IMentor mentor;
	ITutor tutor;
	int maxPlanLength;
	int refinementIters;
	int actionDims;
	int burnIn;
	double discountFactor;
	double explorationRate;
	Random rand;


	// General-purpose constructor
	PlanningSystem(TransitionModel transition, ObservationModel observation, ContentmentModel contentment, IMentor oracle,
		int actionDimensions, int populationSize, int planRefinementIters, int burnInIters, int maxPlanLen, double discount, double explore, Random r) {
		transitionModel = transition;
		observationModel = observation;
		contentmentModel = contentment;
		mentor = oracle;
		rand = r;
		plans = new ArrayList<Plan>();
		if(populationSize < 2)
			throw new IllegalArgumentException("The population size must be at least 2");
		refinementIters = populationSize * planRefinementIters;
		burnIn = burnInIters;
		actionDims = actionDimensions;
		maxPlanLength = maxPlanLen;
		discountFactor = discount;
		explorationRate = explore;
		for(int i = 0; i < populationSize; i++) {
			Plan p = new Plan();
			for(int j = Math.min(maxPlanLen, rand.nextInt(maxPlanLen) + 2); j > 0; j--) {
				// Add a random action vector to the end
				double[] newActions = new double[actionDims];
				for(int k = 0; k < actionDims; k++) {
					newActions[k] = rand.nextDouble();
				}
				p.steps.add(newActions);
			}
			plans.add(p);
		}
		randomPlan = new Plan();
		randomPlan.steps.add(new double[actionDimensions]);
	}


	/// Unmarshaling constructor
	PlanningSystem(JSONObject obj, Random r, TransitionModel transition, ObservationModel observation, ContentmentModel contentment, IMentor oracle) {
		transitionModel = transition;
		observationModel = observation;
		contentmentModel = contentment;
		mentor = oracle;
		rand = r;
		JSONArray plansArr = (JSONArray)obj.get("plans");
		plans = new ArrayList<Plan>();
		Iterator<JSONArray> it = plansArr.iterator();
		while(it.hasNext()) {
			plans.add(new Plan(it.next()));
		}
		maxPlanLength = ((Long)obj.get("maxPlanLength")).intValue();
		discountFactor = ((Double)obj.get("discount")).doubleValue();
		explorationRate = ((Double)obj.get("explore")).doubleValue();
		refinementIters = ((Long)obj.get("refinementIters")).intValue();
		burnIn = ((Long)obj.get("burnIn")).intValue();
		actionDims = ((Long)obj.get("actionDims")).intValue();
		randomPlan = new Plan();
		randomPlan.steps.add(new double[actionDims]);
	}


	/// Marshals this model to a JSON DOM.
	JSONObject marshal() {
		JSONObject obj = new JSONObject();
		JSONArray plansArr = new JSONArray();
		for(int i = 0; i < plans.size(); i++) {
			plansArr.add(plans.get(i).marshal());
		}
		obj.put("plans", plansArr);
		obj.put("maxPlanLength", maxPlanLength);
		obj.put("discount", discountFactor);
		obj.put("explore", explorationRate);
		obj.put("refinementIters", refinementIters);
		obj.put("burnIn", burnIn);
		obj.put("actionDims", actionDims);
		return obj;
	}


	/// Replaces the mentor with the specified one
	void setMentor(IMentor oracle) {
		mentor = oracle;
	}


	void setTutor(ITutor t) {
		tutor = t;
	}


	/// Prints a representation of all the plans to stdout
	void printPlans() {
		for(int i = 0; i < plans.size(); i++)
			plans.get(i).print();
	}


	/// Perturbs a random plan
	void mutate() {
		double d = rand.nextDouble();
		Plan p = plans.get(rand.nextInt(plans.size()));
		if(d < 0.1) { // lengthen the plan
			if(p.size() < maxPlanLength) {
				double[] newActions = new double[actionDims];
				for(int i = 0; i < actionDims; i++) {
					newActions[i] = rand.nextDouble();
				}
				p.steps.add(rand.nextInt(p.size() + 1), newActions);
			}
		}
		else if(d < 0.2) { // shorten the plan
			if(p.size() > 1) {
				p.steps.remove(rand.nextInt(p.size()));
			}
		}
		else if(d < 0.7) { // perturb a single element of an action vector
			double[] actions = p.getActions(rand.nextInt(p.size()));
			int i = rand.nextInt(actions.length);
				actions[i] = Math.max(0.0, Math.min(1.0, actions[i] + 0.03 * rand.nextGaussian()));
		}
		else if(d < 0.9) { // perturb a whole action vector
			double[] actions = p.getActions(rand.nextInt(p.size()));
			for(int i = 0; i < actions.length; i++) {
				actions[i] = Math.max(0.0, Math.min(1.0, actions[i] + 0.02 * rand.nextGaussian()));
			}
		}
		else { // perturb the whole plan
			for(int j = 0; j < p.size(); j++) {
				double[] actions = p.getActions(j);
				for(int i = 0; i < actions.length; i++) {
					actions[i] = Math.max(0.0, Math.min(1.0, actions[i] + 0.01 * rand.nextGaussian()));
				}
			}
		}
	}


	/// Replaces the specified plan with a new one.
	void replace(int childIndex) {
		double d = rand.nextDouble();
		if(d < 0.2) {
			// Clone a random parent (asexual reproduction)
			plans.set(childIndex, new Plan(plans.get(rand.nextInt(plans.size()))));
		} else if(d < 0.7) {
			// Cross-over (sexual reproduction)
			Plan mother = plans.get(rand.nextInt(plans.size()));
			Plan father = plans.get(rand.nextInt(plans.size()));
			int crossOverPoint = rand.nextInt(mother.size());
			Plan child = new Plan();
			for(int i = 0; i < crossOverPoint; i++)
				child.steps.add(Vec.copy(mother.getActions(i)));
			for(int i = crossOverPoint; i < father.size(); i++)
				child.steps.add(Vec.copy(father.getActions(i)));
			plans.set(childIndex, child);		
		} else {
			// Interpolation/extrapolation
			Plan mother = plans.get(rand.nextInt(plans.size()));
			Plan father = plans.get(rand.nextInt(plans.size()));
			int len = Math.min(mother.size(), father.size());
			Plan child = new Plan();
			double alpha = rand.nextDouble() * 2.0;
			for(int i = 0; i < len; i++) {
				double[] a = mother.getActions(i);
				double[] b = father.getActions(i);
				double[] c = new double[a.length];
				for(int j = 0; j < c.length; j++) {
					c[j] = Math.max(0.0, Math.min(1.0, alpha * a[j] + (1.0 - alpha) * b[j]));
				}
				child.steps.add(c);
			}
			plans.set(childIndex, child);
		}
	}


	/// Returns the expected contentment at the end of the plan
	double evaluatePlan(double[] beliefs, Plan plan) {
		return contentmentModel.evaluate(transitionModel.getFinalBeliefs(beliefs, plan)) * Math.pow(discountFactor, plan.steps.size());
	}


	/// Performs a tournament between two randomly-selected plans.
	/// One of them, usually the winner, is replaced.
	void tournament(double[] beliefs) {
		int a = rand.nextInt(plans.size());
		int b = rand.nextInt(plans.size());
		boolean a_prevails;
		if(rand.nextDouble() < 0.3)
			a_prevails = true; // Let a random plan prevail
		else {
			// Let the better plan prevail
			double fitnessA = evaluatePlan(beliefs, plans.get(a));
			double fitnessB = evaluatePlan(beliefs, plans.get(b));
			if(fitnessA >= fitnessB)
				a_prevails = true;
			else
				a_prevails = false;
		}
		replace(a_prevails ? b : a);
	}


	/// Performs several iterations of plan refinement
	void refinePlans(double[] beliefs) {

		// If we are still burning in, then the models are probably not even reliable enough to make refining plans worthwhile
		if(burnIn > 0)
			return;

		for(int i = 0; i < refinementIters; i++) {
			double d = rand.nextDouble();
			if(d < 0.65)
				mutate();
			else
				tournament(beliefs);
		}
	}


	/// Drops the first action in every plan
	void advanceTime() {
		for(int i = 0; i < plans.size(); i++) {
			Plan p = plans.get(i);
			if(p.steps.size() > 0)
			{
				// Move the first action vector in each plan to the end
				double[] tmp = p.steps.get(0);
				p.steps.remove(0);
				p.steps.add(tmp);
			}
		}
	}


	/// Asks the mentor to evaluate the plan, given our current beliefs, and learn from it
	void askMentorToEvaluatePlan(double[] beliefs, Plan plan) {
		double[] anticipatedBeliefs = transitionModel.getFinalBeliefs(beliefs, plan);
		double[] anticipatedObs = observationModel.beliefsToObservations(anticipatedBeliefs);
		double feedback = mentor.evaluate(anticipatedObs);
		if(feedback != IMentor.NO_FEEDBACK)
			contentmentModel.trainIncremental(anticipatedBeliefs, feedback);
	}


	/// Finds the best plan and copies its first step
	void chooseNextActions(double[] beliefs, double[] actions) {

		if(tutor != null) {
			tutor.chooseActions(beliefs, actions);
			return;
		}

		// Find the best plan (according to the contentment model) and ask the mentor to evaluate it
		int planBestIndex = 0;
		double bestContentment = -Double.MAX_VALUE;
		for(int i = 0; i < plans.size(); i++) {
			double d = evaluatePlan(beliefs, plans.get(i));
			if(d > bestContentment) {
				bestContentment = d;
				planBestIndex = i;
			}
		}
		//System.out.println("Best contentment: " + Double.toString(bestContentment));
		Plan bestPlan = plans.get(planBestIndex);
		askMentorToEvaluatePlan(beliefs, bestPlan);

		// Pick a random plan from the population and ask the mentor to evaluate it (for contrast)
		int planBindex = rand.nextInt(plans.size() - 1);
		if(planBindex >= planBestIndex)
			planBindex++;
		askMentorToEvaluatePlan(beliefs, plans.get(planBindex));

		// Make a random one-step plan, and ask the mentor to evaluate it (for contrast)
		double[] action = randomPlan.steps.get(0);
		for(int i = 0; i < action.length; i++)
			action[i] = rand.nextDouble();
		askMentorToEvaluatePlan(beliefs, randomPlan);

		// Copy the first action vector of the best plan for our chosen action
		double[] bestActions = bestPlan.getActions(0);
		if(burnIn > 0 || rand.nextDouble() < explorationRate)
			bestActions = randomPlan.getActions(0);
		burnIn = Math.max(0, burnIn - 1);
		for(int i = 0; i < bestActions.length; i++) {
			actions[i] = bestActions[i];
		}
	}
}
