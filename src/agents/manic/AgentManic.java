package agents.manic;

import java.util.Random;
import java.util.ArrayList;
import java.util.Iterator;
import java.awt.image.BufferedImage;
import java.awt.Color;
import java.awt.Graphics2D;
import java.io.File;
import javax.imageio.ImageIO;
import common.IAgent;
import common.ITeacher;
import common.json.JSONObject;
import common.json.JSONArray;
import common.Matrix;
import common.Vec;


/// A model that maps from current beliefs and actions to anticipated beliefs.
/// This model is trained in a supervised manner.
class TransitionModel {
	Random rand;
	NeuralNet model;
	Matrix trainInput;
	Matrix trainOutput;
	int trainPos;
	int trainSize;
	int trainIters;
	int trainProgress;
	double learningRate;
	double err;
	double prevErr;


	/// General-purpose constructor
	TransitionModel(int input_dims, int output_dims, int total_layers, int queue_size, int trainItersPerPattern, Random r) {

		// Init the model
		rand = r;
		model = new NeuralNet();
		int hidden = Math.max(30, output_dims);
		model.layers.add(new LayerTanh(input_dims, hidden));
		model.layers.add(new LayerTanh(hidden, output_dims));
		model.init(rand);

		// Init the buffers
		trainInput = new Matrix(queue_size, input_dims);
		trainOutput = new Matrix(queue_size, output_dims);

		// Init the meta-parameters
		trainIters = trainItersPerPattern;
		learningRate = 0.03;
	}


	/// Unmarshaling constructor
	TransitionModel(JSONObject obj, Random r) {
		rand = r;
		model = new NeuralNet((JSONObject)obj.get("model"));
		trainInput = new Matrix((JSONObject)obj.get("trainInput"));
		trainOutput = new Matrix((JSONObject)obj.get("trainOutput"));
		trainPos = ((Long)obj.get("trainPos")).intValue();
		trainSize = ((Long)obj.get("trainSize")).intValue();
		trainIters = ((Long)obj.get("trainIters")).intValue();
		trainProgress = ((Long)obj.get("trainProgress")).intValue();
		learningRate = (Double)obj.get("learningRate");
		err = (Double)obj.get("err");
		prevErr = (Double)obj.get("prevErr");
	}


	/// Marshals this model to a JSON DOM.
	JSONObject marshal() {
		JSONObject obj = new JSONObject();
		obj.put("model", model.marshal());
		obj.put("trainInput", trainInput.marshal());
		obj.put("trainOutput", trainOutput.marshal());
		obj.put("trainPos", trainPos);
		obj.put("trainSize", trainSize);
		obj.put("trainIters", trainIters);
		obj.put("trainProgress", trainProgress);
		obj.put("learningRate", learningRate);
		obj.put("err", err);
		obj.put("prevErr", prevErr);
		return obj;
	}


	/// Returns the number of action dims
	int actionDims() { return model.layers.get(0).inputCount() - model.layers.get(model.layers.size() - 1).outputCount(); }


	/// Performs one pattern-presentation of stochastic gradient descent, and dynamically tunes the learning rate
	void doSomeTraining() {

		// Present one pattern
		model.regularize(learningRate, 0.0000001);
		int index = rand.nextInt(trainSize);
		model.trainIncremental(trainInput.row(index), trainOutput.row(index), learningRate);
		err += Vec.squaredDistance(model.layers.get(model.layers.size() - 1).activation, trainOutput.row(index));

		// Measure how we are doing
		trainProgress++;
		if(trainProgress >= trainInput.rows()) {
			trainProgress = 0;
			prevErr = Math.sqrt(err / trainInput.rows());
		}
	}


	/// Refines this model based on a recently performed action and change in beliefs
	void trainIncremental(double[] beliefs, double[] actions, double[] nextBeliefs) {

		// Buffer the pattern
		double[] destIn = trainInput.row(trainPos);
		double[] destOut = trainOutput.row(trainPos);
		trainPos++;
		trainSize = Math.max(trainSize, trainPos);
		if(trainPos >= trainInput.rows())
			trainPos = 0;
		if(beliefs.length + actions.length != destIn.length)
			throw new IllegalArgumentException("size mismatch");
		for(int i = 0; i < beliefs.length; i++)
			destIn[i] = beliefs[i];
		for(int i = 0; i < actions.length; i++)
			destIn[beliefs.length + i] = actions[i];
		for(int i = 0; i < destOut.length; i++)
			destOut[i] = nextBeliefs[i] - beliefs[i];

		// Refine the model
		int iters = Math.min(trainIters, 100 * trainSize);
		for(int i = 0; i < iters; i++)
			doSomeTraining();
	}


	/// Predict the belief vector that will result if the specified action is performed
	void anticipateNextBeliefsInPlace(double[] beliefs, double[] actions, double[] anticipatedBeliefs) {
		double[] pred = model.forwardProp2(beliefs, actions);
		for(int i = 0; i < pred.length; i++) {
			anticipatedBeliefs[i] = Math.max(-1.0, Math.min(1.0, beliefs[i] + pred[i]));
		}
	}


	/// Predict the belief vector that will result if the specified action is performed
	double[] anticipateNextBeliefs(double[] beliefs, double[] actions) {
		double[] anticipatedBeliefs = new double[beliefs.length];
		anticipateNextBeliefsInPlace(beliefs, actions, anticipatedBeliefs);
		return anticipatedBeliefs;
	}


	/// Compute the anticipated belief vector that will result if the specified plan is executed.
	double[] getFinalBeliefs(double[] beliefs, Plan plan) {
		for(int i = 0; i < plan.size(); i++) {
			beliefs = anticipateNextBeliefs(beliefs, plan.getActions(i));
		}
		return beliefs;
	}
}


/// A bidirectional model that maps between beliefs and observations.
/// Mapping from observations to beliefs is done by the encoder.
/// Mapping from beliefs to observations is done by the decoder.
/// These two components are trained together in an unsupervised manner as an autoencoder.
class ObservationModel {
	Random rand;
	NeuralNet decoder;
	NeuralNet encoder;
	NeuralNet decoderExperimental;
	NeuralNet encoderExperimental;
	Matrix train;
	Matrix validation;
	TransitionModel transitionModel;
	int trainPos;
	int trainSize;
	int validationPos;
	int validationSize;
	int trainIters;
	int trainProgress;
	int calibrationIters;
	double learningRate;


	/// General-purpose constructor
	ObservationModel(TransitionModel transition, int observation_dims, int belief_dims, int decoder_layers,
		int encoder_layers, int queue_size, int trainItersPerPattern, int calibrationIterations, Random r) {

		if(belief_dims > observation_dims)
			throw new IllegalArgumentException("observation_dims must be >= belief_dims");

		// Init the encoder
		rand = r;
		int hidden = Math.max(30, (observation_dims + belief_dims) / 2);
		encoder = new NeuralNet();
		encoder.layers.add(new LayerTanh(observation_dims, hidden));
		encoder.layers.add(new LayerTanh(hidden, belief_dims));
		encoder.init(rand);

		// Init the decoder
		decoder = new NeuralNet();
		decoder.layers.add(new LayerTanh(belief_dims, hidden));
		decoder.layers.add(new LayerTanh(hidden, observation_dims));
		decoder.init(rand);

		// Make the experimental nets
		decoderExperimental = new NeuralNet(decoder);
		encoderExperimental = new NeuralNet(encoder);

		// Init the buffers
		train = new Matrix(queue_size, observation_dims);
		validation = new Matrix(queue_size, observation_dims);

		// Init the meta-parameters
		transitionModel = transition;
		trainIters = trainItersPerPattern;
		calibrationIters = calibrationIterations;
		learningRate = 0.03;
	}


	/// Unmarshaling constructor
	ObservationModel(TransitionModel transition, JSONObject obj, Random r) {
		rand = r;
		decoder = new NeuralNet((JSONObject)obj.get("decoder"));
		encoder = new NeuralNet((JSONObject)obj.get("encoder"));
		decoderExperimental = new NeuralNet((JSONObject)obj.get("decoderExperimental"));
		encoderExperimental = new NeuralNet((JSONObject)obj.get("encoderExperimental"));
		train = new Matrix((JSONObject)obj.get("train"));
		validation = new Matrix((JSONObject)obj.get("validation"));
		trainPos = ((Long)obj.get("trainPos")).intValue();
		trainSize = ((Long)obj.get("trainSize")).intValue();
		validationPos = ((Long)obj.get("validationPos")).intValue();
		validationSize = ((Long)obj.get("validationSize")).intValue();
		trainIters = ((Long)obj.get("trainIters")).intValue();
		trainProgress = ((Long)obj.get("trainProgress")).intValue();
		calibrationIters = ((Long)obj.get("calibrationIters")).intValue();
		learningRate = (Double)obj.get("learningRate");
		transitionModel = transition;
	}


	/// Marshals this model to a JSON DOM.
	JSONObject marshal() {
		JSONObject obj = new JSONObject();
		obj.put("decoder", decoder.marshal());
		obj.put("encoder", encoder.marshal());
		obj.put("decoderExperimental", decoderExperimental.marshal());
		obj.put("encoderExperimental", encoderExperimental.marshal());
		obj.put("train", train.marshal());
		obj.put("validation", validation.marshal());
		obj.put("trainPos", trainPos);
		obj.put("trainSize", trainSize);
		obj.put("validationPos", validationPos);
		obj.put("validationSize", validationSize);
		obj.put("trainIters", trainIters);
		obj.put("trainProgress", trainProgress);
		obj.put("calibrationIters", calibrationIters);
		obj.put("learningRate", learningRate);
		return obj;
	}


	/// Performs one pattern-presentation of stochastic gradient descent and dynamically tunes the learning rate
	void doSomeTraining() {

		// Train the decoderExperimental and encoderExperimental together as an autoencoder
		decoderExperimental.regularize(learningRate, 0.00001);
		encoderExperimental.regularize(learningRate, 0.00001);
		int index = rand.nextInt(trainSize);
		double[] observation = train.row(index);
		double[] belief = encoderExperimental.forwardProp(observation);
		double[] prediction = decoderExperimental.forwardProp(belief);
		decoderExperimental.backProp(observation);
		encoderExperimental.backPropFromDecoder(decoderExperimental);
		encoderExperimental.descendGradient(observation, learningRate);
		decoderExperimental.descendGradient(belief, learningRate);

		// Since changing the observation function resets the training data for the transition function,
		// we only want to change our perception when it will lead to big improvements.
		// Here, we test whether our experimental model is significantly better than the one we have been using.
		// If so, then the experimental model becomes the new model.
		trainProgress++;
		if(trainProgress >= train.rows()) {
			// Measure mean squared error
			trainProgress = 0;
			double err1 = 0.0;
			double err2 = 0.0;
			for(int i = 0; i < validationSize; i++) {
				double[] targ = validation.row(i);
				double[] pred1 = decoder.forwardProp(encoder.forwardProp(targ));
				double[] pred2 = decoderExperimental.forwardProp(encoderExperimental.forwardProp(targ));
				for(int j = 0; j < targ.length; j++) {
					err1 += (targ[j] - pred1[j]) * (targ[j] - pred1[j]);
					err2 += (targ[j] - pred2[j]) * (targ[j] - pred2[j]);
				}
			}
			err1 = Math.sqrt(err1 / validationSize);
			err2 = Math.sqrt(err2 / validationSize);
			if(err2 < 0.85 * err1) {
				// Update the observation model and reset the training data for the transition function
				encoder.copy(encoderExperimental);
				decoder.copy(decoderExperimental);
				transitionModel.trainPos = 0;
				transitionModel.trainSize = 0;
			}
			else if(err1 < 0.85 * err2) {
				// This should really never happen
				encoderExperimental.copy(encoder);
				decoderExperimental.copy(decoder);
			}
		}
	}


	/// Refines the encoder and decoder based on the new observation.
	void trainIncremental(double[] observation) {

		// Buffer the pattern
		double[] dest;
		if(validationPos < trainPos) {
			dest = validation.row(validationPos);
			if(++validationPos >= validation.rows())
				validationPos = 0;
			validationSize = Math.max(validationSize, validationPos);
		} else {
			dest = train.row(trainPos);
			trainPos++;
			trainSize = Math.max(trainSize, trainPos);
			if(trainPos >= train.rows())
				trainPos = 0;
		}
		for(int i = 0; i < dest.length; i++)
			dest[i] = observation[i];

		// Train
		int iters = Math.min(trainIters, trainSize);
		for(int i = 0; i < iters; i++)
			doSomeTraining();
	}


	/// Refines the beliefs to correspond with actual observations
	void calibrateBeliefs(double[] beliefs, double[] observations) {
		for(int i = 0; i < calibrationIters; i++) {
			decoder.refineInputs(beliefs, observations, learningRate);
			for(int j = 0; j < beliefs.length; j++)
				beliefs[j] = Math.max(-1.0, Math.min(1.0, beliefs[j]));
		}
	}


	/// Decodes beliefs to predict observations
	double[] beliefsToObservations(double[] beliefs) {
		double[] obs = decoder.forwardProp(beliefs);
		double[] ret = new double[obs.length];
		for(int i = 0; i < obs.length; i++) {
			ret[i] = obs[i];
		}

		return ret;
	}


	/// Encodes observations to predict beliefs
	double[] observationsToBeliefs(double[] observations) {
		double[] bel = encoder.forwardProp(observations);
		double[] ret = new double[bel.length];
		for(int i = 0; i < bel.length; i++) {
			ret[i] = bel[i];
		}

		return ret;
	}
}


/// A model that maps from anticipated beliefs to contentment (or utility).
/// This model is trained by reinforcement from a teacher.
class ContentmentModel {
	Random rand;
	NeuralNet model;
	Matrix samples;
	Matrix contentment;
	int trainPos;
	int trainSize;
	int trainIters;
	double learningRate;
	double[] targBuf;


	// General-purpose constructor
	ContentmentModel(int beliefDims, int total_layers, int queue_size, int trainItersPerPattern, Random r) {

		// Init the model
		rand = r;
		model = new NeuralNet();
		int hidden = Math.min(30, beliefDims * 10);
		model.layers.add(new LayerTanh(beliefDims, hidden));
		model.layers.add(new LayerTanh(hidden, 1));
		model.init(rand);

		// Init the buffers
		samples = new Matrix(queue_size, beliefDims);
		contentment = new Matrix(queue_size, 1);

		// Init the meta-parameters
		trainIters = trainItersPerPattern;
		learningRate = 0.03;
		targBuf = new double[1];
	}


	/// Unmarshaling constructor
	ContentmentModel(JSONObject obj, Random r) {
		rand = r;
		model = new NeuralNet((JSONObject)obj.get("model"));
		samples = new Matrix((JSONObject)obj.get("samples"));
		contentment = new Matrix((JSONObject)obj.get("contentment"));
		trainPos = ((Long)obj.get("trainPos")).intValue();
		trainSize = ((Long)obj.get("trainSize")).intValue();
		trainIters = ((Long)obj.get("trainIters")).intValue();
		learningRate = (Double)obj.get("learningRate");
		targBuf = new double[1];
	}


	/// Marshals this model to a JSON DOM.
	JSONObject marshal() {
		JSONObject obj = new JSONObject();
		obj.put("model", model.marshal());
		obj.put("samples", samples.marshal());
		obj.put("contentment", contentment.marshal());
		obj.put("trainPos", trainPos);
		obj.put("trainSize", trainSize);
		obj.put("trainIters", trainIters);
		obj.put("learningRate", learningRate);
		return obj;
	}


	/// Performs one pattern-presentation of stochastic gradient descent, and dynamically tunes the learning rate
	void doSomeTraining() {

		// Present a sample of beliefs and corresponding contentment for training
		int index = rand.nextInt(trainSize);
		model.regularize(learningRate, 0.000001);
		model.trainIncremental(samples.row(index), contentment.row(index), learningRate);
	}


	/// Refines this model based on feedback from the teacher
	void trainIncremental(double[] sample_beliefs, double sample_contentment) {

		// Buffer the samples
		double[] dest = samples.row(trainPos);
		if(sample_beliefs.length != dest.length)
			throw new IllegalArgumentException("size mismatch");
		for(int i = 0; i < dest.length; i++)
			dest[i] = sample_beliefs[i];
		contentment.row(trainPos)[0] = sample_contentment;
		trainPos++;
		trainSize = Math.max(trainSize, trainPos);
		if(trainPos >= samples.rows())
			trainPos = 0;

		// Do a few iterations of stochastic gradient descent
		int iters = Math.min(trainIters, trainSize);
		for(int i = 0; i < iters; i++)
			doSomeTraining();
	}


	/// Computes the contentment of a particular belief vector
	double evaluate(double[] beliefs) {
		double[] output = model.forwardProp(beliefs);
		return output[0];
	}
}


/// Represents a sequence of action vectors.
class Plan {
	ArrayList<double[]> steps;


	// General-purpose constructor
	Plan() {
		steps = new ArrayList<double[]>();
	}

	// Copy constructor
	Plan(Plan that) {
		steps = new ArrayList<double[]>();
		for(int i = 0; i < that.size(); i++) {
			steps.add(Vec.copy(that.getActions(i)));
		}
	}

	/// Unmarshaling constructor
	Plan(JSONArray stepsArr) {
		steps = new ArrayList<double[]>();
		Iterator<JSONArray> it = stepsArr.iterator();
		while(it.hasNext()) {
			steps.add(Vec.unmarshal(it.next()));
		}
	}

	/// Marshals this model to a JSON DOM.
	JSONArray marshal() {
		JSONArray stepsArr = new JSONArray();
		for(int i = 0; i < steps.size(); i++) {
			stepsArr.add(Vec.marshal(steps.get(i)));
		}
		return stepsArr;
	}

	/// Returns the number of steps (or action vectors) in this plan
	int size() { return steps.size(); }

	/// Returns the ith action vector in this plan
	double[] getActions(int i) { return steps.get(i); }

	/// Prints a representation of the plan to stdout
	void print() {
		System.out.print("[");
		for(int i = 0; i < steps.size(); i++) {
			double[] actions = steps.get(i);
			System.out.print("(");
			for(int j = 0; j < actions.length; j++) {
				if(j > 0)
					System.out.print(",");
				System.out.print(Double.toString(actions[j]));
			}
			System.out.print(")");
		}
		System.out.println("]");
	}
}


/// A genetic algorithm that sequences actions to form a plan intended to maximize contentment.
class PlanningSystem {
	Plan randomPlan;
	ArrayList<Plan> plans;
	TransitionModel transitionModel;
	ObservationModel observationModel;
	ContentmentModel contentmentModel;
	ITeacher teacher;
	int maxPlanLength;
	int refinementIters;
	int actionDims;
	int burnIn;
	double discountFactor;
	double explorationRate;
	Random rand;


	// General-purpose constructor
	PlanningSystem(TransitionModel transition, ObservationModel observation, ContentmentModel contentment, ITeacher oracle,
		int actionDimensions, int populationSize, int planRefinementIters, int burnInIters, int maxPlanLen, double discount, double explore, Random r) {
		transitionModel = transition;
		observationModel = observation;
		contentmentModel = contentment;
		teacher = oracle;
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
	PlanningSystem(JSONObject obj, Random r, TransitionModel transition, ObservationModel observation, ContentmentModel contentment, ITeacher oracle) {
		transitionModel = transition;
		observationModel = observation;
		contentmentModel = contentment;
		teacher = oracle;
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


	/// Replaces the teacher with the specified one
	void setTeacher(ITeacher oracle) {
		teacher = oracle;
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


	/// Asks the teacher to evaluate the plan, given our current beliefs, and learn from it
	void askTeacherToEvaluatePlan(double[] beliefs, Plan plan) {
		double[] anticipatedBeliefs = transitionModel.getFinalBeliefs(beliefs, plan);
		double[] anticipatedObs = observationModel.beliefsToObservations(anticipatedBeliefs);
		double feedback = teacher.evaluate(anticipatedObs);
		contentmentModel.trainIncremental(anticipatedBeliefs, feedback);
	}


	/// Finds the best plan and copies its first step
	void chooseNextActions(double[] beliefs, double[] actions) {

		// Find the best plan (according to the contentment model) and ask the teacher to evaluate it
		int planBestIndex = 0;
		double bestContentment = -Double.MAX_VALUE;
		for(int i = 0; i < plans.size(); i++) {
			double d = evaluatePlan(beliefs, plans.get(i));
			if(d > bestContentment) {
				bestContentment = d;
				planBestIndex = i;
			}
		}
		Plan bestPlan = plans.get(planBestIndex);
		askTeacherToEvaluatePlan(beliefs, bestPlan);

		// Pick a random plan from the population and ask the teacher to evaluate it (for contrast)
		int planBindex = rand.nextInt(plans.size() - 1);
		if(planBindex >= planBestIndex)
			planBindex++;
		askTeacherToEvaluatePlan(beliefs, plans.get(planBindex));

		// Make a random one-step plan, and ask the teacher to evaluate it (for contrast)
		double[] action = randomPlan.steps.get(0);
		for(int i = 0; i < action.length; i++)
			action[i] = rand.nextDouble();
		askTeacherToEvaluatePlan(beliefs, randomPlan);

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


/// Implements a weak artificial general intelligence.
public class AgentManic implements IAgent {
	Random rand;
	TransitionModel transitionModel;
	ObservationModel observationModel;
	ContentmentModel contentmentModel;
	PlanningSystem planningSystem;
	double[] actions;
	double[] beliefs;
	double[] anticipatedBeliefs;


	// General-purpose constructor.
	public AgentManic(Random r) {
		rand = r;
	}

	// This method is called to initialize the agent in a new world.
	// oracle is an object that helps the agent learn what to do in this world.
	// observationDims is the number of double values that the agent observes each time step.
	// beliefDims is the number of double values that the agent uses internally to model the state of the world. (It should generally be <= observationDims.)
	// actionDims is the number of double values the agent uses to specify an action.
	// maxPlanLength specifies the maximum number of time-steps into the future that the agent should attempt to plan.
	public void reset(ITeacher oracle, int observationDims, int beliefDims, int actionDims, int maxPlanLength) {
		transitionModel = new TransitionModel(
			actionDims + beliefDims,
			beliefDims,
			2, // number of layers in the transition model
			500, // size of short term memory for transitions
			100, // number of training iterations to perform with each new sample
			rand);
		observationModel = new ObservationModel(
			transitionModel,
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
			500, // size of short term memory for feedback from the teacher
			50, // number of training iterations to perform with each new sample
			rand);
		planningSystem = new PlanningSystem(
			transitionModel,
			observationModel,
			contentmentModel,
			oracle,
			actionDims,
			30, // population size
			50, // number of iterations to refine each member of the population per time step
			500, // burn-in iterations (the number of times at the start to just pick a random action, so the transition function has a chance to explore its space)
			maxPlanLength,
			0.99, // discount factor (to make short plans be preferred over long plans that ultimately arrive at nearly the same state)
			0.0, // exploration rate (the probability that the agent will choose a random action, just to see what happens)
			rand);
		actions = new double[actionDims];
		beliefs = new double[beliefDims];
		anticipatedBeliefs = new double[beliefDims];
	}


	/// Unmarshaling constructor
	public AgentManic(JSONObject obj, Random r, ITeacher oracle) {
		rand = r;
		transitionModel = new TransitionModel((JSONObject)obj.get("transition"), r);
		observationModel = new ObservationModel(transitionModel, (JSONObject)obj.get("observation"), r);
		contentmentModel = new ContentmentModel((JSONObject)obj.get("contentment"), r);
		planningSystem = new PlanningSystem((JSONObject)obj.get("planning"), r, transitionModel, observationModel, contentmentModel, oracle);
		actions = new double[transitionModel.actionDims()];
		beliefs = Vec.unmarshal((JSONArray)obj.get("beliefs"));
		anticipatedBeliefs = new double[beliefs.length];
	}


	/// Marshals this agent to a JSON DOM.
	public JSONObject marshal() {
		JSONObject obj = new JSONObject();
		obj.put("transition", transitionModel.marshal());
		obj.put("observation", observationModel.marshal());
		obj.put("contentment", contentmentModel.marshal());
		obj.put("planning", planningSystem.marshal());
		obj.put("beliefs", Vec.marshal(beliefs));
		return obj;
	}


	/// Replaces the teacher with the specified one
	public void setTeacher(ITeacher oracle) {
		planningSystem.setTeacher(oracle);
	}


	/// Learns from observations
	void learnFromExperience(double[] observations) {

		// Learn to perceive the world a little better
		observationModel.trainIncremental(observations);

		// Refine beliefs to correspond with the new observations better
		observationModel.calibrateBeliefs(anticipatedBeliefs, observations);

		// Learn to anticipate consequences a little better
		transitionModel.trainIncremental(beliefs, actions, anticipatedBeliefs);
	}


	/// Returns an action vector
	double[] decideWhatToDo() {

		// Make the anticipated beliefs the new beliefs
		double[] tmp = beliefs;
		beliefs = anticipatedBeliefs;
		anticipatedBeliefs = tmp;

		// Drop the first action in every plan
		planningSystem.advanceTime();

		// Try to make the plans better
		planningSystem.refinePlans(beliefs);

		// Choose an action that is expected to maximize contentment (with the assistance of the teacher, if available)
		planningSystem.chooseNextActions(beliefs, actions);

		// Anticipate how the world will change with time
		transitionModel.anticipateNextBeliefsInPlace(beliefs, actions, anticipatedBeliefs);

		// Return the selected actions
		return actions;
	}


	/// A vector of observations goes in. All observed values may be expected to fall between -1 and 1.
	/// Returns a vector of chosen actions. All returned values should fall between 0 and 1.
	public double[] think(double[] observations) {

		// Check the observations
		for(int i = 0; i < observations.length; i++) {
			if(observations[i] < -1.0 || observations[i] > 1.0)
				throw new IllegalArgumentException("Observed values must be between -1 and 1.");
		}

		learnFromExperience(observations);
		return decideWhatToDo();
	}

/*
	public static void testMarshaling() throws Exception {
		// Make an agent
		AgentManic agent = new AgentManic(
			new Random(1234),
			new MyTeacher(),
			8, // observation dims
			3, // belief dims
			2, // action dims
			10); // max plan length

		// Write it to a file
		JSONObject obj = agent.marshal();
		FileWriter file = new FileWriter("test.json");
		file.write(obj.toJSONString());
		file.close();

		// Read it from a file
		JSONParser parser = new JSONParser();
		JSONObject obj2 = (JSONObject)parser.parse(new FileReader("test.json"));
		AgentManic agent2 = new AgentManic(obj2, new Random(1234), new MyTeacher());

		System.out.println("passed");
	}
*/

	/// Generates an image to visualize the space of the contentment model
	public BufferedImage visualizeSpace() {
		if(beliefs.length != 2)
			throw new IllegalArgumentException("Sorry, this method only works with 2D belief spaces");
	
		// Find the min and max locations
		double[] in = new double[2];
		double mi = Double.MAX_VALUE;
		double ma = -Double.MAX_VALUE;
		double[] min_loc = new double[2];
		double[] max_loc = new double[2];
		for(int y = 0; y < 1000; y++) {
			for(int x = 0; x < 1000; x++) {
				in[0] = ((double)x) / 1000.0 * 2.0 - 1.0;
				in[1] = ((double)y) / 1000.0 * 2.0 - 1.0;
				double out = contentmentModel.evaluate(observationModel.observationsToBeliefs(in));
				if(out < mi) {
					mi = out;
					min_loc[0] = in[0];
					min_loc[1] = in[1];
				}
				if(out > ma) {
					ma = out;
					max_loc[0] = in[0];
					max_loc[1] = in[1];
				}
			}
		}

		// Draw the contours of the contentment function
		BufferedImage image = new BufferedImage(1000, 1000, BufferedImage.TYPE_INT_ARGB);
		for(int y = 0; y < 1000; y++) {
			for(int x = 0; x < 1000; x++) {
				in[0] = ((double)x) * 0.002 - 1.0;
				in[1] = ((double)y) * 0.002 - 1.0;
				double out = (contentmentModel.evaluate(observationModel.observationsToBeliefs(in)) - mi) * 256.0 / (ma - mi);
				int g = Math.max(0, Math.min(255, (int)out));
				int gg = g;
				if(g % 5 == 0)
					gg = (128 - (int)(Math.tanh((double)(g - 128) * 0.03) * 127.0));
				image.setRGB(x, y, new Color(g, g, gg).getRGB());
			}
		}

		// Draw magenta dots at the sample locations
		Graphics2D g = image.createGraphics();
		g.setColor(new Color(255, 0, 255));
		for(int i = 0; i < contentmentModel.trainSize; i++) {
			double[] r = observationModel.beliefsToObservations(contentmentModel.samples.row(i));
			int x = (int)((r[0] + 1.0) * 500.0);
			int y = (int)((r[1] + 1.0) * 500.0);
			g.fillOval(x - 2, y - 2, 4, 4);
		}

		// Draw the circle of transitions
		double[] tmp_act = new double[1];
		for(double d = 0; d <= 1.0; d += 0.03125) {
			if(d == 0)
				g.setColor(new Color(255, 0, 0));
			else if(d == 0.25)
				g.setColor(new Color(255, 255, 0));
			else if(d == 0.5)
				g.setColor(new Color(0, 255, 0));
			else if(d == 0.75)
				g.setColor(new Color(0, 255, 255));
			tmp_act[0] = d;
			double[] next = transitionModel.anticipateNextBeliefs(beliefs, tmp_act);
			double[] next_obs = observationModel.beliefsToObservations(next);
			g.fillOval((int)((next_obs[0] + 1.0) * 500.0) - 4, (int)((next_obs[1] + 1.0) * 500.0) - 4, 8, 8);
		}

		// Draw the beliefs
		g.setColor(new Color(255, 128, 0));
		double[] exp_obs = observationModel.beliefsToObservations(beliefs);
		g.fillOval((int)((exp_obs[0] + 1.0) * 500.0) - 4, (int)((exp_obs[1] + 1.0) * 500.0) - 4, 8, 8);

		// Draw the plans
		for(int i = 0; i < planningSystem.plans.size(); i++) {
			Plan plan = planningSystem.plans.get(i);
			double[] prev = beliefs;
			double[] prev_obs = exp_obs;
			for(int j = 0; j < plan.steps.size(); j++) {
				double[] next = transitionModel.anticipateNextBeliefs(prev, plan.steps.get(j));
				double[] next_obs = observationModel.beliefsToObservations(next);
				g.drawLine((int)((prev_obs[0] + 1.0) * 500.0), (int)((prev_obs[1] + 1.0) * 500.0), (int)((next_obs[0] + 1.0) * 500.0), (int)((next_obs[1] + 1.0) * 500.0));
				prev = next;
				prev_obs = next_obs;
			}
		}

		// Draw the chosen action
		g.setColor(new Color(0, 128, 0));
		double[] ant_obs = observationModel.beliefsToObservations(anticipatedBeliefs);
		g.drawLine((int)((exp_obs[0] + 1.0) * 500.0), (int)((exp_obs[1] + 1.0) * 500.0), (int)((ant_obs[0] + 1.0) * 500.0), (int)((ant_obs[1] + 1.0) * 500.0));

		return image;
	}
}
