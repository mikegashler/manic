import java.util.Random;
import java.util.ArrayList;
import java.util.Iterator;


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
	double learningRate;
	double prevErr;


	/// General-purpose constructor
	TransitionModel(int input_dims, int output_dims, int total_layers, int queue_size, int trainItersPerPattern, Random r) {

		// Init the model
		rand = r;
		model = new NeuralNet();
		for(int i = 0; i < total_layers; i++) {
			int in = ((input_dims * (total_layers - i)) + (output_dims * i)) / (total_layers);
			int j = i + 1;
			int out = ((input_dims * (total_layers - j)) + (output_dims * j)) / (total_layers);
			model.layers.add(new Layer(in, out));
		}
		model.init(rand);

		// Init the buffers
		trainInput = new Matrix(queue_size, input_dims);
		trainOutput = new Matrix(queue_size, output_dims);

		// Init the meta-parameters
		trainIters = trainItersPerPattern;
		learningRate = 0.01;
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
		learningRate = (Double)obj.get("learningRate");
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
		obj.put("learningRate", learningRate);
		obj.put("prevErr", prevErr);
		return obj;
	}


	/// Returns the number of action dims
	int actionDims() { return model.layers.get(0).inputCount() - model.layers.get(model.layers.size() - 1).outputCount(); }


	/// Performs one pattern-presentation of stochastic gradient descent, and dynamically tunes the learning rate
	void doSomeTraining() {

		// Present one pattern
		model.regularize(learningRate, 0.0001);
		int index = rand.nextInt(trainSize);
		model.trainIncremental(trainInput.row(index), trainOutput.row(index), learningRate);
	}


	/// Refines this model based on a recently performed action and change in beliefs
	void trainIncremental(double[] beliefs, double[] actions, double[] nextBeliefs) {

		// Buffer the pattern
		double[] destIn = trainInput.row(trainPos);
		double[] destOut = trainOutput.row(trainPos);
		if(++trainPos >= trainInput.rows())
			trainPos = 0;
		trainSize = Math.max(trainSize, trainPos);
		if(beliefs.length + actions.length != destIn.length)
			throw new IllegalArgumentException("size mismatch");
		for(int i = 0; i < beliefs.length; i++)
			destIn[i] = beliefs[i];
		for(int i = 0; i < actions.length; i++)
			destIn[beliefs.length + i] = actions[i];
		for(int i = 0; i < destOut.length; i++)
			destOut[i] = nextBeliefs[i];

		// Refine the model
		int iters = Math.min(trainIters, trainSize);
		for(int i = 0; i < iters; i++)
			doSomeTraining();
	}


	/// Predict the belief vector that will result if the specified action is performed
	void anticipateNextBeliefsInPlace(double[] beliefs, double[] actions, double[] anticipatedBeliefs) {
		double[] pred = model.forwardProp2(beliefs, actions);
		for(int i = 0; i < pred.length; i++) {
			anticipatedBeliefs[i] = pred[i];
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
	int trainPos;
	int trainSize;
	int validationPos;
	int validationSize;
	int trainIters;
	int trainProgress;
	int calibrationIters;
	double learningRate;


	/// General-purpose constructor
	ObservationModel(int observation_dims, int belief_dims, int decoder_layers, int encoder_layers, int queue_size, int trainItersPerPattern, int calibrationIterations, Random r) {
		// Init the decoder
		rand = r;
		decoder = new NeuralNet();
		for(int i = 0; i < decoder_layers; i++) {
			int in = ((belief_dims * (decoder_layers - i)) + (observation_dims * i)) / (decoder_layers);
			int j = i + 1;
			int out = ((belief_dims * (decoder_layers - j)) + (observation_dims * j)) / (decoder_layers);
			Layer l = new Layer(in, out);
			decoder.layers.add(l);
		}
		decoder.init(rand);

		// Init the encoder
		encoder = new NeuralNet();
		for(int i = 0; i < encoder_layers; i++) {
			int in = ((observation_dims * (encoder_layers - i)) + (belief_dims * i)) / (encoder_layers);
			int j = i + 1;
			int out = ((observation_dims * (encoder_layers - j)) + (belief_dims * j)) / (encoder_layers);
			Layer l = new Layer(in, out);
			encoder.layers.add(l);
		}
		encoder.init(rand);

		// Make the experimental nets
		decoderExperimental = new NeuralNet(decoder);
		encoderExperimental = new NeuralNet(encoder);

		// Init the buffers
		train = new Matrix(queue_size, observation_dims);
		validation = new Matrix(queue_size, observation_dims);

		// Init the meta-parameters
		trainIters = trainItersPerPattern;
		calibrationIters = calibrationIterations;
		learningRate = 0.01;
	}


	/// Unmarshaling constructor
	ObservationModel(JSONObject obj, Random r) {
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
		decoderExperimental.regularize(learningRate, 0.0001);
		encoderExperimental.regularize(learningRate, 0.0001);
		int index = rand.nextInt(trainSize);
		double[] observation = train.row(index);
		double[] belief = encoderExperimental.forwardProp(observation);
		double[] prediction = decoderExperimental.forwardProp(belief);
		decoderExperimental.backPropAndBendHinge(observation, learningRate);
		encoderExperimental.backPropFromDecoder(decoder, learningRate);
		encoderExperimental.descendGradient(observation, learningRate);
		decoderExperimental.descendGradient(belief, learningRate);

		// Since changing how we perceive the world can throw off all the other models,
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
				encoder.copy(encoderExperimental);
				decoder.copy(decoderExperimental);
			}
			else if(err1 < 0.85 * err2) {
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
			if(++trainPos >= train.rows())
				trainPos = 0;
			trainSize = Math.max(trainSize, trainPos);
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
	Matrix better;
	Matrix worse;
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
		int fat_end = Math.min(30, beliefDims * 10);
		for(int i = 0; i < total_layers; i++) {
			int in = (i == 0 ? beliefDims : ((fat_end * (total_layers - i)) + i) / (total_layers));
			int j = i + 1;
			int out = ((fat_end * (total_layers - j)) + j) / (total_layers);
			model.layers.add(new Layer(in, out));
		}
		model.init(rand);

		// Init the buffers
		better = new Matrix(queue_size, beliefDims);
		worse = new Matrix(queue_size, beliefDims);

		// Init the meta-parameters
		trainIters = trainItersPerPattern;
		learningRate = 0.01;
		targBuf = new double[1];
	}


	/// Unmarshaling constructor
	ContentmentModel(JSONObject obj, Random r) {
		rand = r;
		model = new NeuralNet((JSONObject)obj.get("model"));
		better = new Matrix((JSONObject)obj.get("better"));
		worse = new Matrix((JSONObject)obj.get("worse"));
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
		obj.put("better", better.marshal());
		obj.put("worse", worse.marshal());
		obj.put("trainPos", trainPos);
		obj.put("trainSize", trainSize);
		obj.put("trainIters", trainIters);
		obj.put("learningRate", learningRate);
		return obj;
	}


	/// Performs one pattern-presentation of stochastic gradient descent, and dynamically tunes the learning rate
	void doSomeTraining() {

		// Present a single pair for rank-based training
		int index = rand.nextInt(trainSize);
		double bet = evaluate(better.row(index));
		double wor = evaluate(worse.row(index));
		if(wor >= bet) {
			model.regularize(learningRate, 0.000001);
			targBuf[0] = wor + 0.05;
			model.trainIncremental(better.row(index), targBuf, learningRate);
			targBuf[0] = bet - 0.05;
			model.trainIncremental(worse.row(index), targBuf, learningRate);
		}
	}


	/// Refines this model based on feedback from the teacher
	void trainIncremental(double[] bet, double[] wor) {

		// Buffer the samples
		double[] dest = better.row(trainPos);
		if(bet.length != dest.length)
			throw new IllegalArgumentException("size mismatch");
		for(int i = 0; i < bet.length; i++)
			dest[i] = bet[i];
		dest = worse.row(trainPos);
		if(wor.length != dest.length)
			throw new IllegalArgumentException("size mismatch");
		for(int i = 0; i < wor.length; i++)
			dest[i] = wor[i];
		trainPos++;
		trainSize = Math.max(trainSize, trainPos);
		if(trainPos >= better.rows())
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
			steps.add(Layer.copyArray(that.getActions(i)));
		}
	}

	/// Unmarshaling constructor
	Plan(JSONArray stepsArr) {
		steps = new ArrayList<double[]>();
		Iterator<JSONArray> it = stepsArr.iterator();
		while(it.hasNext()) {
			steps.add(Layer.unmarshalVector(it.next()));
		}
	}

	/// Marshals this model to a JSON DOM.
	JSONArray marshal() {
		JSONArray stepsArr = new JSONArray();
		for(int i = 0; i < steps.size(); i++) {
			stepsArr.add(Layer.marshalVector(steps.get(i)));
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
	ArrayList<Plan> plans;
	TransitionModel transitionModel;
	ObservationModel observationModel;
	ContentmentModel contentmentModel;
	ITeacher teacher;
	int maxPlanLength;
	int refinementIters;
	int actionDims;
	double discountFactor;
	Random rand;


	// General-purpose constructor
	PlanningSystem(TransitionModel transition, ObservationModel observation, ContentmentModel contentment, ITeacher oracle, int actionDimensions, int populationSize, int planRefinementIters, int maxPlanLen, double discount, Random r) {
		transitionModel = transition;
		observationModel = observation;
		contentmentModel = contentment;
		teacher = oracle;
		rand = r;
		plans = new ArrayList<Plan>();
		if(populationSize < 2)
			throw new IllegalArgumentException("The population size must be at least 2");
		actionDims = actionDimensions;
		maxPlanLength = maxPlanLen;
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
		refinementIters = ((Long)obj.get("refinementIters")).intValue();
		actionDims = ((Long)obj.get("actionDims")).intValue();
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
		obj.put("refinementIters", refinementIters);
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
		if(d < 0.2) { // lengthen the plan
			if(p.size() < maxPlanLength) {
				double[] newActions = new double[actionDims];
				for(int i = 0; i < actionDims; i++) {
					newActions[i] = rand.nextDouble();
				}
				p.steps.add(rand.nextInt(p.size() + 1), newActions);
			}
		}
		else if(d < 0.35) { // shorten the plan
			if(p.size() > 1) {
				p.steps.remove(rand.nextInt(p.size()));
			}
		}
		else if(d < 0.5) { // perturb a whole action vector
			double[] actions = p.getActions(rand.nextInt(p.size()));
			for(int i = 0; i < actions.length; i++) {
				actions[i] = Math.max(0.0, Math.min(1.0, actions[i] + 0.1 * rand.nextGaussian()));
			}
		}
		else if(d < 0.6) { // perturb the whole plan
			for(int j = 0; j < p.size(); j++) {
				double[] actions = p.getActions(j);
				for(int i = 0; i < actions.length; i++) {
					actions[i] = Math.max(0.0, Math.min(1.0, actions[i] + 0.1 * rand.nextGaussian()));
				}
			}
		}
		else { // perturb a single element of an action vector
			double[] actions = p.getActions(rand.nextInt(p.size()));
			int i = rand.nextInt(actions.length);
				actions[i] = Math.max(0.0, Math.min(1.0, actions[i] + 0.1 * rand.nextGaussian()));
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
				child.steps.add(Layer.copyArray(mother.getActions(i)));
			for(int i = crossOverPoint; i < father.size(); i++)
				child.steps.add(Layer.copyArray(father.getActions(i)));
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
		if(rand.nextDouble() < 0.35)
			a_prevails = true; // A random plan prevails
		else {
			// The better plan prevails
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
			if(plans.get(i).steps.size() > 0)
			{
				// Remove the first action vector in each plan
				plans.get(i).steps.remove(0);
				
				// Add a random action vector to the end
				double[] newActions = new double[actionDims];
				for(int j = 0; j < actionDims; j++) {
					newActions[j] = rand.nextDouble();
				}
				plans.get(i).steps.add(newActions);
			}
		}
	}


	/// Finds the best plan and copies its first step
	void chooseNextActions(double[] beliefs, double[] actions) {

		// Evaluate all the plans
		int bestPlan = 0;
		double bestContentment = -Double.MAX_VALUE;
		for(int i = 0; i < plans.size(); i++) {
			double d = evaluatePlan(beliefs, plans.get(i));
			if(d > bestContentment) {
				bestContentment = d;
				bestPlan = i;
			}
		}

		// Pick a random alternate plan from the population
		int alternatePlan = rand.nextInt(plans.size() - 1);
		if(alternatePlan >= bestPlan)
			alternatePlan++;

		// Query the teacher
		int feedback = teacher.compare(beliefs, plans.get(bestPlan), plans.get(alternatePlan), transitionModel, observationModel);
		if(feedback != 0) { // If the teacher provided useful feedback...
			if(feedback < 0) { // If the teacher preferred the alternate plan...
				// Swap the two plans
				int tmp = bestPlan;
				bestPlan = alternatePlan;
				alternatePlan = tmp;
			}

			// Refine the model to prefer the final state of the better plan over the final state of the alternate plan
			double[] better = transitionModel.getFinalBeliefs(beliefs, plans.get(bestPlan));
			double[] worse = transitionModel.getFinalBeliefs(beliefs, plans.get(alternatePlan));
			contentmentModel.trainIncremental(better, worse);
		}

		// Copy the first action vector of the best plan
		if(plans.get(bestPlan).size() > 0)
		{
			double[] bestActions = plans.get(bestPlan).getActions(0);
			for(int i = 0; i < bestActions.length; i++) {
				actions[i] = bestActions[i];
			}
		}
		else
		{
			for(int i = 0; i < actions.length; i++) {
				actions[i] = 0.0;
			}
		}
	}
}


/// Implements a weak artificial general intelligence.
public class ManicAgent {
	Random rand;
	TransitionModel transitionModel;
	ObservationModel observationModel;
	ContentmentModel contentmentModel;
	PlanningSystem planningSystem;
	double[] actions;
	double[] beliefs;
	double[] anticipatedBeliefs;


	// Constructor.
	// r is a random number generator.
	// observationDims is the number of double values that the agent observes each time step. (They should fall within a range near [-1, 1] or [0, 1].)
	// beliefDims is the number of double values that the agent uses internally to model the state of the world. (It should generally be <= observationDims.)
	// actionDims is the number of double values the agent uses to specify an action. (The values will range from 0 to 1.)
	// maxPlanLength specifies the maximum number of time-steps into the future that the agent will attempt to plan.
	ManicAgent(Random r, ITeacher oracle, int observationDims, int beliefDims, int actionDims, int maxPlanLength) {
		rand = r;
		transitionModel = new TransitionModel(
			actionDims + beliefDims,
			beliefDims,
			2, // number of layers in the transition model
			500, // size of short term memory for transitions
			50, // number of training iterations to perform with each new sample
			rand);
		observationModel = new ObservationModel(
			observationDims,
			beliefDims,
			2, // number of layers in the decoder
			2, // number of layers in the encoder
			500, // size of short term memory for observations
			50, // number of training iterations to perform with each new sample
			60, // number of iterations to calibrate beliefs to correspond with observations
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
			20, // number of iterations to refine each member of the population per time step
			maxPlanLength,
			0.98, // discount factor (to make short plans be preferred over long plans that ultimately arrive at nearly the same state)
			rand);
		actions = new double[actionDims];
		beliefs = new double[beliefDims];
		anticipatedBeliefs = new double[beliefDims];
	}


	/// Unmarshaling constructor
	ManicAgent(JSONObject obj, Random r, ITeacher oracle) {
		rand = r;
		transitionModel = new TransitionModel((JSONObject)obj.get("transition"), r);
		observationModel = new ObservationModel((JSONObject)obj.get("observation"), r);
		contentmentModel = new ContentmentModel((JSONObject)obj.get("contentment"), r);
		planningSystem = new PlanningSystem((JSONObject)obj.get("planning"), r, transitionModel, observationModel, contentmentModel, oracle);
		actions = new double[transitionModel.actionDims()];
		beliefs = Layer.unmarshalVector((JSONArray)obj.get("beliefs"));
		anticipatedBeliefs = new double[beliefs.length];
	}


	/// Marshals this agent to a JSON DOM.
	JSONObject marshal() {
		JSONObject obj = new JSONObject();
		obj.put("transition", transitionModel.marshal());
		obj.put("observation", observationModel.marshal());
		obj.put("contentment", contentmentModel.marshal());
		obj.put("planning", planningSystem.marshal());
		obj.put("beliefs", Layer.marshalVector(beliefs));
		return obj;
	}


	/// Replaces the teacher with the specified one
	void setTeacher(ITeacher oracle) {
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


	/// A vector of observations goes in
	/// A vector of chosen actions comes out
	double[] think(double[] observations) {

		learnFromExperience(observations);
		return decideWhatToDo();
	}
}
