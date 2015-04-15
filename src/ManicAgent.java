import java.util.Random;
import java.util.ArrayList;
import java.util.Iterator;


/// A model that maps from current beliefs and actions to anticipated beliefs.
/// This model is trained in a supervised manner.
class TransitionModel {
	Random rand;
	NeuralNet model;
	NeuralNet backup;
	Matrix trainInput;
	Matrix trainOutput;
	Matrix validationInput;
	Matrix validationOutput;
	int trainPos;
	int trainSize;
	int validationPos;
	int validationSize;
	int trainIters;
	int trainProgress;
	double learningRate;
	double prevErr;


	/// General-purpose constructor
	TransitionModel(int input_dims, int output_dims, int total_layers, int queue_size, int trainItersPerPattern, Random r) throws Exception {

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

		// Make the backup
		backup = new NeuralNet(model);

		// Init the buffers
		trainInput = new Matrix(queue_size, input_dims);
		trainOutput = new Matrix(queue_size, output_dims);
		validationInput = new Matrix(queue_size, input_dims);
		validationOutput = new Matrix(queue_size, output_dims);

		// Init the meta-parameters
		trainIters = trainItersPerPattern;
		learningRate = 0.01;
	}


	/// Unmarshaling constructor
	TransitionModel(JSONObject obj, Random r) {
		rand = r;
		model = new NeuralNet((JSONObject)obj.get("model"));
		backup = new NeuralNet((JSONObject)obj.get("backup"));
		trainInput = new Matrix((JSONObject)obj.get("trainInput"));
		trainOutput = new Matrix((JSONObject)obj.get("trainOutput"));
		validationInput = new Matrix((JSONObject)obj.get("validationInput"));
		validationOutput = new Matrix((JSONObject)obj.get("validationOutput"));
		trainPos = ((Long)obj.get("trainPos")).intValue();
		trainSize = ((Long)obj.get("trainSize")).intValue();
		validationPos = ((Long)obj.get("validationPos")).intValue();
		validationSize = ((Long)obj.get("validationSize")).intValue();
		trainIters = ((Long)obj.get("trainIters")).intValue();
		trainProgress = ((Long)obj.get("trainProgress")).intValue();
		learningRate = (Double)obj.get("learningRate");
		prevErr = (Double)obj.get("prevErr");
	}


	/// Marshals this model to a JSON DOM.
	JSONObject marshal() {
		JSONObject obj = new JSONObject();
		obj.put("model", model.marshal());
		obj.put("backup", backup.marshal());
		obj.put("trainInput", trainInput.marshal());
		obj.put("trainOutput", trainOutput.marshal());
		obj.put("validationInput", validationInput.marshal());
		obj.put("validationOutput", validationOutput.marshal());
		obj.put("trainPos", trainPos);
		obj.put("trainSize", trainSize);
		obj.put("validationPos", validationPos);
		obj.put("validationSize", validationSize);
		obj.put("trainIters", trainIters);
		obj.put("trainProgress", trainProgress);
		obj.put("learningRate", learningRate);
		obj.put("prevErr", prevErr);
		return obj;
	}


	/// Returns the number of action dims
	int actionDims() { return model.layers.get(0).inputCount() - model.layers.get(model.layers.size() - 1).outputCount(); }


	/// Performs one pattern-presentation of stochastic gradient descent, and dynamically tunes the learning rate
	void doSomeTraining() throws Exception {

		// Present one pattern
		model.regularize(learningRate, 0.0001);
		int index = rand.nextInt(trainSize);
		model.trainIncremental(trainInput.row(index), trainOutput.row(index), learningRate);

		// Dynamically tune the learning rate (at periodic intervals)
		trainProgress++;
		if(trainProgress >= trainInput.rows()) {
			// Measure mean squared error
			trainProgress = 0;
			double err = 0.0;
			for(int i = 0; i < validationSize; i++) {
				double[] prediction = model.forwardProp(validationInput.row(i));
				double[] targ = validationOutput.row(i);
				for(int j = 0; j < targ.length; j++)
					err += (targ[j] - prediction[j]) * (targ[j] - prediction[j]);
			}
			err /= validationSize;

			// Dynamically tune the learning rate
			if(err <= prevErr || prevErr == 0) { // If the model improved, or this is the first time...
				backup.copy(model); // back up the model
				learningRate = Math.min(0.1, learningRate * 1.2); // gradually increase the learning rate
			} else {
				model.copy(backup); // Restore the weights from the backup
				learningRate = Math.max(1e-6, learningRate * 0.1); // Dramatically decrease the learning rate
				prevErr *= 1.05; // Gradually raise the former threshold (since the training and validation data are always changing).
			}
			prevErr = err;
		}
	}


	/// Refines this model based on a recently performed action and change in beliefs
	void trainIncremental(double[] beliefs, double[] actions, double[] nextBeliefs) throws Exception {

		// Buffer the pattern
		double[] destIn;
		double[] destOut;
		if(validationPos < trainPos) {
			destIn = validationInput.row(validationPos);
			destOut = validationOutput.row(validationPos);
			if(++validationPos >= validationInput.rows())
				validationPos = 0;
			validationSize = Math.max(validationSize, validationPos);
		} else {
			destIn = trainInput.row(trainPos);
			destOut = trainOutput.row(trainPos);
			if(++trainPos >= trainInput.rows())
				trainPos = 0;
			trainSize = Math.max(trainSize, trainPos);
		}
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
	NeuralNet decoderBackup;
	NeuralNet encoderBackup;
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
	double prevErr;


	/// General-purpose constructor
	ObservationModel(int observation_dims, int belief_dims, int decoder_layers, int encoder_layers, int queue_size, int trainItersPerPattern, int calibrationIterations, Random r) throws Exception {
		// Init the decoder
		rand = r;
		decoder = new NeuralNet();
		for(int i = 0; i < decoder_layers; i++) {
			int in = ((belief_dims * (decoder_layers - i)) + (observation_dims * i)) / (decoder_layers);
			int j = i + 1;
			int out = ((belief_dims * (decoder_layers - j)) + (observation_dims * j)) / (decoder_layers);
			decoder.layers.add(new Layer(in, out));
		}
		decoder.init(rand);

		// Init the encoder
		encoder = new NeuralNet();
		for(int i = 0; i < encoder_layers; i++) {
			int in = ((observation_dims * (encoder_layers - i)) + (belief_dims * i)) / (encoder_layers);
			int j = i + 1;
			int out = ((observation_dims * (encoder_layers - j)) + (belief_dims * j)) / (encoder_layers);
			encoder.layers.add(new Layer(in, out));
		}
		encoder.init(rand);

		// Make the backups
		decoderBackup = new NeuralNet(decoder);
		encoderBackup = new NeuralNet(encoder);

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
		decoderBackup = new NeuralNet((JSONObject)obj.get("decoderBackup"));
		encoderBackup = new NeuralNet((JSONObject)obj.get("encoderBackup"));
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
		prevErr = (Double)obj.get("prevErr");
	}


	/// Marshals this model to a JSON DOM.
	JSONObject marshal() {
		JSONObject obj = new JSONObject();
		obj.put("decoder", decoder.marshal());
		obj.put("encoder", encoder.marshal());
		obj.put("decoderBackup", decoderBackup.marshal());
		obj.put("encoderBackup", encoderBackup.marshal());
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
		obj.put("prevErr", prevErr);
		return obj;
	}


	/// Performs one pattern-presentation of stochastic gradient descent and dynamically tunes the learning rate
	void doSomeTraining() throws Exception {

		// Train the decoder and encoder together as an autoencoder
		decoder.regularize(learningRate, 0.0001);
		encoder.regularize(learningRate, 0.0001);
		int index = rand.nextInt(trainSize);
		double[] observation = train.row(index);
		double[] belief = encoder.forwardProp(observation);
		double[] prediction = decoder.forwardProp(belief);
		decoder.backPropAndBendHinge(observation, learningRate);
		encoder.backPropFromDecoder(decoder, learningRate);
		encoder.descendGradient(observation, learningRate);
		decoder.descendGradient(belief, learningRate);

		// Dynamically tune the learning rate at periodic intervals
		trainProgress++;
		if(trainProgress >= train.rows()) {
			// Measure mean squared error
			trainProgress = 0;
			double err = 0.0;
			for(int i = 0; i < validationSize; i++) {
				double[] targ = validation.row(i);
				double[] pred = decoder.forwardProp(encoder.forwardProp(targ));
				for(int j = 0; j < targ.length; j++)
					err += (targ[j] - pred[j]) * (targ[j] - pred[j]);
			}
			err /= validationSize;

			// Dynamically tune the learning rate
			if(err <= prevErr || prevErr == 0) { // If the model improved, or this is the first time...
				encoderBackup.copy(encoder); // back up the encoder
				decoderBackup.copy(decoder); // back up the decoder
				learningRate = Math.min(0.1, learningRate * 1.2); // gradually increase the learning rate
			} else {
				encoder.copy(encoderBackup); // Restore the encoder from the backup
				decoder.copy(decoderBackup); // Restore the decoder from the backup
				learningRate = Math.max(1e-6, learningRate * 0.1); // Dramatically decrease the learning rate
				prevErr *= 1.05; // Gradually raise the former threshold (since the training and validation data are always changing).
			}
			prevErr = err;
		}
	}


	/// Refines the encoder and decoder based on the new observation.
	void trainIncremental(double[] observation) throws Exception {

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
}


/// A model that maps from anticipated beliefs to contentment (or utility).
/// This model is trained by reinforcement from a teacher.
class ContentmentModel {
	Random rand;
	NeuralNet model;
	NeuralNet backup;
	Matrix better;
	Matrix worse;
	int trainPos;
	int trainSize;
	int trainIters;
	int trainProgress;
	double learningRate;
	double prevErr;
	double[] targBuf;


	// General-purpose constructor
	ContentmentModel(int beliefDims, int total_layers, int queue_size, int trainItersPerPattern, Random r) throws Exception {
		// Init the model
		rand = r;
		model = new NeuralNet();
		for(int i = 0; i < total_layers; i++) {
			int in = ((beliefDims * (total_layers - i)) + i) / (total_layers);
			int j = i + 1;
			int out = ((beliefDims * (total_layers - j)) + j) / (total_layers);
			model.layers.add(new Layer(in, out));
		}
		model.init(rand);

		// Make the backup
		backup = new NeuralNet(model);

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
		backup = new NeuralNet((JSONObject)obj.get("backup"));
		better = new Matrix((JSONObject)obj.get("better"));
		worse = new Matrix((JSONObject)obj.get("worse"));
		trainPos = ((Long)obj.get("trainPos")).intValue();
		trainSize = ((Long)obj.get("trainSize")).intValue();
		trainIters = ((Long)obj.get("trainIters")).intValue();
		trainProgress = ((Long)obj.get("trainProgress")).intValue();
		learningRate = (Double)obj.get("learningRate");
		prevErr = (Double)obj.get("prevErr");
	}


	/// Marshals this model to a JSON DOM.
	JSONObject marshal() {
		JSONObject obj = new JSONObject();
		obj.put("model", model.marshal());
		obj.put("backup", backup.marshal());
		obj.put("better", better.marshal());
		obj.put("worse", worse.marshal());
		obj.put("trainPos", trainPos);
		obj.put("trainSize", trainSize);
		obj.put("trainIters", trainIters);
		obj.put("trainProgress", trainProgress);
		obj.put("learningRate", learningRate);
		obj.put("prevErr", prevErr);
		return obj;
	}


	/// Performs one pattern-presentation of stochastic gradient descent, and dynamically tunes the learning rate
	void doSomeTraining() throws Exception {

		// Present a single pair for rank-based training
		int index = rand.nextInt(trainSize);
		double bet = evaluate(better.row(index));
		double wor = evaluate(worse.row(index));
		if(wor >= bet) {
			model.regularize(learningRate, 0.0001);
			targBuf[0] = wor + 0.02;
			model.trainIncremental(better.row(index), targBuf, learningRate);
			targBuf[0] = bet - 0.02;
			model.trainIncremental(worse.row(index), targBuf, learningRate);
		}

		// Dynamically tune the learning rate
		trainProgress++;
		if(trainProgress >= better.rows()) {
			// Measure misclassification rate
			trainProgress = 0;
			double err = 0.0;
			for(int i = 0; i < better.rows(); i++) {
				bet = evaluate(better.row(index));
				wor = evaluate(worse.row(index));
				if(wor >= bet)
					err++;
			}
			err /= better.rows();

			// Dynamically tune the learning rate
			if(err <= prevErr || prevErr == 0) { // If the model improved, or this is the first time...
				backup.copy(model); // back up the model
				learningRate = Math.min(0.1, learningRate * 1.2); // gradually increase the learning rate
			} else {
				model.copy(backup); // Restore the weights from the backup
				learningRate = Math.max(1e-6, learningRate * 0.1); // Dramatically decrease the learning rate
				prevErr *= 1.05; // Gradually raise the former threshold (since the training and validation data are always changing).
			}
			prevErr = err;
		}
	}


	/// Refines this model based on feedback from the teacher
	void trainIncremental(double[] bet, double[] wor) throws Exception {

		// Buffer the samples
		double[] dest = better.row(trainPos);
		for(int i = 0; i < bet.length; i++)
			dest[i] = bet[i];
		dest = worse.row(trainPos);
		for(int i = 0; i < wor.length; i++)
			dest[i] = wor[i];
		if(++trainPos >= better.rows())
			trainPos = 0;
		trainSize = Math.max(trainSize, trainPos);

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
	Random rand;


	// General-purpose constructor
	PlanningSystem(TransitionModel transition, ObservationModel observation, ContentmentModel contentment, ITeacher oracle, int actionDimensions, int populationSize, int planRefinementIters, int maxPlanLen, Random r) {
		transitionModel = transition;
		observationModel = observation;
		contentmentModel = contentment;
		teacher = oracle;
		rand = r;
		plans = new ArrayList<Plan>();
		if(populationSize < 2)
			throw new IllegalArgumentException("The population size must be at least 2");
		for(int i = 0; i < populationSize; i++)
			plans.add(new Plan());
		actionDims = actionDimensions;
		maxPlanLength = maxPlanLen;
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
		obj.put("refinementIters", refinementIters);
		obj.put("actionDims", actionDims);
		return obj;
	}


	/// Perturbs a random plan
	void mutate() {
		double d = rand.nextDouble();
		Plan p = plans.get(rand.nextInt(plans.size()));
		if(d < 0.2 || p.size() == 0) { // lengthen the plan
			if(p.size() < maxPlanLength) {
				double[] newActions = new double[actionDims];
				for(int i = 0; i < actionDims; i++) {
					newActions[i] = rand.nextDouble();
				}
				p.steps.add(rand.nextInt(p.size() + 1), newActions);
			}
		}
		else if(d < 0.35) { // shorten the plan
			if(p.size() > 0) {
				p.steps.remove(rand.nextInt(p.size()));
			}
		}
		else if(d < 0.5) { // perturb a whole action vector
			if(p.size() > 0) {
				double[] actions = p.getActions(rand.nextInt(p.size()));
				for(int i = 0; i < actions.length; i++) {
					actions[i] = Math.max(0.0, Math.min(1.0, actions[i] + 0.1 * rand.nextGaussian()));
				}
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
			if(p.size() > 0) {
				double[] actions = p.getActions(rand.nextInt(p.size()));
				int i = rand.nextInt(actions.length);
					actions[i] = Math.max(0.0, Math.min(1.0, actions[i] + 0.1 * rand.nextGaussian()));
			}
		}
	}


	/// Replaces the specified plan with a new one.
	void replace(int childIndex) {
		double d = rand.nextDouble();
		if(d < 0.6) {
			// Clone a random parent (asexual reproduction)
			plans.set(childIndex, new Plan(plans.get(rand.nextInt(plans.size()))));
		} else if(d < 0.8) {
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
					c[j] = alpha * a[j] + (1.0 - alpha) * b[j];
				}
				child.steps.add(c);
			}
			plans.set(childIndex, child);
		}
	}


	/// Returns the expected contentment at the end of the plan
	double evaluatePlan(double[] beliefs, Plan plan) {
		return contentmentModel.evaluate(transitionModel.getFinalBeliefs(beliefs, plan));
	}


	/// Performs a tournament between two randomly-selected plans.
	/// One of them, usually the winner, is replaced.
	void tournament(double[] beliefs) {
		int a = rand.nextInt(plans.size());
		int b = rand.nextInt(plans.size());
		boolean a_prevails;
		if(rand.nextDouble() < 0.35)
			a_prevails = true;
		else if(evaluatePlan(beliefs, plans.get(a)) >= evaluatePlan(beliefs, plans.get(b)))
			a_prevails = true;
		else
			a_prevails = false;
		replace(a_prevails ? b : a);
	}


	/// Performs one iteration of plan refinement
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
				plans.get(i).steps.remove(0);
		}
	}


	/// Finds the best plan and copies its first step
	void chooseNextActions(double[] beliefs, double[] actions) throws Exception {

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
	static final int CALIBRATION_ITERS = 6;
	static final int DECODER_LAYERS = 2;
	static final int ENCODER_LAYERS = 2;
	static final int TRANSITION_LAYERS = 2;
	static final int CONTENTMENT_LAYERS = 2;
	static final int SHORT_TERM_MEMORY_SIZE = 500;
	static final int TRAIN_ITERS_PER_PATTERN = 50;
	static final int PLANNING_POPULATION_SIZE = 100;
	static final int PLAN_REFINEMENT_ITERS = 500;
	static final int MAX_PLAN_LENGTH = 10;

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
	ManicAgent(Random r, ITeacher oracle, int observationDims, int beliefDims, int actionDims) throws Exception {
		rand = r;
		transitionModel = new TransitionModel(actionDims + beliefDims, beliefDims, TRANSITION_LAYERS, SHORT_TERM_MEMORY_SIZE, TRAIN_ITERS_PER_PATTERN, rand);
		observationModel = new ObservationModel(observationDims, beliefDims, DECODER_LAYERS, ENCODER_LAYERS, SHORT_TERM_MEMORY_SIZE, TRAIN_ITERS_PER_PATTERN, CALIBRATION_ITERS, rand);
		contentmentModel = new ContentmentModel(beliefDims, CONTENTMENT_LAYERS, SHORT_TERM_MEMORY_SIZE, TRAIN_ITERS_PER_PATTERN, rand);
		planningSystem = new PlanningSystem(transitionModel, observationModel, contentmentModel, oracle, actionDims, PLANNING_POPULATION_SIZE, PLAN_REFINEMENT_ITERS, MAX_PLAN_LENGTH, rand);
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


	/// Learns from observations
	void learnFromExperience(double[] observations) throws Exception {

		// Learn to perceive the world a little better
		observationModel.trainIncremental(observations);

		// Refine beliefs to correspond with the new observations better
		observationModel.calibrateBeliefs(anticipatedBeliefs, observations);

		// Learn to anticipate consequences a little better
		transitionModel.trainIncremental(beliefs, actions, anticipatedBeliefs);
	}


	/// Returns an action vector
	double[] decideWhatToDo() throws Exception {

		// Make the anticipated beliefs the new beliefs
		double[] tmp = beliefs;
		beliefs = anticipatedBeliefs;
		anticipatedBeliefs = tmp;

		// Drop the first action in every plan
		planningSystem.advanceTime();

		// Try to find a better plan
		planningSystem.refinePlans(beliefs);

		// Choose an action that is expected to maximize contentment
		planningSystem.chooseNextActions(beliefs, actions);

		// Anticipate how the world will change with time
		transitionModel.anticipateNextBeliefsInPlace(beliefs, actions, anticipatedBeliefs);

		// Return the selected actions
		return actions;
	}


	/// A vector of observations goes in
	/// A vector of chosen actions comes out
	double[] think(double[] observations) throws Exception {

		learnFromExperience(observations);
		return decideWhatToDo();
	}
}
