import java.util.Random;

/// A bidirectional model that maps between beliefs and observations.
/// Mapping from observations to beliefs is done by the encoder.
/// Mapping from beliefs to observations is done by the decoder.
/// These two components are trained together in an unsupervised manner as an autoencoder.
public class ObservationModel {
	public Random rand;
	public NeuralNet decoder;
	public NeuralNet encoder;
	NeuralNet decoderExperimental;
	NeuralNet encoderExperimental;
	public Matrix train;
	public Matrix validation;
	ITutor tutor;
	TransitionModel transitionModel;
	public int trainPos;
	public int trainSize;
	int validationPos;
	int validationSize;
	int trainIters;
	int trainProgress;
	int calibrationIters;
	public double learningRate;


	/// General-purpose constructor
	ObservationModel(TransitionModel transition, int observation_dims, int belief_dims, int decoder_layers,
		int encoder_layers, int queue_size, int trainItersPerPattern, int calibrationIterations, Random r) {

		if(belief_dims > observation_dims)
			throw new IllegalArgumentException("observation_dims must be >= belief_dims");

		// Init the encoder
		rand = r;
		int hidden = Math.max(30, (observation_dims + belief_dims) / 2);
		encoder = new NeuralNet();
		encoder.layers.add(new LayerLinear(observation_dims, hidden));
		encoder.layers.add(new LayerTanh(hidden));
		encoder.layers.add(new LayerLinear(hidden, belief_dims));
		encoder.layers.add(new LayerTanh(belief_dims));
		encoder.init(rand);

		// Init the decoder
		decoder = new NeuralNet();
		decoder.layers.add(new LayerLinear(belief_dims, hidden));
		decoder.layers.add(new LayerTanh(hidden));
		decoder.layers.add(new LayerLinear(hidden, observation_dims));
		decoder.layers.add(new LayerTanh(observation_dims));
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
	ObservationModel(TransitionModel transition, Json obj, Random r) {
		rand = r;
		decoder = new NeuralNet(obj.get("decoder"));
		encoder = new NeuralNet(obj.get("encoder"));
		decoderExperimental = new NeuralNet(obj.get("decoderExperimental"));
		encoderExperimental = new NeuralNet(obj.get("encoderExperimental"));
		train = new Matrix(obj.get("train"));
		validation = new Matrix(obj.get("validation"));
		trainPos = (int)obj.getLong("trainPos");
		trainSize = (int)obj.getLong("trainSize");
		validationPos = (int)obj.getLong("validationPos");
		validationSize = (int)obj.getLong("validationSize");
		trainIters = (int)obj.getLong("trainIters");
		trainProgress = (int)obj.getLong("trainProgress");
		calibrationIters = (int)obj.getLong("calibrationIters");
		learningRate = obj.getDouble("learningRate");
		transitionModel = transition;
	}


	/// Marshals this model to a JSON DOM.
	Json marshal() {
		Json obj = Json.newObject();
		obj.add("decoder", decoder.marshal());
		obj.add("encoder", encoder.marshal());
		obj.add("decoderExperimental", decoderExperimental.marshal());
		obj.add("encoderExperimental", encoderExperimental.marshal());
		obj.add("train", train.marshal());
		obj.add("validation", validation.marshal());
		obj.add("trainPos", trainPos);
		obj.add("trainSize", trainSize);
		obj.add("validationPos", validationPos);
		obj.add("validationSize", validationSize);
		obj.add("trainIters", trainIters);
		obj.add("trainProgress", trainProgress);
		obj.add("calibrationIters", calibrationIters);
		obj.add("learningRate", learningRate);
		return obj;
	}


	void setTutor(ITutor t) {
		tutor = t;
	}


	/// Performs one pattern-presentation of stochastic gradient descent and dynamically tunes the learning rate
	void doSomeTraining() {

		// Train the decoderExperimental and encoderExperimental together as an autoencoder
		decoderExperimental.regularize(learningRate * 0.00001);
		encoderExperimental.regularize(learningRate * 0.00001);
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
				
				encoder = new NeuralNet(encoderExperimental);
				decoder = new NeuralNet(decoderExperimental);
				transitionModel.trainPos = 0;
				transitionModel.trainSize = 0;
			}
			else if(err1 < 0.85 * err2) {
				// This should really never happen
				encoderExperimental = new NeuralNet(encoder);
				decoderExperimental = new NeuralNet(decoder);
			}
			//System.out.println("Observation error: " + Double.toString(err1) + ", " + Double.toString(err2));
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
	public void calibrateBeliefs(double[] beliefs, double[] observations) {
		if(tutor != null)
			Vec.copy(beliefs, tutor.observationsToState(observations));
		for(int i = 0; i < calibrationIters; i++) {
			decoder.refineInputs(beliefs, observations, learningRate);
			for(int j = 0; j < beliefs.length; j++)
				beliefs[j] = Math.max(-1.0, Math.min(1.0, beliefs[j]));
		}
	}


	/// Decodes beliefs to predict observations
	public double[] beliefsToObservations(double[] beliefs) {
		if(tutor != null)
			return tutor.stateToObservations(beliefs);
		double[] obs = decoder.forwardProp(beliefs);
		double[] ret = new double[obs.length];
		for(int i = 0; i < obs.length; i++) {
			ret[i] = obs[i];
		}
		return ret;
	}


	/// Encodes observations to predict beliefs
	public double[] observationsToBeliefs(double[] observations) {
		if(tutor != null)
			return tutor.observationsToState(observations);
		double[] bel = encoder.forwardProp(observations);
		double[] ret = new double[bel.length];
		for(int i = 0; i < bel.length; i++) {
			ret[i] = bel[i];
		}
		return ret;
	}
}
