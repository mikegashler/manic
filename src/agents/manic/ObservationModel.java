package agents.manic;

import common.Matrix;
import common.json.JSONObject;
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
	public void calibrateBeliefs(double[] beliefs, double[] observations) {
		for(int i = 0; i < calibrationIters; i++) {
			decoder.refineInputs(beliefs, observations, learningRate);
			for(int j = 0; j < beliefs.length; j++)
				beliefs[j] = Math.max(-1.0, Math.min(1.0, beliefs[j]));
		}
	}


	/// Decodes beliefs to predict observations
	public double[] beliefsToObservations(double[] beliefs) {
		double[] obs = decoder.forwardProp(beliefs);
		double[] ret = new double[obs.length];
		for(int i = 0; i < obs.length; i++) {
			ret[i] = obs[i];
		}

		return ret;
	}


	/// Encodes observations to predict beliefs
	public double[] observationsToBeliefs(double[] observations) {
		double[] bel = encoder.forwardProp(observations);
		double[] ret = new double[bel.length];
		for(int i = 0; i < bel.length; i++) {
			ret[i] = bel[i];
		}

		return ret;
	}
}
