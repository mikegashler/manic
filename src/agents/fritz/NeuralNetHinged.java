package agents.fritz;

import java.util.Random;
import java.util.ArrayList;
import java.util.Iterator;
import common.json.JSONObject;
import common.json.JSONArray;


class NeuralNetHinged {
	ArrayList<LayerHinged> layers;


	/// General-purpose constructor. (Starts with no layers. You must add at least one.)
	NeuralNetHinged() {
		layers = new ArrayList<LayerHinged>();
	}


	/// Copy constructor
	NeuralNetHinged(NeuralNetHinged that) {
		layers = new ArrayList<LayerHinged>();
		for(int i = 0; i < that.layers.size(); i++) {
			layers.add(new LayerHinged(that.layers.get(i)));
		}
	}


	/// Unmarshaling constructor
	NeuralNetHinged(JSONObject obj) {
		layers = new ArrayList<LayerHinged>();
		JSONArray arrLayers = (JSONArray)obj.get("layers");
		Iterator<JSONObject> it = arrLayers.iterator();
		while(it.hasNext()) {
			JSONObject ob = it.next();
			layers.add(new LayerHinged(ob));
		}
	}


	/// Initializes the weights and biases with small random values
	void init(Random r) {
		for(int i = 0; i < layers.size(); i++) {
			layers.get(i).initWeights(r);
		}
	}


	/// Marshals this object into a JSON DOM.
	JSONObject marshal() {
		JSONObject obj = new JSONObject();
		JSONArray lay = new JSONArray();
		for(int i = 0; i < layers.size(); i++) {
			lay.add(layers.get(i).marshal());
		}
		obj.put("layers", lay);
		return obj;
	}


	/// Copies all the weights and biases from "that" into "this".
	/// (Assumes the corresponding topologies already match.)
	void copy(NeuralNetHinged that) {
		if(layers.size() != that.layers.size())
			throw new IllegalArgumentException("Unexpected number of layers");
		for(int i = 0; i < layers.size(); i++) {
			layers.get(i).copy(that.layers.get(i));
		}
	}


	/// Feeds "in" into this neural network and propagates it forward to compute predicted outputs.
	double[] forwardProp(double[] in) {
		LayerHinged l = null;
		for(int i = 0; i < layers.size(); i++) {
			l = layers.get(i);
			l.feedForward(in);
			l.activate();
			in = l.activation;
		}
		return l.activation;
	}


	/// Feeds the concatenation of "in1" and "in2" into this neural network and propagates it forward to compute predicted outputs.
	double[] forwardProp2(double[] in1, double[] in2) {
		LayerHinged l = layers.get(0);
		l.feedForward2(in1, in2);
		l.activate();
		double[] in = l.activation;
		for(int i = 1; i < layers.size(); i++) {
			l = layers.get(i);
			l.feedForward(in);
			l.activate();
			in = l.activation;
		}
		return l.activation;
	}


	/// Backpropagates the error to the upstream layer.
	void backProp(double[] target) {
		int i = layers.size() - 1;
		LayerHinged l = layers.get(i);
		l.computeError(target);
		l.deactivate();
		for(i--; i >= 0; i--) {
			LayerHinged upstream = layers.get(i);
			l.feedBack(upstream.error);
			upstream.deactivate();
			l = upstream;
		}
	}


	/// Backpropagates the error to the upstream layer.
	/// Also, refines the hinge parameter of the activation function by gradient descent (just because this is a convenient place to do that).
	void backPropAndBendHinge(double[] target, double learningRate) {
		int i = layers.size() - 1;
		LayerHinged l = layers.get(i);
		l.computeError(target);
		l.bendHinge(learningRate);
		l.deactivate();
		for(i--; i >= 0; i--) {
			LayerHinged upstream = layers.get(i);
			l.feedBack(upstream.error);
			upstream.bendHinge(learningRate);
			upstream.deactivate();
			l = upstream;
		}
	}


	/// Backpropagates the error from another neural network. (This is used when training autoencoders.)
	void backPropFromDecoder(NeuralNetHinged decoder, double learningRate) {
		int i = layers.size() - 1;
		LayerHinged l = decoder.layers.get(0);
		LayerHinged upstream = layers.get(i);
		l.feedBack(upstream.error);
		l = upstream;
		l.bendHinge(learningRate);
		l.deactivate();
		for(i--; i >= 0; i--) {
			upstream = layers.get(i);
			l.feedBack(upstream.error);
			upstream.bendHinge(learningRate);
			upstream.deactivate();
			l = upstream;
		}
	}


	/// Updates the weights and biases
	void descendGradient(double[] in, double learningRate) {
		for(int i = 0; i < layers.size(); i++) {
			LayerHinged l = layers.get(i);
			l.updateWeights(in, learningRate);
			in = l.activation;
		}
	}


	/// Keeps the weights and biases from getting too big
	void regularize(double learningRate, double lambda) {
		double amount = learningRate * lambda;
		double smallerAmount = 0.1 * amount;
		for(int i = 0; i < layers.size(); i++) {
			LayerHinged lay = layers.get(i);
			lay.straightenHinge(amount);
			lay.regularizeWeights(smallerAmount);
		}
	}


	/// Refines the weights and biases with on iteration of stochastic gradient descent.
	void trainIncremental(double[] in, double[] target, double learningRate) {
		forwardProp(in);
		backPropAndBendHinge(target, learningRate);
		descendGradient(in, learningRate);
	}


	/// Refines "in" with one iteration of stochastic gradient descent.
	void refineInputs(double[] in, double[] target, double learningRate) {
		forwardProp(in);
		backProp(target);
		layers.get(0).refineInputs(in, learningRate);
	}
}
