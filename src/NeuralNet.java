import java.util.Random;
import java.util.ArrayList;
import java.util.Iterator;

class Layer {
	Matrix weights; // rows are inputs, cols are outputs
	double[] bias;
	double[] net;
	double[] activation;
	double[] error;
	double[] hinge;


	Layer(int inputs, int outputs) throws Exception {
		weights = new Matrix();
		weights.setSize(inputs, outputs);
		bias = new double[outputs];
		net = new double[outputs];
		activation = new double[outputs];
		error = new double[outputs];
		hinge = new double[outputs];
	}


	static double[] copyArray(double[] src) {
		double[] dest = new double[src.length];
		for(int i = 0; i < src.length; i++) {
			dest[i] = src[i];
		}
		return dest;
	}


	Layer(Layer that) throws Exception {
		weights = new Matrix(that.weights);
		bias = copyArray(that.bias);
		net = copyArray(that.net);
		activation = copyArray(that.activation);
		error = copyArray(that.error);
		hinge = copyArray(that.hinge);
	}


	Layer(JSONObject obj) {
		weights = new Matrix((JSONObject)obj.get("weights"));
		bias = unmarshalVector((JSONArray)obj.get("bias"));
		net = new double[weights.cols()];
		activation = new double[weights.cols()];
		error = new double[weights.cols()];
		hinge = unmarshalVector((JSONArray)obj.get("hinge"));
		if(bias.length != weights.cols() || hinge.length != weights.cols())
			throw new IllegalArgumentException("mismatching sizes");
	}


	static double[] unmarshalVector(JSONArray arr) {
		Iterator<Double> it = arr.iterator();
		double[] v = new double[arr.size()];
		int i = 0;
		while(it.hasNext()) {
			v[i++] = it.next();
		}
		return v;
	}


	static JSONArray marshalVector(double[] vec) {
		JSONArray v = new JSONArray();
		for(int i = 0; i < vec.length; i++)
			v.add(vec[i]);
		return v;
	}
	

	/// Marshal this object into a JSON DOM.
	JSONObject marshal() {
		JSONObject obj = new JSONObject();
		obj.put("weights", weights.marshal());
		obj.put("bias", marshalVector(bias));
		obj.put("hinge", marshalVector(hinge));
		return obj;
	}


	void copy(Layer src) throws Exception {
		if(src.weights.rows() != weights.rows() || src.weights.cols() != weights.cols())
			throw new IllegalArgumentException("mismatching sizes");
		weights.setSize(0, src.weights.cols());
		weights.copyPart(src.weights, 0, 0, src.weights.rows(), src.weights.cols());
		for(int i = 0; i < bias.length; i++) {
			bias[i] = src.bias[i];
			hinge[i] = src.hinge[i];
		}
	}


	int inputCount() { return weights.rows(); }
	int outputCount() { return weights.cols(); }


	void initWeights(Random r) {
		double dev = Math.max(0.3, 1.0 / weights.rows());
		for(int i = 0; i < weights.rows(); i++) {
			double[] row = weights.row(i);
			for(int j = 0; j < weights.cols(); j++) {
				row[j] = dev * r.nextGaussian();
			}
		}
		for(int j = 0; j < weights.cols(); j++) {
			bias[j] = dev * r.nextGaussian();
			hinge[j] = 0.0;
		}
	}


	void feedForward(double[] in) {
		if(in.length != weights.rows())
			throw new IllegalArgumentException("size mismatch");
		for(int i = 0; i < net.length; i++)
			net[i] = bias[i];
		for(int j = 0; j < weights.rows(); j++) {
			double v = in[j];
			double[] w = weights.row(j);
			for(int i = 0; i < weights.cols(); i++)
				net[i] += v * w[i];
		}
	}


	void feedForward2(double[] in1, double[] in2) {
		if(in1.length + in2.length != weights.rows())
			throw new IllegalArgumentException("size mismatch");
		for(int i = 0; i < net.length; i++)
			net[i] = bias[i];
		for(int j = 0; j < in1.length; j++) {
			double v = in1[j];
			double[] w = weights.row(j);
			for(int i = 0; i < weights.cols(); i++)
				net[i] += v * w[i];
		}
		for(int j = 0; j < in2.length; j++) {
			double v = in2[j];
			double[] w = weights.row(in1.length + j);
			for(int i = 0; i < weights.cols(); i++)
				net[i] += v * w[i];
		}
	}


	void activate() {
		for(int i = 0; i < net.length; i++) {
			//activation[i] = Math.tanh(net[i]);
			activation[i] = hinge[i] * (Math.sqrt(net[i] * net[i] + 1) - 1) + net[i];
		}
	}


	void computeError(double[] target) {
		if(target.length != activation.length)
			throw new IllegalArgumentException("size mismatch");
		for(int i = 0; i < activation.length; i++)
			error[i] = target[i] - activation[i];
	}


	void deactivate() {
		for(int i = 0; i < error.length; i++) {
			//error[i] *= (1.0 - activation[i] * activation[i]);
			error[i] *= (net[i] * hinge[i] / Math.sqrt(net[i] * net[i] + 1) + 1);
		}
	}


	void feedBack(double[] upstream) {
		if(upstream.length != weights.rows())
			throw new IllegalArgumentException("size mismatch");
		for(int j = 0; j < weights.rows(); j++) {
			double[] w = weights.row(j);
			double d = 0.0;
			for(int i = 0; i < weights.cols(); i++) {
				d += error[i] * w[i];
			}
			upstream[j] = d;
		}
	}


	void refineInputs(double[] inputs, double learningRate) {
		if(inputs.length != weights.rows())
			throw new IllegalArgumentException("size mismatch");
		for(int j = 0; j < weights.rows(); j++) {
			double[] w = weights.row(j);
			double d = 0.0;
			for(int i = 0; i < weights.cols(); i++) {
				d += error[i] * w[i];
			}
			inputs[j] += learningRate * d;
		}
	}


	void updateWeights(double[] in, double learningRate) {
		for(int i = 0; i < bias.length; i++) {
			bias[i] += learningRate * error[i];
		}
		for(int j = 0; j < weights.rows(); j++) {
			double[] w = weights.row(j);
			double x = learningRate * in[j];
			for(int i = 0; i < weights.cols(); i++) {
				w[i] += x * error[i];
			}
		}
	}


	void bendHinge(double learningRate) {
		for(int i = 0; i < hinge.length; i++) {
			hinge[i] = Math.max(-1.0, Math.min(1.0, hinge[i] + learningRate * error[i] * (Math.sqrt(net[i] * net[i] + 1.0) - 1.0)));
		}
	}


	// Applies both L2 and L1 regularization to the hinge
	void straightenHinge(double lambda) {
		for(int i = 0; i < hinge.length; i++) {
			hinge[i] *= (1.0 - lambda);
			if(hinge[i] < 0.0)
				hinge[i] += lambda;
			else
				hinge[i] -= lambda;
		}
	}


	// Applies both L2 and L1 regularization to the weights and bias values
	void regularizeWeights(double lambda) {
		for(int i = 0; i < weights.rows(); i++) {
			double[] row = weights.row(i);
			for(int j = 0; j < row.length; j++) {
				row[j] *= (1.0 - lambda);
				if(row[j] < 0.0)
					row[j] += lambda;
				else
					row[j] -= lambda;
			}
		}
		for(int j = 0; j < bias.length; j++) {
			bias[j] *= (1.0 - lambda);
			if(bias[j] < 0.0)
				bias[j] += lambda;
			else
				bias[j] -= lambda;
		}
	}
}


class NeuralNet {
	ArrayList<Layer> layers;


	/// General-purpose constructor. (Starts with no layers. You must add at least one.)
	NeuralNet() {
		layers = new ArrayList<Layer>();
	}


	/// Copy constructor
	NeuralNet(NeuralNet that) throws Exception {
		layers = new ArrayList<Layer>();
		for(int i = 0; i < that.layers.size(); i++) {
			layers.add(new Layer(that.layers.get(i)));
		}
	}


	/// Unmarshaling constructor
	NeuralNet(JSONObject obj) {
		layers = new ArrayList<Layer>();
		JSONArray arrLayers = (JSONArray)obj.get("layers");
		Iterator<JSONObject> it = arrLayers.iterator();
		while(it.hasNext()) {
			JSONObject ob = it.next();
			layers.add(new Layer(ob));
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
	void copy(NeuralNet that) throws Exception {
		if(layers.size() != that.layers.size())
			throw new IllegalArgumentException("Unexpected number of layers");
		for(int i = 0; i < layers.size(); i++) {
			layers.get(i).copy(that.layers.get(i));
		}
	}


	/// Feeds "in" into this neural network and propagates it forward to compute predicted outputs.
	double[] forwardProp(double[] in) {
		Layer l = null;
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
		Layer l = layers.get(0);
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
		Layer l = layers.get(i);
		l.computeError(target);
		l.deactivate();
		for(i--; i > 0; i--) {
			Layer upstream = layers.get(i);
			l.feedBack(upstream.error);
			upstream.deactivate();
			l = upstream;
		}
	}


	/// Backpropagates the error to the upstream layer.
	/// Also, refines the hinge parameter of the activation function by gradient descent (just because this is a convenient place to do that).
	void backPropAndBendHinge(double[] target, double learningRate) {
		int i = layers.size() - 1;
		Layer l = layers.get(i);
		l.computeError(target);
		l.bendHinge(learningRate);
		l.deactivate();
		for(i--; i >= 0; i--) {
			Layer upstream = layers.get(i);
			l.feedBack(upstream.error);
			upstream.bendHinge(learningRate);
			upstream.deactivate();
			l = upstream;
		}
	}


	/// Backpropagates the error from another neural network. (This is used when training autoencoders.)
	void backPropFromDecoder(NeuralNet decoder, double learningRate) {
		int i = layers.size() - 1;
		Layer l = decoder.layers.get(0);
		Layer upstream = layers.get(i);
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
			Layer l = layers.get(i);
			l.updateWeights(in, learningRate);
			in = l.activation;
		}
	}


	/// Keeps the weights and biases from getting too big
	void regularize(double learningRate, double lambda) {
		double amount = learningRate * lambda;
		double smallerAmount = 0.1 * amount;
		for(int i = 0; i < layers.size(); i++) {
			Layer lay = layers.get(i);
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
