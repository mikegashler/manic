import java.util.Random;
import java.util.ArrayList;
import java.util.Iterator;
import java.awt.image.BufferedImage;
import java.awt.Color;
import java.io.File;
import javax.imageio.ImageIO;


public class NeuralNet
{
	public ArrayList<Layer> layers;


	/// General-purpose constructor. (Starts with no layers. You must add at least one.)
	NeuralNet()
	{
		layers = new ArrayList<Layer>();
	}


	/// Copy constructor
	NeuralNet(NeuralNet that)
	{
		layers = new ArrayList<Layer>();
		for(int i = 0; i < that.layers.size(); i++)
		{
			layers.add(that.layers.get(i).clone());
		}
	}


	/// Unmarshals from a JSON DOM.
	NeuralNet(Json n)
	{
		layers = new ArrayList<Layer>();
		Json l = n.get("layers");
		for(int i = 0; i < l.size(); i++)
			layers.add(Layer.unmarshal(l.get(i)));
	}


	/// Marshal this neural network into a JSON DOM.
	Json marshal()
	{
		Json ob = Json.newObject();
		Json l = Json.newList();
		ob.add("layers", l);
		for(int i = 0; i < layers.size(); i++)
			l.add(layers.get(i).marshal());
		return ob;
	}


	/// Initializes the weights and biases with small random values
	void init(Random r)
	{
		for(int i = 0; i < layers.size(); i++)
		{
			layers.get(i).initWeights(r);
		}
	}


	/// Feeds "in" into this neural network and propagates it forward to compute predicted outputs.
	double[] forwardProp(double[] in)
	{
		for(int i = 0; i < layers.size(); i++)
		{
			in = layers.get(i).forwardProp(in);
		}
		return in;
	}


	/// Feeds the concatenation of "in1" and "in2" into this neural network and propagates it forward to compute predicted outputs.
	double[] forwardProp2(double[] in1, double[] in2)
	{
		double[] in = ((LayerLinear)layers.get(0)).forwardProp2(in1, in2);
		for(int i = 1; i < layers.size(); i++)
		{
			in = layers.get(i).forwardProp(in);
		}
		return in;
	}


	/// Backpropagates the error to the upstream layer.
	void backProp(double[] target)
	{
		int i = layers.size() - 1;
		Layer l = layers.get(i);
		l.computeError(target);
		for(i--; i >= 0; i--)
		{
			Layer upstream = layers.get(i);
			l.backProp(upstream);
			l = upstream;
		}
	}


	/// Backpropagates the error from another neural network. (This is used when training autoencoders.)
	void backPropFromDecoder(NeuralNet decoder)
	{
		int i = layers.size() - 1;
		Layer l = decoder.layers.get(0);
		Layer upstream = layers.get(i);
		l.backProp(upstream);
		l = upstream;
		for(i--; i >= 0; i--)
		{
			upstream = layers.get(i);
			l.backProp(upstream);
			l = upstream;
		}
	}


	/// Updates the weights and biases
	void descendGradient(double[] in, double learningRate)
	{
		for(int i = 0; i < layers.size(); i++)
		{
			Layer l = layers.get(i);
			l.scaleGradient(0.0);
			l.updateGradient(in);
			l.step(learningRate);
			in = l.activation;
		}
	}


	/// Keeps the weights and biases from getting too big
	void regularize(double amount)
	{
		for(int i = 0; i < layers.size(); i++)
		{
			Layer lay = layers.get(i);
			lay.regularizeWeights(amount);
		}
	}


	/// Refines the weights and biases with on iteration of stochastic gradient descent.
	void trainIncremental(double[] in, double[] target, double learningRate)
	{
		forwardProp(in);
		backProp(target);
		//backPropAndBendHinge(target, learningRate);
		descendGradient(in, learningRate);
	}


	/// Refines "in" with one iteration of stochastic gradient descent.
	void refineInputs(double[] in, double[] target, double learningRate)
	{
		forwardProp(in);
		backProp(target);
		((LayerLinear)layers.get(0)).refineInputs(in, learningRate);
	}


	static void testMath()
	{
		NeuralNet nn = new NeuralNet();
		LayerLinear l1 = new LayerLinear(2, 3);
		l1.weights.row(0)[0] = 0.1;
		l1.weights.row(0)[1] = 0.0;
		l1.weights.row(0)[2] = 0.1;
		l1.weights.row(1)[0] = 0.1;
		l1.weights.row(1)[1] = 0.0;
		l1.weights.row(1)[2] = -0.1;
		l1.bias[0] = 0.1;
		l1.bias[1] = 0.1;
		l1.bias[2] = 0.0;
		nn.layers.add(l1);
		nn.layers.add(new LayerTanh(3));

		LayerLinear l2 = new LayerLinear(3, 2);
		l2.weights.row(0)[0] = 0.1;
		l2.weights.row(0)[1] = 0.1;
		l2.weights.row(1)[0] = 0.1;
		l2.weights.row(1)[1] = 0.3;
		l2.weights.row(2)[0] = 0.1;
		l2.weights.row(2)[1] = -0.1;
		l2.bias[0] = 0.1;
		l2.bias[1] = -0.2;
		nn.layers.add(l2);
		nn.layers.add(new LayerTanh(2));

		System.out.println("l1 weights:" + l1.weights.toString());
		System.out.println("l1 bias:" + Vec.toString(l1.bias));
		System.out.println("l2 weights:" + l2.weights.toString());
		System.out.println("l2 bias:" + Vec.toString(l2.bias));

		System.out.println("----Forward prop");
		double in[] = new double[2];
		in[0] = 0.3;
		in[1] = -0.2;
		double[] out = nn.forwardProp(in);
		System.out.println("activation:" + Vec.toString(out));

		System.out.println("----Back prop");
		double targ[] = new double[2];
		targ[0] = 0.1;
		targ[1] = 0.0;
		nn.backProp(targ);
		System.out.println("error 2:" + Vec.toString(l2.error));
		System.out.println("error 1:" + Vec.toString(l1.error));
		
		nn.descendGradient(in, 0.1);
		System.out.println("----Descending gradient");
		System.out.println("l1 weights:" + l1.weights.toString());
		System.out.println("l1 bias:" + Vec.toString(l1.bias));
		System.out.println("l2 weights:" + l2.weights.toString());
		System.out.println("l2 bias:" + Vec.toString(l2.bias));

		if(Math.abs(l1.weights.row(0)[0] - 0.10039573704287) > 0.0000000001)
			throw new IllegalArgumentException("failed");
		if(Math.abs(l1.weights.row(0)[1] - 0.0013373814241446) > 0.0000000001)
			throw new IllegalArgumentException("failed");
		if(Math.abs(l1.bias[1] - 0.10445793808048) > 0.0000000001)
			throw new IllegalArgumentException("failed");
		System.out.println("passed");
	}
}
