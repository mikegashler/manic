import java.util.Random;
import java.awt.image.BufferedImage;
import java.awt.Color;
import java.io.File;
import javax.imageio.ImageIO;
import java.io.FileWriter;
import java.io.FileReader;

class MyTeacher implements ITeacher {
	// Prefer any plan that we believe will make the first observed element smaller
	public int compare(double[] beliefs, Plan planA, Plan planB, TransitionModel transitionModel, ObservationModel observationModel) {
		double[] endBeliefsA = transitionModel.getFinalBeliefs(beliefs, planA);
		double[] endBeliefsB = transitionModel.getFinalBeliefs(beliefs, planB);
		double[] endObsA = observationModel.beliefsToObservations(endBeliefsA);
		double[] endObsB = observationModel.beliefsToObservations(endBeliefsB);
		if(endObsA[0] < endObsB[0])
			return 1;
		else if(endObsB[0] < endObsA[0])
			return -1;
		else
			return 0;
	}
}

class Main {

	static void testNeuralNetMath() throws Exception {
		NeuralNet nn = new NeuralNet();
		Layer l1 = new Layer(2, 3);
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

		Layer l2 = new Layer(3, 2);
		l2.weights.row(0)[0] = 0.1;
		l2.weights.row(0)[1] = 0.1;
		l2.weights.row(1)[0] = 0.1;
		l2.weights.row(1)[1] = 0.3;
		l2.weights.row(2)[0] = 0.1;
		l2.weights.row(2)[1] = -0.1;
		l2.bias[0] = 0.1;
		l2.bias[1] = -0.2;
		nn.layers.add(l2);

		System.out.println("l1 weights:");
		l1.weights.print();
		System.out.println("l1 bias:");
		Matrix.printVec(l1.bias);
		System.out.println("l2 weights:");
		l2.weights.print();
		System.out.println("l2 bias:");
		Matrix.printVec(l2.bias);

		System.out.println("----Forward prop");
		double in[] = new double[2];
		in[0] = 0.3;
		in[1] = -0.2;
		double[] out = nn.forwardProp(in);
		System.out.println("activation:");
		Matrix.printVec(out);

		System.out.println("----Back prop");
		double targ[] = new double[2];
		targ[0] = 0.1;
		targ[1] = 0.0;
		nn.backProp(targ);
		System.out.println("error 2:");
		Matrix.printVec(l2.error);
		System.out.println("error 1:");
		Matrix.printVec(l1.error);
		
		
		nn.descendGradient(in, 0.1);
		System.out.println("----Descending gradient");
		System.out.println("l1 weights:");
		l1.weights.print();
		System.out.println("l1 bias:");
		Matrix.printVec(l1.bias);
		System.out.println("l2 weights:");
		l2.weights.print();
		System.out.println("l2 bias:");
		Matrix.printVec(l2.bias);

		if(Math.abs(l1.weights.row(0)[0] - 0.10039573704287) > 0.0000000001)
			throw new IllegalArgumentException("failed");
		if(Math.abs(l1.weights.row(0)[1] - 0.0013373814241446) > 0.0000000001)
			throw new IllegalArgumentException("failed");
		if(Math.abs(l1.bias[1] - 0.10445793808048) > 0.0000000001)
			throw new IllegalArgumentException("failed");
		System.out.println("passed");
	}

	public static void testMarshaling() throws Exception {
		// Make an agent
		ManicAgent agent = new ManicAgent(new Random(1234), new MyTeacher(), 8, 3, 2);

		// Write it to a file
		JSONObject obj = agent.marshal();
		FileWriter file = new FileWriter("test.json");
		file.write(obj.toJSONString());
		file.close();

		// Read it from a file
		JSONParser parser = new JSONParser();
		JSONObject obj2 = (JSONObject)parser.parse(new FileReader("test.json"));
		ManicAgent agent2 = new ManicAgent(obj2, new Random(1234), new MyTeacher());
		
		System.out.println("passed");
	}


	public static void testNeuralNet() throws Exception {
		// Make some data
		Random rand = new Random(1234);
		Matrix features = new Matrix();
		features.setSize(1000, 2);
		Matrix labels = new Matrix();
		labels.setSize(1000, 2);
		for(int i = 0; i < 1000; i++) {
			
			double x = rand.nextDouble() * 2 - 1;
			double y = rand.nextDouble() * 2 - 1;
			features.row(i)[0] = x;
			features.row(i)[1] = y;
			labels.row(i)[0] = (y < x * x ? 0.9 : 0.1);
			labels.row(i)[1] = (x < y * y ? 0.1 : 0.9);
		}

		// Train on it
		NeuralNet nn = new NeuralNet();
		nn.layers.add(new Layer(2, 30));
		nn.layers.add(new Layer(30, 2));
		nn.init(rand);
		int iters = 10000000;
		double learningRate = 0.01;
		double lambda = 0.0001;
		for(int i = 0; i < iters; i++) {
			int index = rand.nextInt(features.rows());
			nn.regularize(learningRate, lambda);
			nn.trainIncremental(features.row(index), labels.row(index), 0.01);
			if(i % 1000000 == 0)
				System.out.println(Double.toString(((double)i * 100)/ iters) + "%");
		}

		// Visualize it
		for(int i = 0; i < nn.layers.size(); i++) {
			System.out.print("Layer " + Integer.toString(i) + ": ");
			for(int j = 0; j < nn.layers.get(i).hinge.length; j++)
				System.out.print(Double.toString(nn.layers.get(i).hinge[j]) + ", ");
			System.out.println();
		}
		BufferedImage image = new BufferedImage(100, 200, BufferedImage.TYPE_INT_ARGB);
		double[] in = new double[2];
		for(int y = 0; y < 100; y++) {
			for(int x = 0; x < 100; x++) {
				in[0] = ((double)x) / 100 * 2 - 1;
				in[1] = ((double)y) / 100 * 2 - 1;
				double[] out = nn.forwardProp(in);
				int g = Math.max(0, Math.min(255, (int)(out[0] * 256)));
				image.setRGB(x, y, new Color(g, g, g).getRGB());
				g = Math.max(0, Math.min(255, (int)(out[1] * 256)));
				image.setRGB(x, y + 100, new Color(g, g, g).getRGB());
			}
		}
		ImageIO.write(image, "png", new File("viz.png"));
	}

	static void testContentmentFunction() throws Exception {
		System.out.println("training...");
		Random r = new Random(1234);
		ManicAgent agent = new ManicAgent(r, new MyTeacher(), 2, 2, 2);
		double[] bet = new double[2];
		double[] wor = new double[2];
		for(int i = 0; i < 10000; i++) {
/*if(i % 100 == 0) {
			System.out.println("l1 weights:");
			agent.contentmentModel.model.layers.get(0).weights.print();
			System.out.println("l2 weights:");
			agent.contentmentModel.model.layers.get(1).weights.print();
			System.out.println("\n\n");
}*/
		
			bet[0] = 0.1 * r.nextDouble();
			bet[1] = 0.1 * r.nextDouble();
			wor[0] = 0.1 * r.nextDouble();
			wor[1] = 0.1 * r.nextDouble();
			if(bet[0] * bet[0] + bet[1] * bet[1] < wor[0] * wor[0] + wor[1] * wor[1])
				agent.contentmentModel.trainIncremental(bet, wor);
			else
				agent.contentmentModel.trainIncremental(wor, bet);
		}
		
		System.out.println("visualizing...");
		BufferedImage image = new BufferedImage(1000, 1000, BufferedImage.TYPE_INT_ARGB);
		double[] in = new double[2];
		double mi = 10000000;
		double ma = -10000000;
		double[] min_loc = new double[2];
		double[] max_loc = new double[2];
		for(int y = 0; y < 1000; y++) {
			for(int x = 0; x < 1000; x++) {
				in[0] = ((double)x) / 1000.0 * 2.0 - 1.0;
				in[1] = ((double)y) / 1000.0 * 2.0 - 1.0;
				double out = agent.contentmentModel.evaluate(in);
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
		System.out.println("Min=" + Double.toString(mi) + ", Max=" + Double.toString(ma));
		System.out.println("MinLoc=(" + Double.toString(min_loc[0]) + ", " + Double.toString(min_loc[1]) + ")");
		System.out.println("MaxLoc=(" + Double.toString(max_loc[0]) + ", " + Double.toString(max_loc[1]) + ")");
		for(int y = 0; y < 1000; y++) {
			for(int x = 0; x < 1000; x++) {
				in[0] = ((double)x) / 1000.0 * 2.0 - 1.0;
				in[1] = ((double)y) / 1000.0 * 2.0 - 1.0;
				double out = (agent.contentmentModel.evaluate(in) - mi) * 256.0 / (ma - mi);
				int g = Math.max(0, Math.min(255, (int)out));
				image.setRGB(x, y, new Color(g, g, g).getRGB());
			}
		}
		ImageIO.write(image, "png", new File("viz.png"));
	}

	

	static void makeRandomObs(double[] obs, Random r) {
		obs[1] = 0.1 * r.nextDouble();
		obs[2] = 0.1 * r.nextDouble();
		obs[3] = 0.1 * r.nextDouble();
		obs[4] = 0.1 * r.nextDouble();
		obs[0] = obs[1] * obs[1] + obs[2] * obs[2] + obs[3] * obs[3] + obs[4] * obs[4];
	}

	public static void testIntegration() throws Exception {
		Random r = new Random(1234);
		ManicAgent ma = new ManicAgent(r, new MyTeacher(), 5, 4, 5);

		double[] obs = new double[5];
		for(int j = 0; j < 1000; j++) {
			makeRandomObs(obs, r);
			obs[0] = obs[1] * obs[1] + obs[2] * obs[2] + obs[3] * obs[3] + obs[4] * obs[4];
			if(j > 990)
				System.out.println("-------------------------");
//			else
//				System.out.println(Integer.toString(j));

			if(j % 1 == 0) {
				Random rr = new Random(999);
				
				// Evaluate the observation model
				double sse = 0.0;
				for(int k = 0; k < 100; k++) {
					makeRandomObs(obs, rr);
					double[] beliefs = ma.observationModel.observationsToBeliefs(obs);
					double[] pred = ma.observationModel.beliefsToObservations(beliefs);
					for(int l = 0; l < obs.length; l++)
						sse += (obs[l] - pred[l]) * (obs[l] - pred[l]);
				}
				System.out.println("Encoder+Decoder: " + Double.toString(sse));

				// Evaluate contentment model
				int cc = 0;
				for(int k = 0; k < 1000; k++) {
					makeRandomObs(obs, rr);
					double a = obs[0];
					double aa = ma.contentmentModel.evaluate(ma.observationModel.observationsToBeliefs(obs));

					makeRandomObs(obs, rr);
					double b = obs[0];
					double bb = ma.contentmentModel.evaluate(ma.observationModel.observationsToBeliefs(obs));

					boolean correct = false;
					if(a < b && aa > bb)
						correct = true;
					else if(a > b && aa < bb)
						correct = true;
					if(correct)
						cc++;
				}
				System.out.println("Encoder+Contentment correctly evaluate terminating states: " + Integer.toString(cc) + "/1000");

			}

			// Do some playing
			for(int i = 0; i < 10; i++) {
				// think
				double[] act = ma.think(obs);

				// print what is going ont
				if(j > 990) {
					System.out.print("obs: ");
					for(int k = 0; k < obs.length; k++)
						System.out.print(obs[k] + ", ");
					System.out.print("\nact: ");
					for(int k = 0; k < act.length; k++)
						System.out.print(act[k] + ", ");
					System.out.println();
				}

				// apply the action
				int index = 1;
				if(act[2] > act[index]) index = 2;
				if(act[3] > act[index]) index = 3;
				if(act[4] > act[index]) index = 4;
				obs[index] += act[0];
				obs[0] = obs[1] * obs[1] + obs[2] * obs[2] + obs[3] * obs[3] + obs[4] * obs[4];
			}

		}

	}


	public static void main(String[] args) throws Exception {
		//testNeuralNetMath();
		//testNeuralNet();
		//testMarshaling();
		//testContentmentFunction();
		//testIntegration();
		System.out.println("Please view getting_started.html in your favorite browser.");
	}
}
