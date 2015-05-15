import java.io.PrintWriter;

/** Provides static methods for operating on arrays of doubles */
public class Vec
{
	public static void print(double[] vec) {
		System.out.print("[");
		if(vec.length > 0) {
			System.out.print(Double.toString(vec[0]));
			for(int i = 1; i < vec.length; i++) {
				System.out.print(",	" + Double.toString(vec[i]));
			}
		}
		System.out.print("]");
	}

	public static void println(double[] vec) {
		print(vec);
		System.out.println();
	}

	public static void setAll(double[] vec, double val) {
		for(int i = 0; i < vec.length; i++)
			vec[i] = val;
	}

	public static double squaredMagnitude(double[] vec) {
		double d = 0.0;
		for(int i = 0; i < vec.length; i++)
			d += vec[i] * vec[i];
		return d;
	}

	public static double dotProduct(double[] a, double[] b) {
		if(a.length != b.length)
			throw new IllegalArgumentException("mismatching sizes");
		double d = 0.0;
		for(int i = 0; i < a.length; i++)
			d += a[i] * b[i];
		return d;
	}

	public static double squaredDistance(double[] a, double[] b) {
		if(a.length != b.length)
			throw new IllegalArgumentException("mismatching sizes");
		double d = 0.0;
		for(int i = 0; i < a.length; i++) {
			double t = a[i] - b[i];
			d += t * t;
		}
		return d;
	}
}
