import java.util.ArrayList;
import java.lang.StringBuilder;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.nio.file.Paths;
import java.nio.file.Files;

abstract class Json
{
	abstract void write(StringBuilder sb);

	public static Json newObject()
	{
		return new JObject();
	}

	public static Json newList()
	{
		return new JList();
	}

	public static Json parseNode(StringParser p)
	{
		p.skipWhitespace();
		if(p.remaining() == 0)
			throw new RuntimeException("Unexpected end of JSON file");
		char c = p.peek();
		if(c == '"')
			return new JString(JString.parseString(p));
		else if(c == '{')
			return JObject.parseObject(p);
		else if(c == '[')
			return JList.parseList(p);
		else if(c == 't')
		{
			p.expect("true");
			return new JBool(true);
		}
		else if(c == 'f')
		{
			p.expect("false");
			return new JBool(false);
		}
		else if(c == 'n')
		{
			p.expect("null");
			return new JNull();
		}
		else if((c >= '0' && c <= '9') || c == '-')
			return JDouble.parseNumber(p);
		else
			throw new RuntimeException("Unexpected token at " + p.str.substring(p.pos, Math.min(p.remaining(), 50)));
	}

	public int size()
	{
		return this.asList().size();
	}

	public Json get(String name)
	{
		return this.asObject().field(name);
	}

	public Json get(int index)
	{
		return this.asList().get(index);
	}

	public boolean getBool(String name)
	{
		return get(name).asBool();
	}

	public boolean getBool(int index)
	{
		return get(index).asBool();
	}

	public long getLong(String name)
	{
		return get(name).asLong();
	}

	public long getLong(int index)
	{
		return get(index).asLong();
	}

	public double getDouble(String name)
	{
		return get(name).asDouble();
	}

	public double getDouble(int index)
	{
		return get(index).asDouble();
	}

	public String getString(String name)
	{
		return get(name).asString();
	}

	public String getString(int index)
	{
		return get(index).asString();
	}

	public void add(String name, Json val)
	{
		this.asObject().add(name, val);
	}

	public void add(String name, boolean val)
	{
		this.asObject().add(name, new Json.JBool(val));
	}

	public void add(String name, long val)
	{
		this.asObject().add(name, new Json.JLong(val));
	}

	public void add(String name, double val)
	{
		this.asObject().add(name, new Json.JDouble(val));
	}

	public void add(String name, String val)
	{
		this.asObject().add(name, new Json.JString(val));
	}

	public void add(Json item)
	{
		this.asList().add(item);
	}

	public void add(boolean val)
	{
		this.asList().add(new Json.JBool(val));
	}

	public void add(long val)
	{
		this.asList().add(new Json.JLong(val));
	}

	public void add(double val)
	{
		this.asList().add(new Json.JDouble(val));
	}

	public void add(String val)
	{
		this.asList().add(new Json.JString(val));
	}

	public boolean asBool()
	{
		return ((JBool)this).value;
	}

	public long asLong()
	{
		return ((JLong)this).value;
	}

	public double asDouble()
	{
		return ((JDouble)this).value;
	}

	public String asString()
	{
		return ((JString)this).value;
	}

	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		write(sb);
		return sb.toString();
	}

	private JObject asObject()
	{
		return (JObject)this;
	}

	private JList asList()
	{
		return (JList)this;
	}

	public void save(String filename)
	{
		try
		{
			BufferedWriter out = new BufferedWriter(new FileWriter(filename));
			out.write(toString());
			out.close();
		}
		catch(Exception e)
		{
			throw new RuntimeException(e);
		}
	}

	public static Json parse(String s)
	{
		StringParser p = new StringParser(s);
		return Json.parseNode(p);
	}

	public static Json load(String filename)
	{
		String contents;
		try
		{
			contents = new String(Files.readAllBytes(Paths.get(filename)));
		}
		catch(Exception e)
		{
			throw new RuntimeException(e);
		}
		return parse(contents);
	}

	private static class StringParser
	{
		String str;
		int pos;

		StringParser(String s)
		{
			str = s;
			pos = 0;
		}

		int remaining()
		{
			return str.length() - pos;
		}

		char peek()
		{
			return str.charAt(pos);
		}

		void advance(int n)
		{
			pos += n;
		}

		void skipWhitespace()
		{
			while(pos < str.length() && str.charAt(pos) <= ' ')
				pos++;
		}

		void expect(String s)
		{
			if(!str.substring(pos, Math.min(str.length(), pos + s.length())).equals(s))
				throw new RuntimeException("Expected \"" + s + "\", Got \"" + str.substring(pos, Math.min(str.length(), pos + s.length())) + "\"");
			pos += s.length();
		}

		String until(char c)
		{
			int i = pos;
			while(i < str.length() && str.charAt(i) != c)
				i++;
			String s = str.substring(pos, i);
			pos = i;
			return s;
		}

		String until(char a, char b)
		{
			int i = pos;
			while(i < str.length() && str.charAt(i) != a && str.charAt(i) != b)
				i++;
			String s = str.substring(pos, i);
			pos = i;
			return s;
		}

		String whileReal()
		{
			int i = pos;
			while(i < str.length())
			{
				char c = str.charAt(i);
				if((c >= '0' && c <= '9') ||
					c == '-' ||
					c == '+' ||
					c == '.' ||
					c == 'e' ||
					c == 'E')
					i++;
				else
					break;
			}
			String s = str.substring(pos, i);
			pos = i;
			return s;
		}
	}

	private static class NameVal
	{
		String name;
		Json value;

		NameVal(String nam, Json val)
		{
			if(nam == null)
				throw new IllegalArgumentException("The name cannot be null");
			if(val == null)
				val = new JNull();
			name = nam;
			value = val;
		}
	}

	private static class JObject extends Json
	{
		ArrayList<NameVal> fields;

		JObject()
		{
			fields = new ArrayList<NameVal>();
		}

		public void add(String name, Json val)
		{
			fields.add(new NameVal(name, val));
		}

		Json fieldIfExists(String name)
		{
			for(NameVal nv : fields)
			{
				if(nv.name.equals(name))
					return nv.value;
			}
			return null;
		}

		Json field(String name)
		{
			Json n = fieldIfExists(name);
			if(n == null)
				throw new RuntimeException("No field named \"" + name + "\" found.");
			return n;
		}

		void write(StringBuilder sb)
		{
			sb.append("{");
			for(int i = 0; i < fields.size(); i++)
			{
				if(i > 0)
					sb.append(",");
				NameVal nv = fields.get(i);
				JString.write(sb, nv.name);
				sb.append(":");
				nv.value.write(sb);
			}
			sb.append("}");
		}

		static JObject parseObject(StringParser p)
		{
			p.expect("{");
			JObject newOb = new JObject();
			boolean readyForField = true;
			while(p.remaining() > 0)
			{
				char c = p.peek();
				if(c <= ' ')
				{
					p.advance(1);
				}
				else if(c == '}')
				{
					p.advance(1);
					return newOb;
				}
				else if(c == ',')
				{
					if(readyForField)
						throw new RuntimeException("Unexpected ','");
					p.advance(1);
					readyForField = true;
				}
				else if(c == '\"')
				{
					if(!readyForField)
						throw new RuntimeException("Expected a ',' before the next field in JSON file");
					p.skipWhitespace();
					String name = JString.parseString(p);
					p.skipWhitespace();
					p.expect(":");
					Json value = Json.parseNode(p);
					newOb.add(name, value);
					readyForField = false;
				}
				else
					throw new RuntimeException("Expected a '}' or a '\"'. Got " + p.str.substring(p.pos, p.pos + 10));
			}
			throw new RuntimeException("Expected a matching '}' in JSON file");
		}
	}

	private static class JList extends Json
	{
		ArrayList<Json> list;

		JList()
		{
			list = new ArrayList<Json>();
		}

		public void add(Json item)
		{
			if(item == null)
				item = new JNull();
			list.add(item);
		}

		public int size()
		{
			return list.size();
		}

		public Json get(int index)
		{
			return list.get(index);
		}

		void write(StringBuilder sb)
		{
			sb.append("[");
			for(int i = 0; i < list.size(); i++)
			{
				if(i > 0)
					sb.append(",");
				list.get(i).write(sb);
			}
			sb.append("]");
		}

		static JList parseList(StringParser p)
		{
			p.expect("[");
			JList newList = new JList();
			boolean readyForValue = true;
			while(p.remaining() > 0)
			{
				p.skipWhitespace();
				char c = p.peek();
				if(c == ']')
				{
					p.advance(1);
					return newList;
				}
				else if(c == ',')
				{
					if(readyForValue)
						throw new RuntimeException("Unexpected ',' in JSON file");
					p.advance(1);
					readyForValue = true;
				}
				else
				{
					if(!readyForValue)
						throw new RuntimeException("Expected a ',' or ']' in JSON file");
					newList.list.add(Json.parseNode(p));
					readyForValue = false;
				}
			}
			throw new RuntimeException("Expected a matching ']' in JSON file");
		}
	}

	private static class JBool extends Json
	{
		boolean value;

		JBool(boolean val)
		{
			value = val;
		}

		void write(StringBuilder sb)
		{
			sb.append(value ? "true" : "false");
		}
	}

	private static class JLong extends Json
	{
		long value;

		JLong(long val)
		{
			value = val;
		}

		void write(StringBuilder sb)
		{
			sb.append(value);
		}
	}

	private static class JDouble extends Json
	{
		double value;

		JDouble(double val)
		{
			value = val;
		}

		void write(StringBuilder sb)
		{
			sb.append(value);
		}

		static Json parseNumber(StringParser p)
		{
			String s = p.whileReal();
			if(s.indexOf('.') >= 0)
				return new JDouble(Double.parseDouble(s));
			else
				return new JLong(Long.parseLong(s));
		}
	}

	private static class JString extends Json
	{
		String value;

		JString(String val)
		{
			value = val;
		}

		static void write(StringBuilder sb, String value)
		{
			sb.append('"');
			for(int i = 0; i < value.length(); i++)
			{
				char c = value.charAt(i);
				if(c < ' ')
				{
					switch(c)
					{
						case '\b': sb.append("\\b"); break;
						case '\f': sb.append("\\f"); break;
						case '\n': sb.append("\\n"); break;
						case '\r': sb.append("\\r"); break;
						case '\t': sb.append("\\t"); break;
						default:
							sb.append(c);
					}
				}
				else if(c == '\\')
					sb.append("\\\\");
				else if(c == '"')
					sb.append("\\\"");
				else
					sb.append(c);
			}
			sb.append('"');
		}

		void write(StringBuilder sb)
		{
			write(sb, value);
		}

		static String parseString(StringParser p)
		{
			StringBuilder sb = new StringBuilder();
			p.expect("\"");
			while(p.remaining() > 0)
			{
				char c = p.peek();
				if(c == '\"')
				{
					p.advance(1);
					return sb.toString();
				}
				else if(c == '\\')
				{
					p.advance(1);
					c = p.peek();
					p.advance(1);
					switch(c)
					{
						case '"': sb.append('"'); break;
						case '\\': sb.append('\\'); break;
						case '/': sb.append('/'); break;
						case 'b': sb.append('\b'); break;
						case 'f': sb.append('\f'); break;
						case 'n': sb.append('\n'); break;
						case 'r': sb.append('\r'); break;
						case 't': sb.append('\t'); break;
						case 'u': throw new RuntimeException("Sorry, unicode characters are not yet supported");
						default: throw new RuntimeException("Unrecognized escape sequence");
					}
				}
				else
				{
					sb.append(c);
					p.advance(1);
				}
			}
			throw new RuntimeException("No closing \"");
		}
	}

	private static class JNull extends Json
	{
		JNull()
		{
		}

		void write(StringBuilder sb)
		{
			sb.append("null");
		}
	}
}
