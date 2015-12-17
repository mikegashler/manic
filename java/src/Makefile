build: clean
	javac -d ../class -Xmaxerrs 3 Main.java

clean:
	find . -name "*.class" -exec rm {} \;
	find ../class -name "*.class" -exec rm {} \;

run:
	java -cp ../class Main
