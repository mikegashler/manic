#!/bin/bash
set -e
cd ../src
for (( i=0; i < 10; i++ )); do echo ""; done
javac -d ../class -Xmaxerrs 3 Main.java
