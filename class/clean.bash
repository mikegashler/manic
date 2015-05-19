#!/bin/bash
set -e
find ../src -name "*.class" -exec rm {} \;
find . -name "*.class" -exec rm {} \;
