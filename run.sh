#!/bin/bash
nox -s test 
nox -s lint
nox -s release -- minor

