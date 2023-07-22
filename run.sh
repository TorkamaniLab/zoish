#!/bin/bash

# Exit if any command fails
set -e

# Display each command before executing it
set -x

# Run the tests session with nox
echo "Running tests with Nox..."
nox -s tests

# If the tests passed, run the linting session
echo "Tests passed. Now running linting with Nox..."
nox -s lint

# If we reached this point, both sessions passed
echo "Both tests and linting passed successfully."
