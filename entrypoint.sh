#!/bin/sh

# Exit if any command fails
set -e

# Check for necessary environment variables
if [ -z "$username" ]; then
    echo "Error: username is not set" >&2
    exit 1
fi

if [ -z "$password" ]; then
    echo "Error: password is not set" >&2
    exit 1
fi

if [ -z "$gitusername" ]; then
    echo "Error: gitusername is not set" >&2
    exit 1
fi

if [ -z "$gitpassword" ]; then
    echo "Error: gitpassword is not set" >&2
    exit 1
fi

# Set git configuration
git config --global user.email "h.javedani@gmail.com"
git config --global user.name "$gitusername"
git config --global user.password "$gitpassword"

# Print out a hello message
echo "Hi from poetry script!"

# Run nox for the minor release
nox -s release -- minor "$gitusername" 'h.javedani@gmail.com' "$gitpassword"

# Build using poetry
echo "Building the package..."
poetry build
if [ $? -ne 0 ]; then
    echo "Failed to build the package!" >&2
    exit 1
fi
echo "Package built successfully."

# Publish using poetry
echo "Publishing the package..."
poetry publish --username="$username" --password="$password" || {
    echo "Failed to publish the package! Here is the output of 'poetry publish' command:" >&2;
    poetry publish --username="$username" --password="$password" --verbose;
    exit 1;
}
echo "Package published successfully."
