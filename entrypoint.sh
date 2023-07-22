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

# Print environment variables (for debugging purposes, remove in production)
echo "username: $username"
echo "password: $password"
echo "gitusername: $gitusername"
echo "gitpassword: $gitpassword"

# Run nox for the minor release
nox -s release -- minor "$gitusername" 'h.javedani@gmail.com' "$gitpassword"

# Build and publish using poetry
poetry build
poetry publish --username="$username" --password="$password"
