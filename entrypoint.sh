#!/bin/sh
echo "hi from poetry"

echo $username
echo $password

poetry publish --username=$username --password=$password

