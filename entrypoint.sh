#!/bin/sh
echo "hi from poetry "

echo $username
echo $password
echo $gitusername
echo $gitpassword

git config --global user.email "h.javedani@gmail.com"
git config --global user.name $gitusername
git config --global user.password $gitpassword


nox -s release -- minor $gitusername 'h.javedani@gmail.com' $gitpassword

poetry build
poetry publish --username=$username --password=$password

