#!/bin/sh
echo "hi from poetry"

echo $username
echo $password
echo $gitusername
echo $gitpassword

git config --global user.email "h.javedani@gmail.com"
git config --global user.name $gitusername
git config --global user.password $gitpassword
git remote set-url origin https://github.com/$gitusername/zoish.git 


nox -s release --version=minor --username=$gitusername

poetry publish --username=$username --password=$password

