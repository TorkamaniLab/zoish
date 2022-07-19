#!/bin/sh
echo "hi from poetry "

echo $username
echo $password
echo $gitusername
echo $gitpassword

git config --global user.email "h.javedani@gmail.com"
git config --global user.name $gitusername
git config --global user.password $gitpassword
<<<<<<< HEAD
git remote set-url origin https://github.com/$gitusername/zoish.git 
=======
>>>>>>> f4c35da600f9f056e0fcdd772fed207c4067b705


nox -s release -- minor $gitusername 'h.javedani@gmail.com' $gitpassword

poetry build
poetry publish --username=$username --password=$password

