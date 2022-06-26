#!/bin/sh
echo "hi from poetry"
exec . /venv/bin/activate && poetry publish username=$username password=$password
