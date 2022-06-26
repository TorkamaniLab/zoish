#!/bin/sh
echo "hi from poetry"
exec source/venv/bin/activate && poetry publish username=$username password=$password
