#!/bin/sh
ls -a 
exec poetry publish username=u password=p
echo "hi from poetry"