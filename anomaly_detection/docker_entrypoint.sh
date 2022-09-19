#!/bin/bash

if [ $1 = "client" ]
then
  python client.py ${@:2}
elif [ $1 = "server" ]
then
  python server.py ${@:2}
else
  echo "Unknown option $1"
fi