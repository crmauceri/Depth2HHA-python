#!/usr/bin/env bash

for filename in ${1}; do
    python getHHA.py ${2} filename
done