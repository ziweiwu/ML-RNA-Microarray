#!/usr/bin/env bash
COMMIT_MSG=$1

git add -A
git commit -m $COMMIT_MSG
git push origin development