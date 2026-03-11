#!/bin/bash
git checkout master
git merge claude/ClaudeCode-Agent
git push origin master
git checkout claude/ClaudeCode-Agent