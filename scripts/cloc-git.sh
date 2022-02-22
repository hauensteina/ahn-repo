#!/usr/bin/env bash

# Count lines of code in a git repo by language.
# Example Usage:
# cloc-git https://github.com/evalEmpire/perl5i.git

git clone --depth 1 "$1" temp-linecount-repo &&
  printf "('temp-linecount-repo' will be deleted automatically)\n\n\n" &&
  cloc temp-linecount-repo &&
  rm -rf temp-linecount-repo

