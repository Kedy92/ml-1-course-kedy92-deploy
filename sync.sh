#!/bin/bash
# Pull latest updates from the book repository

# Add upstream if not already set
if ! git remote | grep -q upstream; then
    echo "Adding upstream remote..."
    git remote add upstream https://github.com/UA-classroom/practical-ml-book.git
fi

echo "Fetching updates from upstream..."
git fetch upstream

echo "Merging upstream/main..."
git merge upstream/main --no-edit

echo "Done! Your repo is up to date."
