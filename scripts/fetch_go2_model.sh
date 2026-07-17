#!/usr/bin/env bash
# Fetch the official MuJoCo Go2 model (sparse checkout of just the Go2 folder).
# The model is not vendored in git (binary meshes + its own .git); run this once.
set -e
DEST="assets/mujoco_menagerie"
if [ -d "$DEST/unitree_go2" ]; then
  echo "Go2 model already present at $DEST/unitree_go2"
  exit 0
fi
mkdir -p assets
cd assets
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie
git sparse-checkout set unitree_go2
echo "Fetched Go2 model -> $DEST/unitree_go2"
