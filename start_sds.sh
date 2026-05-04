#!/bin/bash

# 1. Activate the environment we built
source sds_env/bin/activate

# 2. Set the GPU library paths automatically
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import os, nvidia; print(os.path.dirname(nvidia.__file__))" | xargs -I {} find {} -name "lib" | tr '\n' ':')

echo "------------------------------------------------"
echo "SDS Environment Active & RTX 4050 Linked!"
echo "You can now run: python main.py"
echo "------------------------------------------------"