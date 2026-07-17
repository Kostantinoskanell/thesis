#!/usr/bin/env bash
# Quick capability probe of the WSL2 Ubuntu env before the (large) RL install.
echo "python3: $(python3 --version 2>&1)"
echo "pip:     $(python3 -m pip --version 2>/dev/null || echo MISSING)"
python3 -c "import venv; print('venv:    ok')" 2>/dev/null || echo "venv:    MISSING (need: sudo apt install python3-venv)"
echo "git:     $(git --version 2>/dev/null || echo MISSING)"
echo "nvidia:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo MISSING)"
echo "disk:    $(df -h "$HOME" | awk 'NR==2{print $4" free of "$2}')"
free -h | awk 'NR==2{print "mem:     "$2" total, "$7" available"}'
