#!/usr/bin/env bash
__conda_setup="$('/data/disk01/home/jasonfan/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/data/disk01/home/jasonfan/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/data/disk01/home/jasonfan/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/data/disk01/home/jasonfan/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate mladev
NEW_SHELL=$1
shift
ARGV=$*

pushd `dirname $0` >/dev/null
BASEDIR=`pwd`
popd >/dev/null

$NEW_SHELL $ARGV