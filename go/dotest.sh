#!/bin/bash
set -x
set -e

logfile=/tmp/raftlog

go test -v -race -run $@ |& tee ${logfile}

go run ./tools/raft-testlog-viz/main.go < ${logfile}
