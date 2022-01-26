#!/bin/zsh
# ex: ./eval.sh args.json
# args.json > {"testname": "TestFLonRaft", "client_num": "10", "counts": "8", "sec": "150"}
JSON=$(cat "${1}")
TESTNAME=$(echo $JSON | jq -r '.testname')
CLIENT_NUM=$(echo $JSON | jq -r '.client_num')
COUNTS=$(echo $JSON | jq -r '.counts')
SEC=$(echo $JSON | jq -r '.sec')

DIR="./data/${TESTNAME}/${CLIENT_NUM}-${SEC}"
mkdir -p ${DIR}

for i in {1..$COUNTS}; do
	export SEC=${SEC} CLIENT_NUM=${CLIENT_NUM} && go test -v -race -run ${TESTNAME} |& tee /tmp/raftlog | grep "aggregated model" | cut -d " " -f 1,6,8 | sed -r 's/[0-9]+:[0-9]+:[0-9]+\.[0-9]+/&,/' > "${DIR}/${i}.csv"
done

