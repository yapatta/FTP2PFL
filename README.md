# research_impl
For research

## Test

```go
$ go test -v -race -run TestFederatedLearning |& tee /tmp/raftlog
... logging output
... test should PASS
$ go run ./tools/raft-testlog-viz/main.go < /tmp/raftlog
PASS TestElectionFollowerComesBack map[0:true 1:true 2:true TEST:true] ; entries: 150
... Emitted file:///tmp/TestElectionFollowerComesBack.html

PASS
```
