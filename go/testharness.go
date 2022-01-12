// Test harness for writing tests for Raft.
//
// Eli Bendersky [https://eli.thegreenplace.net]
// This code is in the public domain.
package raft

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"reflect"
	"sync"
	"testing"
	"time"
)

const ModelSize = 784 * 4
const LastModelSize = 10 * 4

type Upload struct {
	Model []byte `json:"model"`
}

func init() {
	log.SetFlags(log.Ltime | log.Lmicroseconds)
	seed := time.Now().UnixNano()
	fmt.Println("seed", seed)
	rand.Seed(seed)
}

type Harness struct {
	mu sync.Mutex

	// cluster is a list of all the raft servers participating in a cluster.
	cluster []*Server
	storage []*MapStorage

	// commitChans has a channel per server in cluster with the commit channel for
	// that server.
	commitChans []chan CommitEntry

	// commits at index i holds the sequence of commits made by server[i] so far.
	// It is populated by goroutines that listen on the corresponding commitChans
	// channel.
	commits [][]CommitEntry

	// connected has a bool per server in cluster, specifying whether this server
	// is currently connected to peers (if false, it's partitioned and no messages
	// will pass to or from it).
	connected []bool

	// alive has a bool per server in cluster, specifying whether this server is
	// currently alive (false means it has crashed and wasn't restarted yet).
	// connected implies alive.
	alive []bool

	srv *http.Server

	srvWg *sync.WaitGroup

	n int
	t *testing.T
}

// NewHarness creates a new test Harness, initialized with n servers connected
// to each other.
func NewHarness(t *testing.T, n int) *Harness {
	ns := make([]*Server, n)
	connected := make([]bool, n)
	alive := make([]bool, n)
	commitChans := make([]chan CommitEntry, n)
	commits := make([][]CommitEntry, n)
	ready := make(chan interface{})
	storage := make([]*MapStorage, n)

	// Create all Servers in this cluster, assign ids and peer ids.
	for i := 0; i < n; i++ {
		peerIds := make([]int, 0)
		for p := 0; p < n; p++ {
			if p != i {
				peerIds = append(peerIds, p)
			}
		}

		storage[i] = NewMapStorage()
		commitChans[i] = make(chan CommitEntry)
		ns[i] = NewServer(i, peerIds, storage[i], ready, commitChans[i])
		ns[i].Serve()
		alive[i] = true
	}

	// Connect all peers to each other.
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != j {
				ns[i].ConnectToPeer(j, ns[j].GetListenAddr())
			}
		}
		connected[i] = true
	}
	// ElectionTimerを始動
	close(ready)

	srvWg := &sync.WaitGroup{}
	srvWg.Add(1)

	h := &Harness{
		cluster:     ns,
		storage:     storage,
		commitChans: commitChans,
		commits:     commits,
		connected:   connected,
		alive:       alive,
		srv:         nil,
		srvWg:       srvWg,
		n:           n,
		t:           t,
	}

	// TODO: Timelimitを追加
	srv := &http.Server{
		Handler: h.NewHandler(),
		Addr:    ":8888",
	}
	h.srv = srv

	go func() {
		defer srvWg.Done()
		if err := srv.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatalf("ListenAndServe(): %v", err)
		}
	}()

	InitializePython()
	SetPath()

	for i := 0; i < n; i++ {
		go h.collectCommits(i)
	}
	return h
}

func (h *Harness) NewHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/upload", h.HTTPUploadModel)
	mux.HandleFunc("/download", h.HTTPDownloadModel)
	return mux
}

func (h *Harness) HTTPUploadModel(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		if val := r.Header.Get("Content-Type"); val != "" {
			if val != "application/json" {
				msg := "Content-Type header is not application/json"
				http.Error(w, msg, http.StatusUnsupportedMediaType)
				return
			}
		}
		body := r.Body
		defer body.Close()

		dec := json.NewDecoder(body)
		dec.DisallowUnknownFields()

		var u Upload
		if err := dec.Decode(&u); err != nil {
			msg := "JSON decode failed"
			http.Error(w, msg, http.StatusInternalServerError)
			return
		}

		origLeaderId, _ := h.CheckSingleLeader()
		if origLeaderId < 0 {
			msg := "Leader doesn't exist"
			http.Error(w, msg, http.StatusInternalServerError)
			return
		}

		if !h.SubmitToServer(origLeaderId, u.Model) {
			msg := fmt.Sprintf("want id=%d leader, but it's not", origLeaderId)
			http.Error(w, msg, http.StatusInternalServerError)
			return
		}

		w.WriteHeader(http.StatusOK)
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
		w.Write([]byte("only post method allowed"))
	}
}

func (h *Harness) HTTPDownloadModel(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:

		origLeaderId, _ := h.CheckSingleLeader()
		if origLeaderId < 0 {
			msg := "Leader doesn't exist"
			http.Error(w, msg, http.StatusInternalServerError)
			return
		}

		lastCommitId := len(h.commits[origLeaderId]) - 1
		if lastCommitId < 0 {
			msg := "Commit ID doesn't exist"
			http.Error(w, msg, http.StatusInternalServerError)
			return
		}

		lastCommit := h.commits[origLeaderId][lastCommitId]
		m := lastCommit.Command.([]byte)

		u := &Upload{
			Model: m,
		}

		jsonResp, err := json.Marshal(u)
		if err != nil {
			msg := fmt.Sprintf("json.Marshal failed: %v", err)
			http.Error(w, msg, http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.Write(jsonResp)
		w.WriteHeader(http.StatusOK)
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
		w.Write([]byte("only get method allowed"))
	}
}

// Shutdown shuts down all the servers in the harness and waits for them to
// stop running.
func (h *Harness) Shutdown() {
	for i := 0; i < h.n; i++ {
		h.cluster[i].DisconnectAll()
		h.mu.Lock()
		h.connected[i] = false
		h.mu.Unlock()
	}
	for i := 0; i < h.n; i++ {
		if h.alive[i] {
			h.alive[i] = false
			h.cluster[i].Shutdown()
		}
	}
	for i := 0; i < h.n; i++ {
		close(h.commitChans[i])
	}

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if err := h.srv.Shutdown(ctx); err != nil {
		log.Printf("Failed to gracefully shutdown: %v", err)
	}
	h.srvWg.Wait()

	FinalizePython()
}

// DisconnectPeer disconnects a server from all other servers in the cluster.
func (h *Harness) DisconnectPeer(id int) {
	tlog("Disconnect %d", id)
	h.cluster[id].DisconnectAll()
	for j := 0; j < h.n; j++ {
		if j != id {
			h.cluster[j].DisconnectPeer(id)
		}
	}
	h.mu.Lock()
	h.connected[id] = false
	h.mu.Unlock()
}

// ReconnectPeer connects a server to all other servers in the cluster.
func (h *Harness) ReconnectPeer(id int) {
	tlog("Reconnect %d", id)
	for j := 0; j < h.n; j++ {
		if j != id && h.alive[j] {
			if err := h.cluster[id].ConnectToPeer(j, h.cluster[j].GetListenAddr()); err != nil {
				h.t.Fatal(err)
			}
			if err := h.cluster[j].ConnectToPeer(id, h.cluster[id].GetListenAddr()); err != nil {
				h.t.Fatal(err)
			}
		}
	}
	h.mu.Lock()
	h.connected[id] = true
	h.mu.Unlock()
}

// CrashPeer "crashes" a server by disconnecting it from all peers and then
// asking it to shut down. We're not going to use the same server instance
// again, but its storage is retained.
func (h *Harness) CrashPeer(id int) {
	tlog("Crash %d", id)
	h.DisconnectPeer(id)
	h.alive[id] = false
	h.cluster[id].Shutdown()

	// Clear out the commits slice for the crashed server; Raft assumes the client
	// has no persistent state. Once this server comes back online it will replay
	// the whole log to us.
	h.mu.Lock()
	h.commits[id] = h.commits[id][:0] // 空のsliceになる [0:0)なので
	h.mu.Unlock()
}

// RestartPeer "restarts" a server by creating a new Server instance and giving
// it the appropriate storage, reconnecting it to peers.
func (h *Harness) RestartPeer(id int) {
	if h.alive[id] {
		log.Fatalf("id=%d is alive in RestartPeer", id)
	}
	tlog("Restart %d", id)

	peerIds := make([]int, 0)
	for p := 0; p < h.n; p++ {
		if p != id {
			peerIds = append(peerIds, p)
		}
	}

	ready := make(chan interface{})
	h.cluster[id] = NewServer(id, peerIds, h.storage[id], ready, h.commitChans[id])
	h.cluster[id].Serve()
	h.ReconnectPeer(id)
	close(ready)
	h.alive[id] = true
	sleepMs(20)
}

// CheckSingleLeader checks that only a single server thinks it's the leader.
// Returns the leader's id and term. It retries several times if no leader is
// identified yet.
func (h *Harness) CheckSingleLeader() (int, int) {
	//v 8回試してリーダーを見つけられなかったらダメ
	for r := 0; r < 8; r++ {
		leaderId := -1
		leaderTerm := -1
		for i := 0; i < h.n; i++ {
			h.mu.Lock()
			isConnected := h.connected[i]
			h.mu.Unlock()
			// connectしているリーダーは一人だけ
			if isConnected {
				_, term, isLeader := h.cluster[i].cm.Report()
				if isLeader {
					if leaderId < 0 {
						leaderId = i
						leaderTerm = term
					} else {
						h.t.Fatalf("both %d and %d think they're leaders", leaderId, i)
					}
				}
			}
		}
		if leaderId >= 0 {
			return leaderId, leaderTerm
		}
		time.Sleep(150 * time.Millisecond)
	}

	h.t.Fatalf("leader not found")
	return -1, -1
}

// CheckNoLeader checks that no connected server considers itself the leader.
func (h *Harness) CheckNoLeader() {
	for i := 0; i < h.n; i++ {
		if h.connected[i] {
			_, _, isLeader := h.cluster[i].cm.Report()
			if isLeader {
				h.t.Fatalf("server %d leader; want none", i)
			}
		}
	}
}

// CheckCommitted verifies that all connected servers have cmd committed with
// the same index. It also verifies that all commands *before* cmd in
// the commit sequence match. For this to work properly, all commands submitted
// to Raft should be unique positive ints.
// Returns the number of servers that have this command committed, and its
// log index.
// TODO: this check may be too strict. Consider tha a server can commit
// something and crash before notifying the channel. It's a valid commit but
// this checker will fail because it may not match other servers. This scenario
// is described in the paper...
func (h *Harness) CheckCommitted(cmd int) (nc int, index int) {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Find the length of the commits slice for connected servers.
	commitsLen := -1
	for i := 0; i < h.n; i++ {
		if h.connected[i] {
			if commitsLen >= 0 {
				// If this was set already, expect the new length to be the same.
				if len(h.commits[i]) != commitsLen {
					h.t.Fatalf("commits[%d] = %d, commitsLen = %d", i, h.commits[i], commitsLen)
				}
			} else {
				commitsLen = len(h.commits[i])
			}
		}
	}

	// Check consistency of commits from the start and to the command we're asked
	// about. This loop will return once a command=cmd is found.
	for c := 0; c < commitsLen; c++ {
		cmdAtC := -1
		for i := 0; i < h.n; i++ {
			if h.connected[i] {
				cmdOfN := h.commits[i][c].Command.(int)
				if cmdAtC >= 0 {
					if cmdOfN != cmdAtC {
						h.t.Errorf("got %d, want %d at h.commits[%d][%d]", cmdOfN, cmdAtC, i, c)
					}
				} else {
					cmdAtC = cmdOfN
				}
			}
		}
		if cmdAtC == cmd {
			// Check consistency of Index.
			index := -1
			nc := 0
			for i := 0; i < h.n; i++ {
				if h.connected[i] {
					if index >= 0 && h.commits[i][c].Index != index {
						h.t.Errorf("got Index=%d, want %d at h.commits[%d][%d]", h.commits[i][c].Index, index, i, c)
					} else {
						index = h.commits[i][c].Index
					}
					nc++
				}
			}
			return nc, index
		}
	}

	// If there's no early return, we haven't found the command we were looking
	// for.
	h.t.Errorf("cmd=%d not found in commits", cmd)
	return -1, -1
}

// CheckCommittedN verifies that cmd was committed by exactly n connected
// servers.
func (h *Harness) CheckCommittedN(cmd int, n int) {
	nc, _ := h.CheckCommitted(cmd)
	if nc != n {
		h.t.Errorf("CheckCommittedN got nc=%d, want %d", nc, n)
	}
}

// CheckNotCommitted verifies that no command equal to cmd has been committed
// by any of the active servers yet.
func (h *Harness) CheckNotCommitted(cmd int) {
	h.mu.Lock()
	defer h.mu.Unlock()

	for i := 0; i < h.n; i++ {
		if h.connected[i] {
			for c := 0; c < len(h.commits[i]); c++ {
				gotCmd := h.commits[i][c].Command.(int)
				if gotCmd == cmd {
					h.t.Errorf("found %d at commits[%d][%d], expected none", cmd, i, c)
				}
			}
		}
	}
}

func (h *Harness) CheckModelCommitted(cmd []byte) (nc int, index int) {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Find the length of the commits slice for connected servers.
	commitsLen := -1
	for i := 0; i < h.n; i++ {
		if h.connected[i] {
			if commitsLen >= 0 {
				// If this was set already, expect the new length to be the same.
				if len(h.commits[i]) != commitsLen {
					h.t.Fatalf("commits[%d] = %d, commitsLen = %d", i, h.commits[i], commitsLen)
				}
			} else {
				commitsLen = len(h.commits[i])
			}
		}
	}

	// Check consistency of commits from the start and to the command we're asked
	// about. This loop will return once a command=cmd is found.
	for c := 0; c < commitsLen; c++ {
		cmdAtC := []byte("")
		for i := 0; i < h.n; i++ {
			if h.connected[i] {
				cmdOfN := []byte(fmt.Sprintf("%s", h.commits[i][c].Command))
				if len(cmdAtC) >= 0 {
					if reflect.DeepEqual(cmdOfN, cmdAtC) {
						h.t.Errorf("got %d, want %d at h.commits[%d][%d]", cmdOfN, cmdAtC, i, c)
					}
				} else {
					cmdAtC = cmdOfN
				}
			}
		}

		if reflect.DeepEqual(cmdAtC, cmd) {
			// Check consistency of Index.
			index := -1
			nc := 0
			for i := 0; i < h.n; i++ {
				if h.connected[i] {
					if index >= 0 && h.commits[i][c].Index != index {
						h.t.Errorf("got Index=%d, want %d at h.commits[%d][%d]", h.commits[i][c].Index, index, i, c)
					} else {
						index = h.commits[i][c].Index
					}
					nc++
				}
			}
			return nc, index
		}
	}

	// If there's no early return, we haven't found the command we were looking
	// for.
	h.t.Errorf("cmd=%d not found in commits", cmd)
	return -1, -1
}

func (h *Harness) CheckModelCommittedN(cmd []byte, n int) {
	nc, _ := h.CheckModelCommitted(cmd)
	if nc != n {
		h.t.Errorf("CheckCommittedN got nc=%d, want %d", nc, n)
	}
}

func (h *Harness) CheckModelNotCommitted(cmd []byte) {
	h.mu.Lock()
	defer h.mu.Unlock()

	for i := 0; i < h.n; i++ {
		if h.connected[i] {
			for c := 0; c < len(h.commits[i]); c++ {
				gotCmd := []byte(fmt.Sprintf("%s", h.commits[i][c].Command))

				if reflect.DeepEqual(gotCmd, cmd) {
					h.t.Errorf("found %d at commits[%d][%d], expected none", cmd, i, c)
				}
			}
		}
	}
}

// SubmitToServer submits the command to serverId.
func (h *Harness) SubmitToServer(serverId int, cmd interface{}) bool {
	return h.cluster[serverId].cm.Submit(cmd)
}

func tlog(format string, a ...interface{}) {
	format = "[TEST] " + format
	log.Printf(format, a...)
}

func sleepMs(n int) {
	time.Sleep(time.Duration(n) * time.Millisecond)
}

// collectCommits reads channel commitChans[i] and adds all received entries
// to the corresponding commits[i]. It's blocking and should be run in a
// separate goroutine. It returns when commitChans[i] is closed.
func (h *Harness) collectCommits(i int) {
	// 受け取ったchannelをずっと
	for c := range h.commitChans[i] {
		h.mu.Lock()
		tlog("collectCommits(%d) got %+v", i, c)
		h.commits[i] = append(h.commits[i], c)
		h.mu.Unlock()
	}
}
