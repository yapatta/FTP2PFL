package raft

import (
	"bytes"
	"encoding/gob"
)

type Model struct {
	Buf string
}

func (m *Model) Encode() ([]byte, error) {
	buf := bytes.NewBuffer(nil)

	_ := gob.NewEncoder(buf).Encode(m.Buf)
	return buf.Bytes(), nil
}
