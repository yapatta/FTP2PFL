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

	if err := gob.NewEncoder(buf).Encode(m); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func (m *Model) Decode(data []byte) error {
	buf := bytes.NewBuffer(data)
	var aux Model
	if err := gob.NewDecoder(buf).Decode(&aux); err != nil {
		return err
	}
	m = &aux

	return nil
}
