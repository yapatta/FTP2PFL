package raft

import "os"

// TODO
/**
- tensorflowが作成したhd5ファイル(重みが保存)を読み込み(f)
-  読み取ったバイナリをraftで送りつける
- データ構造としてGoでは扱わない!!
**/

func ReadModel(fn string) ([]byte, error) {
	return os.ReadFile(fn)
}

/*
type Model struct {
	Buf []byte
}

func (m *Model) Encode() ([]byte, error) {
	buf := bytes.NewBuffer(nil)

	if err := gob.NewEncoder(buf).Encode(m); err != nil {
		return nil, err
	}

	m.Buf = buf.Bytes()

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
*/
