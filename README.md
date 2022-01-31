# 耐障害性を持ったモバイル向けP2P連合学習システム

## 趣旨

yapattaが卒論研究で実装したもの. 分散合意アルゴリズムRaft[^https://raft.github.io/raft.pdf]を用いることで, 集計サーバがダウンしても他のノードが集計サーバになれば固定の集計サーバを持たずにかつ耐障害性を持った状態で連合学習[^https://arxiv.org/abs/1602.05629]を継続できるのでは？という発想のもと生まれたシステム. モバイル向けと謳っているだけあって、バッテリ残量のシミューレーションが入っており, バッテリー残量が一定以上のノードが集計サーバに選ばれる.  

## install

```sh
$ cd ../python
$ python3 -m venv [newenvname]
$ source ./[newenvname]/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Test

### 1. APIサーバ起動

```sh
$ python3 ./python/server.py
```

### 2. Goのテスト実行

下のように実行すると, HTMLファイルに実行結果が記録される. 

```go
$ cd ./go
$ go test -v -race -run TestFLBasic |& tee /tmp/raftlog
... logging output
... test should PASS
$ go run ./tools/raft-testlog-viz/main.go < /tmp/raftlog
PASS TestFLBasic map[0:true 1:true 2:true TEST:true] ; entries: 150
... Emitted file:///tmp/TestElectionFollowerComesBack.html

PASS
```

もしくは

```json:basic.json
{"testname": "TestFLBasic", "counts": "4", "sec": "180"}
```

のようにjsonファイルにTest名, 評価試行回数, 評価時間を指定して

```go
$ cd ./go
$ ./eval_all.sh basic.json
```

と実行すると`./go/data/[testname]/[ノード数]-[時間]/[1-9][0-9]+.csv`に評価結果が出力される. モデルが集計された時間のみが記録される. 

その後, 

```sh
$ cd ./data
$ python3 accuracy.py TestFLBasic 180
$ python3 loss.py TestFLBasic 180
```

と実行すると, `go/data/accuracy/TestFLBasic-180-accuracy.pdf`, `go/data/loss/TestFLBasic-180-loss.pdf`のようなファイルが生成される. これらの画像は連合学習におけるモデルの精度, 損失をグラフ化したものである. 

!["TestFLBasic-180-accuracy.pdf"]("./go/data/accuracy/TestFLBasic-180-accuracy.pdf", "TestFLBasic-180-accuracy.pdf")

![TestFLBasic-180-loss.pdf]("./go/data/loss/TestFLBasic-180-loss.pdf
", "TestFLBasic-180-loss.pdf")
