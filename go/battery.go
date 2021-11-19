package raft

import (
	"log"
	"math"
	"math/rand"
	"sync"
	"time"
)

type Battery struct {
	mu       sync.Mutex
	percent  int // 0..100
	onCharge bool
	shutdown chan struct{}
	live     bool
}

func NewBattery() *Battery {
	b := new(Battery)
	b.percent = initialPercent()
	b.onCharge = false
	b.live = true
	b.shutdown = make(chan struct{})

	return b
}

func (b *Battery) Enough() bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	return b.live && (b.onCharge || b.percent >= 20)
}

func (b *Battery) Stop() {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.live = false
	close(b.shutdown)
}

func (b *Battery) NormalAction() {
	for {
		b.mu.Lock()

		if b.onCharge {
			b.percent++
			if b.percent >= 100 {
				time.Sleep(500 * time.Millisecond)
				b.onCharge = false
			}
		} else {
			b.percent--
			if b.percent < 10 {
				time.Sleep(600 * time.Millisecond)
				b.onCharge = true
			}
		}

		if !b.live {
			select {
			case <-b.shutdown:
				return
			default:
				log.Fatal("accept error")
			}
		}

		log.Printf("battery: %v", b.percent)

		time.Sleep(200 * time.Millisecond)
		b.mu.Unlock()
	}
}

func initialPercent() int {
	f := math.Abs(rand.NormFloat64())
	ret := int(100*f + 20)
	if ret > 100 {
		ret = 100
	}

	return ret
}
