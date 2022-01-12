package raft

// #cgo CFLAGS: -I/usr/include/python3.9
// #cgo LDFLAGS: -lpython3.9
// #include <Python.h>
import "C"
import (
	"fmt"
	"unsafe"
)

func SetPath() {
	sys := C.PyImport_ImportModule(C.CString("sys"))
	// GoのstringをCのcharに型変換（変換しないとPyRun_SimpleStringに型合ってないよって怒られる）
	sys_path := C.PyObject_GetAttrString(sys, C.CString("path"))

	cdir1 := C.CString("../python/venv/lib/python3.9/site-packages")
	dir1 := C.PyUnicode_FromString(cdir1)
	C.PyList_Append(sys_path, dir1)

	cdir2 := C.CString("../python/")
	dir2 := C.PyUnicode_FromString(cdir2)
	C.PyList_Append(sys_path, dir2)

	C.free(unsafe.Pointer(cdir1))
	C.free(unsafe.Pointer(cdir2))
}

func ExecString(s string) int {
	c := C.CString(s)
	defer C.free(unsafe.Pointer(c))

	ret := C.PyRun_SimpleString(c)
	return int(ret)
}

func ClientLearn(id int) bool {
	pythonStr := fmt.Sprintf("import client\nclient.learn(%v)", id)
	if c := ExecString(pythonStr); c != 0 {
		return false
		// log.Fatalln("learn() in client.py failed")
	}

	return true
}

func ServerAggregate() bool {
	pythonStr := "import aggregator as ag\nag.aggregate()"
	if c := ExecString(pythonStr); c != 0 {
		return false
		// log.Fatalln("aggregate() in aggregator.py failed")
	}

	return true
}

func ServerInitialLearn() bool {
	pythonStr := "import aggregator as ag\nag.initial_learn()"
	if c := ExecString(pythonStr); c != 0 {
		return false
		// log.Fatalln("initial_learn() in aggregator.py failed")
	}
	return true
}

// For test: Success
func ExecFLManually() bool {
	C.Py_Initialize()
	defer C.Py_Finalize()

	SetPath()

	// Initial Learn
	if !ServerInitialLearn() {
		// log.Fatalln("initial_learn() in aggregator.py failed")
		return false
	}

	// Client Learn
	for i := 0; i < 3; i++ {
		if !ClientLearn(i) {
			// log.Fatalln("learn() in client.py failed")
			return false
		}
	}

	if !ServerAggregate() {
		// log.Fatalln("aggregate() in aggregator.py failed")
		return false
	}

	return true
}
