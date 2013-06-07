package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"path"
	"strings"
)

type GPU_Getter interface {
	GetGPU() (q *data.Slice, recycle bool) // get quantity data (GPU), indicate need to recycle
}

type Getter interface {
	Get() (q *data.Slice, recycle bool) // get quantity data (GPU or CPU), indicate need to recycle
}

type Saver interface {
	Getter
	autoFname() string // file name for autosave
	needSave() bool
	saved()
}

// Download a quantity to host,
// or just return its data when already on host.
func Download(q Getter) *data.Slice {
	buf, recycle := q.Get()
	if buf.CPUAccess() {
		util.Assert(recycle == false)
		return buf
	}
	h := hostBuf(buf.NComp(), buf.Mesh())
	data.Copy(h, buf)
	if recycle {
		cuda.RecycleBuffer(buf)
	}
	return h
}

func hostBuf(nComp int, m *data.Mesh) *data.Slice {
	return data.NewSlice(nComp, m) // TODO use pool of page-locked buffers
}

func Save(q Saver) {
	SaveAs(q, q.autoFname())
}

func SaveAs(q Getter, fname string) {
	if !path.IsAbs(fname) && !strings.HasPrefix(fname, OD) {
		fname = path.Clean(OD + "/" + fname)
	}
	if path.Ext(fname) == "" {
		fname += ".dump"
	}
	if s, ok := q.(GPU_Getter); ok {
		buffer, recylce := s.GetGPU()
		if recylce {
			defer cuda.RecycleBuffer(buffer)
		}
		goSaveCopy(fname, buffer, Time)
	} else {
		h, recycle := q.Get()
		util.Assert(recycle == false)
		data.MustWriteFile(fname, h, Time) // not async, but only for stuff already on CPU. could be improved
	}
}

type Autosaver interface {
	Autosave(period float64)
}

func Autosave(what Autosaver, period float64) {
	what.Autosave(period)
}

//func ReadFile(fname string)*data.Slice{
//	s, _ := data.MustReadFile(fname)
//	return s
//}

func init() {
	world.Func("save", Save)
	world.Func("saveas", SaveAs)
	world.Func("autosave", Autosave)
	//world.Func("readfile", ReadFile)
}

// notify that it may need to be saved.
func notifySave(q Saver, goodstep bool) {
	if goodstep && q.needSave() {
		Save(q)
		q.saved()
	}
}