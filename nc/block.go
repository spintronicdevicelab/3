package nc

import ()

// Block is a [][][]float32 with square layout and contiguous underlying storage:
// 	len(block[0]) == len(block[1]) == ...
// 	len(block[i][0]) == len(block[i][1]) == ...
// 	len(block[i][j][0]) == len(block[i][j][1]) == ...
type Block [][][]float32

// Make a block of float32's of size N[0] x N[1] x N[2].
func MakeBlock(N [3]int) Block {
	checkSize(N[:])
	sliced := make([][][]float32, N[0])
	for i := range sliced {
		sliced[i] = make([][]float32, N[1])
	}
	storage := make([]float32, N[0]*N[1]*N[2])
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = storage[(i*N[1]+j)*N[2]+0 : (i*N[1]+j)*N[2]+N[2]]
		}
	}
	return Block(sliced)
}

// Total number of scalar elements.
func (b Block) NFloat() int {
	return len(b) * len(b[0]) * len(b[0][0])
}

// BlockSize is the size of the block (N0, N1, N2)
// as was passed to MakeBlock()
func (b Block) BlockSize() [3]int {
	return [3]int{len(b), len(b[0]), len(b[0][0])}
}

// Returns the contiguous underlying storage.
// Contains first all X component, than Y, than Z.
func (b Block) Contiguous() []float32 {
	return ([][][]float32)(b)[0][0][:b.NFloat()]
}

// Set all elements to a.
//func (b Block) Memset(a float32) {
//	storage := b.Contiguous()
//	for i := range storage {
//		storage[i] = a
//	}
//}

// Check if all sizes are > 0
func checkSize(size []int) {
	for i, s := range size {
		if s < 1 {
			Panic("MakeBlock: N", i, " out of range:", s)
		}
	}
}

func (b Block) Range(i1, i2 int) []float32 {
	return b.Contiguous()[i1:i2]
}
