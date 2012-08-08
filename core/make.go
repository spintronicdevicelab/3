package core

func MakeVectors(size [3]int) [3][][][]float32 {
	return [3][][][]float32{MakeFloats(size), MakeFloats(size), MakeFloats(size)}
}

func MakeFloats(size [3]int) [][][]float32 {
	storage := make([]float32, Prod(size))
	return Reshape(storage, size)
}
