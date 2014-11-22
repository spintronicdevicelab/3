package cuda

import (
	"github.com/mumax/3/data"
	"unsafe"
)

// Add Slonczewski ST torque to torque (Tesla).
// see slonczewski.cu
func AddOommfSlonczewskiTorque(torque, m, J *data.Slice, fixedP LUTPtrs, Msat, alpha, pfix, pfree, Î»fix, Î»free, Îµ_prime LUTPtr, regions *Bytes, mesh *data.Mesh) {
	N := torque.Len()
	cfg := make1DConf(N)
	thickness := float32(mesh.WorldSize()[Z])

	k_addoommfslonczewskitorque_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z), J.DevPtr(Z),
		fixedP[X], fixedP[Y], fixedP[Z],
		unsafe.Pointer(Msat), unsafe.Pointer(alpha),
		thickness, unsafe.Pointer(pfix), unsafe.Pointer(pfree),
		unsafe.Pointer(Î»fix), unsafe.Pointer(Î»free), unsafe.Pointer(Îµ_prime),
		regions.Ptr, N, cfg)
}
