package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
	"math"
)

// Adaptive Heun solver.
type Heun struct{}

// Adaptive Heun method, can be used as solver.Step
func (_ *Heun) Step() {
	y := M.Buffer()
	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)

	if FixDt != 0 {
		Dt_si = FixDt
	}

	dt := float32(Dt_si * GammaLL)
	util.Assert(dt > 0)

	// stage 1
	torqueFn(dy0)
	cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy

	// stage 2
	dy := cuda.Buffer(3, y.Size())
	defer cuda.Recycle(dy)
	Time += Dt_si
	torqueFn(dy)

	err := cuda.MaxVecDiff(dy0, dy) * float64(dt)
	errnorm := cuda.Buffer(1, y.Size())

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// Passed absolute error. Check relative error...
		errVec := cuda.Buffer(3, y.Size())
		defer cuda.Recycle(errVec)
		defer cuda.Recycle(errnorm)
		cuda.Madd2(errVec,dy0,dy,1,-1.0)
		cuda.VecNorm(errnorm,errVec)
		ddtnorm := cuda.Buffer(1, y.Size())
		defer cuda.Recycle(ddtnorm)
		cuda.VecNorm(ddtnorm, dy)
		maxdm := cuda.MaxVecNorm(dy)
		fail := 0
		rlerr := float64(0.0)
		if maxdm < MinSlope { // Only step using relerr if dmdt is big enough. Overcomes equilibrium problem
			fail = 0
		} else {
			cuda.Divide(errnorm, errnorm, ddtnorm)   //re-use errnorm
			rlerr = float64(cuda.MaxAbs(errnorm))
			fail = 1
		}
		if fail == 0 || RelErr < 0.0 || rlerr < RelErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
			// step OK
			cuda.Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
			M.normalize()
			NSteps++
			if fail == 0 {
				adaptDt(math.Pow(MaxErr/err, 1./2.))
			} else {
				adaptDt(math.Pow(RelErr/rlerr, 1./2.))
			}
			setLastErr(err)
			setMaxTorque(dy)
		} else {
			// undo bad step
			util.Assert(FixDt == 0)
			Time -= Dt_si
			cuda.Madd2(y, y, dy0, 1, -dt)
			NUndone++
			adaptDt(math.Pow(RelErr/rlerr, 1./3.))
		}
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time -= Dt_si
		cuda.Madd2(y, y, dy0, 1, -dt)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./3.))
	}
}

func (_ *Heun) Free() {}
