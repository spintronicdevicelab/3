package ptx

//This file is auto-generated. Editing is futile.

func init() { Code["reducesum"] = REDUCESUM }

const REDUCESUM = `
//
// Generated by NVIDIA NVVM Compiler
// Compiler built on Sat Sep 22 02:35:14 2012 (1348274114)
// Cuda compilation tools, release 5.0, V0.2.1221
//

.version 3.1
.target sm_30
.address_size 64

	.file	1 "/tmp/tmpxft_00001600_00000000-9_reducesum.cpp3.i"
	.file	2 "/home/arne/src/code.google.com/p/nimble-cube/gpu/ptx/reducesum.cu"
	.file	3 "/usr/local/cuda-5.0/nvvm/ci_include.h"
	.file	4 "/usr/local/cuda/bin/../include/sm_20_atomic_functions.h"
// __cuda_local_var_33841_32_non_const_sdata has been demoted

.visible .entry reducesum(
	.param .u64 reducesum_param_0,
	.param .u64 reducesum_param_1,
	.param .u32 reducesum_param_2
)
{
	.reg .pred 	%p<8>;
	.reg .s32 	%r<38>;
	.reg .f32 	%f<33>;
	.reg .s64 	%rd<13>;
	// demoted variable
	.shared .align 4 .b8 __cuda_local_var_33841_32_non_const_sdata[2048];

	ld.param.u64 	%rd4, [reducesum_param_0];
	ld.param.u64 	%rd5, [reducesum_param_1];
	ld.param.u32 	%r9, [reducesum_param_2];
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd4;
	.loc 2 11 1
	mov.u32 	%r37, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r2, %tid.x;
	mad.lo.s32 	%r36, %r37, %r10, %r2;
	mov.u32 	%r11, %nctaid.x;
	mul.lo.s32 	%r4, %r37, %r11;
	.loc 2 11 1
	setp.ge.s32 	%p1, %r36, %r9;
	mov.f32 	%f31, 0f00000000;
	mov.f32 	%f32, %f31;
	@%p1 bra 	BB0_2;

BB0_1:
	.loc 2 11 1
	mul.wide.s32 	%rd6, %r36, 4;
	add.s64 	%rd7, %rd2, %rd6;
	ld.global.f32 	%f6, [%rd7];
	add.f32 	%f32, %f32, %f6;
	add.s32 	%r36, %r36, %r4;
	.loc 2 11 1
	setp.lt.s32 	%p2, %r36, %r9;
	mov.f32 	%f31, %f32;
	@%p2 bra 	BB0_1;

BB0_2:
	.loc 2 11 1
	mul.wide.s32 	%rd8, %r2, 4;
	mov.u64 	%rd9, __cuda_local_var_33841_32_non_const_sdata;
	add.s64 	%rd3, %rd9, %rd8;
	st.shared.f32 	[%rd3], %f31;
	bar.sync 	0;
	.loc 2 11 1
	setp.lt.u32 	%p3, %r37, 66;
	@%p3 bra 	BB0_6;

BB0_3:
	.loc 2 11 1
	mov.u32 	%r7, %r37;
	shr.u32 	%r37, %r7, 1;
	.loc 2 11 1
	setp.ge.u32 	%p4, %r2, %r37;
	@%p4 bra 	BB0_5;

	.loc 2 11 1
	ld.shared.f32 	%f7, [%rd3];
	add.s32 	%r15, %r37, %r2;
	mul.wide.u32 	%rd10, %r15, 4;
	add.s64 	%rd12, %rd9, %rd10;
	ld.shared.f32 	%f8, [%rd12];
	add.f32 	%f9, %f7, %f8;
	st.shared.f32 	[%rd3], %f9;

BB0_5:
	.loc 2 11 1
	bar.sync 	0;
	.loc 2 11 1
	setp.gt.u32 	%p5, %r7, 131;
	@%p5 bra 	BB0_3;

BB0_6:
	.loc 2 11 1
	setp.gt.s32 	%p6, %r2, 31;
	@%p6 bra 	BB0_8;

	.loc 2 11 1
	ld.volatile.shared.f32 	%f10, [%rd3];
	ld.volatile.shared.f32 	%f11, [%rd3+128];
	add.f32 	%f12, %f10, %f11;
	st.volatile.shared.f32 	[%rd3], %f12;
	ld.volatile.shared.f32 	%f13, [%rd3+64];
	ld.volatile.shared.f32 	%f14, [%rd3];
	add.f32 	%f15, %f14, %f13;
	st.volatile.shared.f32 	[%rd3], %f15;
	ld.volatile.shared.f32 	%f16, [%rd3+32];
	ld.volatile.shared.f32 	%f17, [%rd3];
	add.f32 	%f18, %f17, %f16;
	st.volatile.shared.f32 	[%rd3], %f18;
	ld.volatile.shared.f32 	%f19, [%rd3+16];
	ld.volatile.shared.f32 	%f20, [%rd3];
	add.f32 	%f21, %f20, %f19;
	st.volatile.shared.f32 	[%rd3], %f21;
	ld.volatile.shared.f32 	%f22, [%rd3+8];
	ld.volatile.shared.f32 	%f23, [%rd3];
	add.f32 	%f24, %f23, %f22;
	st.volatile.shared.f32 	[%rd3], %f24;
	ld.volatile.shared.f32 	%f25, [%rd3+4];
	ld.volatile.shared.f32 	%f26, [%rd3];
	add.f32 	%f27, %f26, %f25;
	st.volatile.shared.f32 	[%rd3], %f27;

BB0_8:
	.loc 2 11 1
	setp.ne.s32 	%p7, %r2, 0;
	@%p7 bra 	BB0_10;

	.loc 2 11 1
	ld.shared.f32 	%f28, [__cuda_local_var_33841_32_non_const_sdata];
	.loc 3 1844 5
	atom.global.add.f32 	%f29, [%rd1], %f28;

BB0_10:
	.loc 2 12 2
	ret;
}


`
