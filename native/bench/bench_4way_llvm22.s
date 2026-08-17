	.file	"bench_4way.c"
	.text
	.globl	main                            // -- Begin function main
	.p2align	2
	.type	main,@function
main:                                   // @main
	.cfi_startproc
// %bb.0:
	str	d8, [sp, #-96]!                 // 8-byte Folded Spill
	.cfi_def_cfa_offset 96
	stp	x29, x30, [sp, #8]              // 16-byte Folded Spill
	str	x28, [sp, #24]                  // 8-byte Spill
	stp	x26, x25, [sp, #32]             // 16-byte Folded Spill
	stp	x24, x23, [sp, #48]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #64]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #80]             // 16-byte Folded Spill
	add	x29, sp, #8
	.cfi_def_cfa w29, 88
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w25, -56
	.cfi_offset w26, -64
	.cfi_offset w28, -72
	.cfi_offset w30, -80
	.cfi_offset w29, -88
	.cfi_offset b8, -96
	sub	sp, sp, #256, lsl #12           // =1048576
	sub	sp, sp, #128
	movi	v0.2d, #0000000000000000
	sub	x8, x29, #136
	mov	w9, #2                          // =0x2
	sub	x2, x29, #136
	mov	w0, wzr
	mov	w1, #128                        // =0x80
	str	xzr, [x8, #120]
	str	x9, [x8]
	stur	q0, [x8, #104]
	stur	q0, [x8, #88]
	stur	q0, [x8, #72]
	stur	q0, [x8, #56]
	stur	q0, [x8, #40]
	stur	q0, [x8, #24]
	stur	q0, [x8, #8]
	bl	sched_setaffinity
	mov	w0, #64                         // =0x40
	mov	w1, #524288                     // =0x80000
	bl	aligned_alloc
	mov	x19, x0
	mov	w0, #64                         // =0x40
	mov	w1, #524288                     // =0x80000
	bl	aligned_alloc
	cntd	x8
	index	z0.d, #0, #1
	cntw	x9
	mov	z1.d, x8
	mov	w8, #52501                      // =0xcd15
	mov	z2.d, x9
	movk	w8, #1883, lsl #16
	mov	z4.d, #1                        // =0x1
	ptrue	p1.d
	mov	z3.d, x8
	ptrue	p0.b
	mov	x20, x0
	mov	x8, xzr
.LBB0_1:                                // =>This Inner Loop Header: Depth=1
	movprfx	z6, z4
	mla	z6.d, p1/m, z0.d, z3.d
	add	z5.d, z0.d, z1.d
	add	x9, x19, x8
	add	z0.d, z0.d, z2.d
	mad	z5.d, p1/m, z3.d, z4.d
	st1b	{ z6.b }, p0, [x19, x8]
	incb	x8, all, mul #2
	cmp	x8, #128, lsl #12               // =524288
	str	z5, [x9, #1, mul vl]
	b.ne	.LBB0_1
// %bb.2:
	mov	x21, #57072                     // =0xdef0
	adrp	x0, .Lstr
	add	x0, x0, :lo12:.Lstr
	movk	x21, #39612, lsl #16
	movk	x21, #22136, lsl #32
	movk	x21, #4660, lsl #48
	bl	puts
	cntd	x22
	rdvl	x3, #8
	adrp	x0, .L.str.5
	add	x0, x0, :lo12:.L.str.5
	mov	w1, #65536                      // =0x10000
	mov	w2, #1000                       // =0x3e8
	mov	w4, w22
                                        // kill: def $w3 killed $w3 killed $x3
	mov	w23, #1000                      // =0x3e8
	bl	printf
	mov	x0, x19
	mov	x1, x20
	mov	x2, x21
	mov	x3, x21
	mov	x4, #-59                        // =0xffffffffffffffc5
	mov	w5, #65536                      // =0x10000
	bl	scalar_schoolbook
	//APP
	mrs	x24, CNTVCT_EL0
	//NO_APP
.LBB0_3:                                // =>This Inner Loop Header: Depth=1
	mov	x0, x19
	mov	x1, x20
	mov	x2, x21
	mov	x3, x21
	mov	x4, #-59                        // =0xffffffffffffffc5
	mov	w5, #65536                      // =0x10000
	bl	scalar_schoolbook
	subs	w23, w23, #1
	b.ne	.LBB0_3
// %bb.4:
	//APP
	mrs	x8, CNTVCT_EL0
	//NO_APP
	fmov	d1, #23.00000000
	adrp	x0, .L.str.6
	add	x0, x0, :lo12:.L.str.6
	sub	x8, x8, x24
	adrp	x1, .L.str
	add	x1, x1, :lo12:.L.str
	ucvtf	d0, x8
	mov	x8, #70368744177664             // =0x400000000000
	movk	x8, #16783, lsl #48
	fmul	d0, d0, d1
	fmov	d1, x8
	fdiv	d8, d0, d1
	fdiv	d1, d8, d8
	fmov	d0, d8
	bl	printf
	mov	x0, x19
	mov	x1, x20
	mov	x2, x21
	mov	x3, x21
	mov	x4, #-59                        // =0xffffffffffffffc5
	mov	w5, #65536                      // =0x10000
	bl	scalar_int128
	mov	w24, #1000                      // =0x3e8
	//APP
	mrs	x23, CNTVCT_EL0
	//NO_APP
.LBB0_5:                                // =>This Inner Loop Header: Depth=1
	mov	x0, x19
	mov	x1, x20
	mov	x2, x21
	mov	x3, x21
	mov	x4, #-59                        // =0xffffffffffffffc5
	mov	w5, #65536                      // =0x10000
	bl	scalar_int128
	subs	w24, w24, #1
	b.ne	.LBB0_5
// %bb.6:
	//APP
	mrs	x8, CNTVCT_EL0
	//NO_APP
	fmov	d1, #23.00000000
	adrp	x0, .L.str.6
	add	x0, x0, :lo12:.L.str.6
	sub	x8, x8, x23
	adrp	x1, .L.str.1
	add	x1, x1, :lo12:.L.str.1
	ucvtf	d0, x8
	mov	x8, #70368744177664             // =0x400000000000
	movk	x8, #16783, lsl #48
	fmul	d0, d0, d1
	fmov	d1, x8
	fdiv	d0, d0, d1
	fdiv	d1, d8, d0
	bl	printf
	mov	x0, x19
	mov	x1, x20
	mov	x2, x21
	mov	x3, x21
	mov	x4, #-59                        // =0xffffffffffffffc5
	mov	w5, #65536                      // =0x10000
	bl	vector_schoolbook
	mov	w24, #1000                      // =0x3e8
	//APP
	mrs	x23, CNTVCT_EL0
	//NO_APP
.LBB0_7:                                // =>This Inner Loop Header: Depth=1
	mov	x0, x19
	mov	x1, x20
	mov	x2, x21
	mov	x3, x21
	mov	x4, #-59                        // =0xffffffffffffffc5
	mov	w5, #65536                      // =0x10000
	bl	vector_schoolbook
	subs	w24, w24, #1
	b.ne	.LBB0_7
// %bb.8:
	//APP
	mrs	x8, CNTVCT_EL0
	//NO_APP
	fmov	d1, #23.00000000
	adrp	x0, .L.str.6
	add	x0, x0, :lo12:.L.str.6
	sub	x8, x8, x23
	adrp	x1, .L.str.2
	add	x1, x1, :lo12:.L.str.2
	ucvtf	d0, x8
	mov	x8, #70368744177664             // =0x400000000000
	movk	x8, #16783, lsl #48
	fmul	d0, d0, d1
	fmov	d1, x8
	fdiv	d0, d0, d1
	fdiv	d1, d8, d0
	bl	printf
	mov	x0, x19
	mov	x1, x20
	mov	x2, x21
	mov	x3, x21
	mov	x4, #-59                        // =0xffffffffffffffc5
	mov	w5, #65536                      // =0x10000
	bl	vector_int128
	mov	w24, #1000                      // =0x3e8
	//APP
	mrs	x23, CNTVCT_EL0
	//NO_APP
.LBB0_9:                                // =>This Inner Loop Header: Depth=1
	mov	x0, x19
	mov	x1, x20
	mov	x2, x21
	mov	x3, x21
	mov	x4, #-59                        // =0xffffffffffffffc5
	mov	w5, #65536                      // =0x10000
	bl	vector_int128
	subs	w24, w24, #1
	b.ne	.LBB0_9
// %bb.10:
	//APP
	mrs	x8, CNTVCT_EL0
	//NO_APP
	fmov	d1, #23.00000000
	adrp	x0, .L.str.6
	add	x0, x0, :lo12:.L.str.6
	sub	x8, x8, x23
	adrp	x1, .L.str.3
	add	x1, x1, :lo12:.L.str.3
	ucvtf	d0, x8
	mov	x8, #70368744177664             // =0x400000000000
	movk	x8, #16783, lsl #48
	fmul	d0, d0, d1
	fmov	d1, x8
	fdiv	d0, d0, d1
	fdiv	d1, d8, d0
	bl	printf
	add	x1, sp, #128, lsl #12           // =524288
	mov	x0, x19
	mov	x2, x21
	mov	x3, x21
	mov	x4, #-59                        // =0xffffffffffffffc5
	mov	w5, #65536                      // =0x10000
	add	x25, sp, #128, lsl #12          // =524288
	bl	scalar_int128
	mov	x1, sp
	mov	x0, x19
	mov	x2, x21
	mov	x3, x21
	mov	x4, #-59                        // =0xffffffffffffffc5
	mov	w5, #65536                      // =0x10000
	mov	x26, sp
	bl	scalar_schoolbook
	rdvl	x8, #1
	neg	x24, x22
	ptrue	p1.d
	lsr	x23, x8, #4
	mov	x9, xzr
	add	x10, x24, #16, lsl #12          // =65536
	mov	x8, xzr
.LBB0_11:                               // =>This Inner Loop Header: Depth=1
	ld1d	{ z0.d }, p1/z, [x26, x9, lsl #3]
	ld1d	{ z1.d }, p1/z, [x25, x9, lsl #3]
	incd	x8
	cmpne	p0.d, p1/z, z0.d, z1.d
	b.ne	.LBB0_13
// %bb.12:                              //   in Loop: Header=BB0_11 Depth=1
	cmp	x10, x9
	mov	x9, x8
	b.ne	.LBB0_11
.LBB0_13:
	ptrue	p2.d
	cmpeq	p1.d, p1/z, z0.d, z1.d
	ptest	p2, p0.b
	b.eq	.LBB0_15
// %bb.14:
	brkb	p0.b, p2/z, p0.b
	mov	z0.d, p1/z, #1                  // =0x1
	adrp	x0, .L.str.10
	add	x0, x0, :lo12:.L.str.10
	adrp	x1, .L.str.7
	add	x1, x1, :lo12:.L.str.7
	cntp	x9, p0, p0.d
	whilels	p0.d, xzr, x9
	sub	w9, w9, w23, lsl #1
	add	w2, w9, w8
	lastb	x22, p0, z0.d
	bl	printf
	b	.LBB0_16
.LBB0_15:
	ptrue	p0.d
	ptest	p0, p1.b
	cset	w22, lo
.LBB0_16:
	mov	x1, sp
	mov	x0, x19
	mov	x2, x21
	mov	x3, x21
	mov	x4, #-59                        // =0xffffffffffffffc5
	mov	w5, #65536                      // =0x10000
	mov	x25, sp
	bl	vector_schoolbook
	ptrue	p1.d
	mov	x10, xzr
	add	x9, x24, #16, lsl #12           // =65536
	add	x11, sp, #128, lsl #12          // =524288
	mov	x8, xzr
.LBB0_17:                               // =>This Inner Loop Header: Depth=1
	ld1d	{ z0.d }, p1/z, [x25, x10, lsl #3]
	ld1d	{ z1.d }, p1/z, [x11, x10, lsl #3]
	incd	x8
	cmpne	p0.d, p1/z, z0.d, z1.d
	b.ne	.LBB0_19
// %bb.18:                              //   in Loop: Header=BB0_17 Depth=1
	cmp	x9, x10
	mov	x10, x8
	b.ne	.LBB0_17
.LBB0_19:
	ptrue	p1.d
	ptest	p1, p0.b
	b.eq	.LBB0_21
// %bb.20:
	brkb	p0.b, p1/z, p0.b
	adrp	x0, .L.str.10
	add	x0, x0, :lo12:.L.str.10
	adrp	x1, .L.str.8
	add	x1, x1, :lo12:.L.str.8
	cntp	x9, p0, p0.d
	sub	w9, w9, w23, lsl #1
	add	w2, w9, w8
	bl	printf
	adrp	x22, .L.str.13
	add	x22, x22, :lo12:.L.str.13
	b	.LBB0_22
.LBB0_21:
	adrp	x8, .L.str.13
	add	x8, x8, :lo12:.L.str.13
	tst	w22, #0x1
	adrp	x9, .L.str.12
	add	x9, x9, :lo12:.L.str.12
	csel	x22, x9, x8, ne
.LBB0_22:
	mov	x1, sp
	mov	x0, x19
	mov	x2, x21
	mov	x3, x21
	mov	x4, #-59                        // =0xffffffffffffffc5
	mov	w5, #65536                      // =0x10000
	mov	x21, sp
	bl	vector_int128
	ptrue	p1.d
	mov	x10, xzr
	add	x9, x24, #16, lsl #12           // =65536
	add	x11, sp, #128, lsl #12          // =524288
	mov	x8, xzr
.LBB0_23:                               // =>This Inner Loop Header: Depth=1
	ld1d	{ z0.d }, p1/z, [x21, x10, lsl #3]
	ld1d	{ z1.d }, p1/z, [x11, x10, lsl #3]
	incd	x8
	cmpne	p0.d, p1/z, z0.d, z1.d
	b.ne	.LBB0_25
// %bb.24:                              //   in Loop: Header=BB0_23 Depth=1
	cmp	x9, x10
	mov	x10, x8
	b.ne	.LBB0_23
.LBB0_25:
	ptrue	p1.d
	ptest	p1, p0.b
	b.eq	.LBB0_27
// %bb.26:
	brkb	p0.b, p1/z, p0.b
	adrp	x0, .L.str.10
	add	x0, x0, :lo12:.L.str.10
	adrp	x1, .L.str.9
	add	x1, x1, :lo12:.L.str.9
	cntp	x9, p0, p0.d
	sub	w9, w9, w23, lsl #1
	add	w2, w9, w8
	bl	printf
	adrp	x22, .L.str.13
	add	x22, x22, :lo12:.L.str.13
.LBB0_27:
	adrp	x0, .L.str.11
	add	x0, x0, :lo12:.L.str.11
	mov	x1, x22
	bl	printf
	mov	x0, x19
	bl	free
	mov	x0, x20
	bl	free
	mov	w0, wzr
	add	sp, sp, #256, lsl #12           // =1048576
	add	sp, sp, #128
	.cfi_def_cfa wsp, 96
	ldp	x20, x19, [sp, #80]             // 16-byte Folded Reload
	ldr	x28, [sp, #24]                  // 8-byte Reload
	ldp	x22, x21, [sp, #64]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #48]             // 16-byte Folded Reload
	ldp	x26, x25, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #8]              // 16-byte Folded Reload
	ldr	d8, [sp], #96                   // 8-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w25
	.cfi_restore w26
	.cfi_restore w28
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	ret
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        // -- End function
	.p2align	2                               // -- Begin function scalar_schoolbook
	.type	scalar_schoolbook,@function
scalar_schoolbook:                      // @scalar_schoolbook
	.cfi_startproc
// %bb.0:
	cbz	x5, .LBB1_6
// %bb.1:
	cntd	x11
	lsr	x8, x3, #32
	mov	w9, w3
	cmp	x5, x11
	b.lo	.LBB1_3
// %bb.2:
	rdvl	x10, #1
	sub	x12, x1, x0
	cmp	x12, x10
	b.hs	.LBB1_7
.LBB1_3:
	mov	x11, xzr
.LBB1_4:
	lsl	x12, x11, #3
	sub	x11, x5, x11
	mov	x13, #4294967296                // =0x100000000
	add	x10, x1, x12
	add	x12, x0, x12
.LBB1_5:                                // =>This Inner Loop Header: Depth=1
	ldr	x14, [x12], #8
	lsr	x15, x14, #32
	umull	x17, w14, w8
	umull	x18, w14, w9
	umull	x16, w15, w9
	mul	x14, x14, x2
	adds	x16, x16, x17
	lsr	x17, x16, #32
	umaddl	x15, w15, w8, x17
	csel	x17, x13, xzr, hs
	cmn	x18, x16, lsl #32
	adc	x15, x15, x17
	msub	x14, x15, x4, x14
	cmp	x14, x4
	csel	x15, xzr, x4, lo
	subs	x11, x11, #1
	sub	x14, x14, x15
	str	x14, [x10], #8
	b.ne	.LBB1_5
.LBB1_6:
	ret
.LBB1_7:
	mov	z3.d, #0x100000000
	mov	z0.d, x9
	neg	x11, x11
	mov	z1.d, x8
	mov	z2.d, x2
	mov	x10, xzr
	mov	z4.d, x4
	mov	z5.d, #-1                       // =0xffffffffffffffff
	and	x11, x5, x11
	mov	z6.d, #1                        // =0x1
	ptrue	p0.d
.LBB1_8:                                // =>This Inner Loop Header: Depth=1
	ld1d	{ z7.d }, p0/z, [x0, x10, lsl #3]
	lsr	z16.d, z7.d, #32
	mov	z17.d, z7.d
	mul	z7.d, z7.d, z2.d
	mul	z18.d, z16.d, z0.d
	and	z17.d, z17.d, #0xffffffff
	movprfx	z19, z18
	mla	z19.d, p0/m, z17.d, z1.d
	mul	z17.d, z17.d, z0.d
	lsr	z20.d, z19.d, #32
	cmphi	p1.d, p0/z, z18.d, z19.d
	lsl	z18.d, z19.d, #32
	eor	z17.d, z17.d, z5.d
	mad	z16.d, p0/m, z1.d, z20.d
	cmphi	p2.d, p0/z, z18.d, z17.d
	add	z16.d, p1/m, z16.d, z3.d
	add	z16.d, p2/m, z16.d, z6.d
	mls	z7.d, p0/m, z16.d, z4.d
	sub	z16.d, z7.d, z4.d
	umin	z7.d, p0/m, z7.d, z16.d
	st1d	{ z7.d }, p0, [x1, x10, lsl #3]
	incd	x10
	cmp	x11, x10
	b.ne	.LBB1_8
// %bb.9:
	cmp	x5, x11
	b.eq	.LBB1_6
	b	.LBB1_4
.Lfunc_end1:
	.size	scalar_schoolbook, .Lfunc_end1-scalar_schoolbook
	.cfi_endproc
                                        // -- End function
	.p2align	2                               // -- Begin function scalar_int128
	.type	scalar_int128,@function
scalar_int128:                          // @scalar_int128
	.cfi_startproc
// %bb.0:
	cbz	x5, .LBB2_8
// %bb.1:
	cmp	x5, #1
	mov	x8, xzr
	b.eq	.LBB2_6
// %bb.2:
	sub	x9, x1, x0
	cmp	x9, #16
	b.lo	.LBB2_6
// %bb.3:
	dup	v0.2d, x2
	dup	v1.2d, x4
	and	x8, x5, #0xfffffffffffffffe
	ptrue	p0.d, vl2
	mov	x9, x0
	mov	x10, x1
	and	x11, x5, #0xfffffffffffffffe
.LBB2_4:                                // =>This Inner Loop Header: Depth=1
	ldr	q2, [x9], #16
	subs	x11, x11, #2
	mov	x12, v2.d[1]
	fmov	x13, d2
	mul	z2.d, z2.d, z0.d
	umulh	x13, x3, x13
	umulh	x12, x3, x12
	fmov	d4, x13
	fmov	d3, x12
	mov	v4.d[1], v3.d[0]
	mls	z2.d, p0/m, z1.d, z4.d
	sub	v3.2d, v2.2d, v1.2d
	umin	z2.d, p0/m, z2.d, z3.d
	str	q2, [x10], #16
	b.ne	.LBB2_4
// %bb.5:
	cmp	x5, x8
	b.eq	.LBB2_8
.LBB2_6:
	lsl	x10, x8, #3
	sub	x8, x5, x8
	add	x9, x1, x10
	add	x10, x0, x10
.LBB2_7:                                // =>This Inner Loop Header: Depth=1
	ldr	x11, [x10], #8
	umulh	x12, x11, x3
	mul	x11, x11, x2
	msub	x11, x4, x12, x11
	cmp	x11, x4
	csel	x12, xzr, x4, lo
	subs	x8, x8, #1
	sub	x11, x11, x12
	str	x11, [x9], #8
	b.ne	.LBB2_7
.LBB2_8:
	ret
.Lfunc_end2:
	.size	scalar_int128, .Lfunc_end2-scalar_int128
	.cfi_endproc
                                        // -- End function
	.p2align	2                               // -- Begin function vector_schoolbook
	.type	vector_schoolbook,@function
vector_schoolbook:                      // @vector_schoolbook
	.cfi_startproc
// %bb.0:
	cntd	x8
	mov	z2.d, x3
	mov	z1.d, x2
	rbit	x9, x8
	mov	z0.d, x4
	clz	x9, x9
	lsr	z3.d, z2.d, #32
	lsr	x9, x5, x9
	mul	x9, x9, x8
	cbz	x9, .LBB3_6
// %bb.1:
	mov	z4.d, #1                        // =0x1
	ptrue	p0.d
	mov	x8, xzr
.LBB3_2:                                // =>This Inner Loop Header: Depth=1
	ld1d	{ z5.d }, p0/z, [x0, x8, lsl #3]
	lsr	z6.d, z5.d, #32
	umullb	z7.d, z5.s, z3.s
	umullt	z16.d, z5.s, z2.s
	umullb	z6.d, z6.s, z2.s
	add	z7.d, z6.d, z7.d
	cmphi	p1.d, p0/z, z6.d, z7.d
	usra	z16.d, z7.d, #32
	and	z7.d, z7.d, #0xffffffff
	umullb	z6.d, z5.s, z2.s
	mul	z5.d, z5.d, z1.d
	lsl	z7.d, z7.d, #32
	mov	z17.d, p1/z, #1                 // =0x1
	add	z7.d, z6.d, z7.d
	lsl	z17.d, z17.d, #32
	cmphi	p1.d, p0/z, z6.d, z7.d
	add	z6.d, z16.d, z17.d
	add	z6.d, p1/m, z6.d, z4.d
	mls	z5.d, p0/m, z0.d, z6.d
	cmphs	p1.d, p0/z, z5.d, z0.d
	sub	z5.d, p1/m, z5.d, z0.d
	st1d	{ z5.d }, p0, [x1, x8, lsl #3]
	incd	x8
	cmp	x8, x9
	b.lo	.LBB3_2
// %bb.3:
	cmp	x8, x5
	b.hs	.LBB3_5
.LBB3_4:
	whilelo	p0.d, x8, x5
	ld1d	{ z4.d }, p0/z, [x0, x8, lsl #3]
	lsr	z5.d, z4.d, #32
	umullb	z3.d, z4.s, z3.s
	umullt	z6.d, z4.s, z2.s
	mul	z1.d, z4.d, z1.d
	umullb	z5.d, z5.s, z2.s
	umullb	z2.d, z4.s, z2.s
	add	z3.d, z5.d, z3.d
	cmphi	p1.d, p0/z, z5.d, z3.d
	usra	z6.d, z3.d, #32
	and	z3.d, z3.d, #0xffffffff
	lsl	z3.d, z3.d, #32
	mov	z5.d, p1/z, #1                  // =0x1
	add	z3.d, z2.d, z3.d
	lsl	z5.d, z5.d, #32
	cmphi	p1.d, p0/z, z2.d, z3.d
	mov	z3.d, #1                        // =0x1
	add	z2.d, z6.d, z5.d
	add	z2.d, p1/m, z2.d, z3.d
	mls	z1.d, p0/m, z0.d, z2.d
	cmphs	p1.d, p0/z, z1.d, z0.d
	sub	z1.d, p1/m, z1.d, z0.d
	st1d	{ z1.d }, p0, [x1, x8, lsl #3]
.LBB3_5:
	ret
.LBB3_6:
	mov	x8, xzr
	cmp	xzr, x5
	b.lo	.LBB3_4
	b	.LBB3_5
.Lfunc_end3:
	.size	vector_schoolbook, .Lfunc_end3-vector_schoolbook
	.cfi_endproc
                                        // -- End function
	.p2align	2                               // -- Begin function vector_int128
	.type	vector_int128,@function
vector_int128:                          // @vector_int128
	.cfi_startproc
// %bb.0:
	cntd	x8
	mov	z1.d, x2
	mov	z2.d, x3
	rbit	x9, x8
	mov	z0.d, x4
	clz	x9, x9
	lsr	x9, x5, x9
	mul	x9, x9, x8
	cbz	x9, .LBB4_6
// %bb.1:
	ptrue	p0.d
	mov	x8, xzr
.LBB4_2:                                // =>This Inner Loop Header: Depth=1
	ld1d	{ z3.d }, p0/z, [x0, x8, lsl #3]
	umulh	z4.d, z3.d, z2.d
	mul	z3.d, z3.d, z1.d
	mls	z3.d, p0/m, z0.d, z4.d
	cmphs	p1.d, p0/z, z3.d, z0.d
	sub	z3.d, p1/m, z3.d, z0.d
	st1d	{ z3.d }, p0, [x1, x8, lsl #3]
	incd	x8
	cmp	x8, x9
	b.lo	.LBB4_2
// %bb.3:
	cmp	x8, x5
	b.hs	.LBB4_5
.LBB4_4:
	whilelo	p0.d, x8, x5
	ld1d	{ z3.d }, p0/z, [x0, x8, lsl #3]
	umulh	z2.d, z3.d, z2.d
	mul	z1.d, z3.d, z1.d
	mls	z1.d, p0/m, z0.d, z2.d
	cmphs	p1.d, p0/z, z1.d, z0.d
	sub	z1.d, p1/m, z1.d, z0.d
	st1d	{ z1.d }, p0, [x1, x8, lsl #3]
.LBB4_5:
	ret
.LBB4_6:
	mov	x8, xzr
	cmp	xzr, x5
	b.lo	.LBB4_4
	b	.LBB4_5
.Lfunc_end4:
	.size	vector_int128, .Lfunc_end4-vector_int128
	.cfi_endproc
                                        // -- End function
	.type	.L.str,@object                  // @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"scalar schoolbook"
	.size	.L.str, 18

	.type	.L.str.1,@object                // @.str.1
.L.str.1:
	.asciz	"scalar __uint128_t"
	.size	.L.str.1, 19

	.type	.L.str.2,@object                // @.str.2
.L.str.2:
	.asciz	"vector schoolbook"
	.size	.L.str.2, 18

	.type	.L.str.3,@object                // @.str.3
.L.str.3:
	.asciz	"vector int128(svmulh)"
	.size	.L.str.3, 22

	.type	.L.str.5,@object                // @.str.5
.L.str.5:
	.asciz	"    N=%d ITERS=%d VL=%dbit(N=%d) HIP12 2.3GHz\n\n"
	.size	.L.str.5, 48

	.type	.L.str.6,@object                // @.str.6
.L.str.6:
	.asciz	"    %-25s %7.3f cyc/elem  (%.2fx vs schoolbook)\n"
	.size	.L.str.6, 49

	.type	.L.str.7,@object                // @.str.7
.L.str.7:
	.asciz	"schoolbook"
	.size	.L.str.7, 11

	.type	.L.str.8,@object                // @.str.8
.L.str.8:
	.asciz	"vec-schoolbook"
	.size	.L.str.8, 15

	.type	.L.str.9,@object                // @.str.9
.L.str.9:
	.asciz	"vec-int128"
	.size	.L.str.9, 11

	.type	.L.str.10,@object               // @.str.10
.L.str.10:
	.asciz	"    MISMATCH %s[%d]\n"
	.size	.L.str.10, 21

	.type	.L.str.11,@object               // @.str.11
.L.str.11:
	.asciz	"\n    Correctness: %s\n"
	.size	.L.str.11, 22

	.type	.L.str.12,@object               // @.str.12
.L.str.12:
	.asciz	"ALL MATCH"
	.size	.L.str.12, 10

	.type	.L.str.13,@object               // @.str.13
.L.str.13:
	.asciz	"FAILED"
	.size	.L.str.13, 7

	.type	.Lstr,@object                   // @str
	.section	.rodata.str1.4,"aMS",@progbits,1
	.p2align	2, 0x0
.Lstr:
	.asciz	"=== 4-way mulmod benchmark ==="
	.size	.Lstr, 31

	.ident	"clang version 22.1.8 (https://github.com/llvm/llvm-project ca7933e47d3a3451d81e72ac174dcb5aa28b59d1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
