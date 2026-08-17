	.arch armv9-a+crc
	.file	"bench_4way.c"
	.text
	.align	2
	.p2align 5,,15
	.type	scalar_schoolbook, %function
scalar_schoolbook:
.LFB27:
	.cfi_startproc
	cbz	x5, .L1
	sub	x6, x5, #1
	lsr	x11, x3, 32
	and	x7, x3, 4294967295
	cmp	x6, 54
	bls	.L3
	whilewr	p15.d, x0, x1
	b.nlast	.L3
	mov	x3, 0
	mov	z31.d, x4
	mov	z30.d, x2
	mov	z29.d, x7
	mov	z28.d, x11
	whilelo	p6.d, wzr, w5
	ptrue	p7.b, all
	mov	z27.d, #4294967296
	movi	d26, #0
	.p2align 5,,15
.L4:
	ld1d	z25.d, p6/z, [x0, x3, lsl 3]
	lsr	z24.d, z25.d, #32
	movprfx	z22, z25
	and	z22.d, z22.d, #0xffffffff
	mul	z1.d, z29.d, z24.d
	movprfx	z0, z1
	mla	z0.d, p7/m, z22.d, z28.d
	cmphi	p15.d, p7/z, z1.d, z0.d
	mul	z22.d, z22.d, z29.d
	lsr	z20.d, z0.d, #32
	sel	z23.d, p15, z27.d, z26.d
	mul	z25.d, z25.d, z30.d
	mla	z20.d, p7/m, z24.d, z28.d
	lsl	z0.d, z0.d, #32
	add	z0.d, z0.d, z22.d
	cmphi	p15.d, p7/z, z22.d, z0.d
	mov	z21.d, p15/z, #-1
	sub	z21.d, z23.d, z21.d
	add	z21.d, z21.d, z20.d
	mls	z25.d, p7/m, z31.d, z21.d
	cmpls	p5.d, p7/z, z31.d, z25.d
	sub	z25.d, p5/m, z25.d, z31.d
	st1d	z25.d, p6, [x1, x3, lsl 3]
	incd	x3
	whilelo	p6.d, w3, w5
	b.any	.L4
.L1:
	ret
	.p2align 2,,3
.L3:
	lsl	x5, x5, 3
	mov	x7, 0
	.p2align 5,,15
.L7:
	ldr	x6, [x0, x7]
	lsr	x8, x6, 32
	umull	x10, w6, w3
	mul	x12, x2, x6
	umull	x9, w8, w3
	umaddl	x6, w11, w6, x9
	lsr	x13, x6, 32
	add	x14, x10, x6, lsl 32
	cmp	x14, x10
	cset	x10, cc
	cmp	x6, x9
	cset	x6, cc
	umaddl	x8, w11, w8, x13
	add	x6, x10, x6, lsl 32
	add	x6, x6, x8
	msub	x6, x6, x4, x12
	sub	x8, x6, x4
	cmp	x4, x6
	csel	x6, x8, x6, ls
	str	x6, [x1, x7]
	add	x7, x7, 8
	cmp	x5, x7
	bne	.L7
	ret
	.cfi_endproc
.LFE27:
	.size	scalar_schoolbook, .-scalar_schoolbook
	.align	2
	.p2align 5,,15
	.type	scalar_int128, %function
scalar_int128:
.LFB28:
	.cfi_startproc
	lsl	x8, x5, 3
	mov	x6, 0
	cbz	x5, .L15
	.p2align 5,,15
.L18:
	ldr	x7, [x0, x6]
	umulh	x5, x7, x3
	mul	x7, x2, x7
	msub	x5, x4, x5, x7
	sub	x7, x5, x4
	cmp	x4, x5
	csel	x5, x7, x5, ls
	str	x5, [x1, x6]
	add	x6, x6, 8
	cmp	x8, x6
	bne	.L18
.L15:
	ret
	.cfi_endproc
.LFE28:
	.size	scalar_int128, .-scalar_int128
	.align	2
	.p2align 5,,15
	.type	vector_int128, %function
vector_int128:
.LFB31:
	.cfi_startproc
	cntd	x6
	mov	z30.d, x3
	mov	z31.d, x2
	mov	z29.d, x4
	mov	x2, 0
	udiv	x3, x5, x6
	mul	x3, x3, x6
	cbz	x3, .L23
	ptrue	p7.b, all
	.p2align 5,,15
.L24:
	ld1d	z0.d, p7/z, [x0, x2, lsl 3]
	movprfx	z27, z0
	umulh	z27.d, p7/m, z27.d, z30.d
	mul	z0.d, p7/m, z0.d, z31.d
	mls	z0.d, p7/m, z29.d, z27.d
	cmphs	p6.d, p7/z, z0.d, z29.d
	sub	z0.d, p6/m, z0.d, z29.d
	st1d	z0.d, p7, [x1, x2, lsl 3]
	incd	x2
	cmp	x3, x2
	bhi	.L24
.L23:
	cmp	x5, x2
	bls	.L22
	whilelo	p7.d, x2, x5
	ld1d	z28.d, p7/z, [x0, x2, lsl 3]
	umulh	z30.d, p7/m, z30.d, z28.d
	mul	z28.d, p7/m, z28.d, z31.d
	mls	z28.d, p7/m, z29.d, z30.d
	cmphs	p6.d, p7/z, z28.d, z29.d
	sub	z28.d, p6/m, z28.d, z29.d
	st1d	z28.d, p7, [x1, x2, lsl 3]
.L22:
	ret
	.cfi_endproc
.LFE31:
	.size	vector_int128, .-vector_int128
	.align	2
	.p2align 5,,15
	.type	vector_schoolbook, %function
vector_schoolbook:
.LFB30:
	.cfi_startproc
	cntd	x6
	mov	z30.d, x3
	mov	z31.d, x2
	ptrue	p7.b, all
	mov	x2, 0
	lsr	z28.d, z30.d, #32
	udiv	x3, x5, x6
	mov	z29.d, x4
	mul	x3, x3, x6
	cbz	x3, .L29
	.p2align 5,,15
.L30:
	ld1d	z3.d, p7/z, [x0, x2, lsl 3]
	umullb	z26.d, z3.s, z30.s
	lsr	z16.d, z3.d, #32
	umullb	z16.d, z16.s, z30.s
	movprfx	z6, z16
	umlalb	z6.d, z3.s, z28.s
	movprfx	z7, z6
	and	z7.d, z7.d, #0xffffffff
	cmplo	p15.d, p7/z, z6.d, z16.d
	lsl	z7.d, z7.d, #32
	lsr	z6.d, z6.d, #32
	add	z7.d, z26.d, z7.d
	umlalt	z6.d, z3.s, z30.s
	cmplo	p14.d, p7/z, z7.d, z26.d
	mul	z3.d, p7/m, z3.d, z31.d
	mov	z4.d, p14/z, #1
	mov	z5.d, p15/z, #1
	lsl	z5.d, z5.d, #32
	add	z5.d, z6.d, z5.d
	add	z4.d, z5.d, z4.d
	mls	z3.d, p7/m, z29.d, z4.d
	cmphs	p6.d, p7/z, z3.d, z29.d
	sub	z3.d, p6/m, z3.d, z29.d
	st1d	z3.d, p7, [x1, x2, lsl 3]
	incd	x2
	cmp	x3, x2
	bhi	.L30
.L29:
	cmp	x5, x2
	bls	.L28
	whilelo	p7.d, x2, x5
	ld1d	z24.d, p7/z, [x0, x2, lsl 3]
	lsr	z2.d, z24.d, #32
	umullb	z2.d, z2.s, z30.s
	movprfx	z0, z2
	umlalb	z0.d, z24.s, z28.s
	cmplo	p15.d, p7/z, z0.d, z2.d
	umullb	z25.d, z24.s, z30.s
	movprfx	z1, z0
	and	z1.d, z1.d, #0xffffffff
	mov	z23.d, p15/z, #1
	lsl	z1.d, z1.d, #32
	lsr	z0.d, z0.d, #32
	add	z1.d, z25.d, z1.d
	umlalt	z0.d, z24.s, z30.s
	lsl	z23.d, z23.d, #32
	mul	z24.d, p7/m, z24.d, z31.d
	cmplo	p15.d, p7/z, z1.d, z25.d
	add	z23.d, z0.d, z23.d
	mov	z27.d, p15/z, #1
	add	z27.d, z23.d, z27.d
	mls	z24.d, p7/m, z29.d, z27.d
	cmphs	p6.d, p7/z, z24.d, z29.d
	sub	z24.d, p6/m, z24.d, z29.d
	st1d	z24.d, p7, [x1, x2, lsl 3]
.L28:
	ret
	.cfi_endproc
.LFE30:
	.size	vector_schoolbook, .-vector_schoolbook
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align	3
.LC10:
	.string	"ALL MATCH"
	.align	3
.LC11:
	.string	"FAILED"
	.align	3
.LC14:
	.string	"=== 4-way mulmod benchmark ==="
	.align	3
.LC15:
	.string	"    N=%d ITERS=%d VL=%dbit(N=%d) HIP12 2.3GHz\n\n"
	.align	3
.LC0:
	.string	"scalar schoolbook"
	.align	3
.LC16:
	.string	"    %-25s %7.3f cyc/elem  (%.2fx vs schoolbook)\n"
	.align	3
.LC18:
	.string	"    MISMATCH %s[%d]\n"
	.align	3
.LC19:
	.string	"\n    Correctness: %s\n"
	.section	.text.startup,"ax",@progbits
	.align	2
	.p2align 5,,15
	.global	main
	.type	main, %function
main:
.LFB32:
	.cfi_startproc
	sub	sp, sp, #352
	.cfi_def_cfa_offset 352
	mov	x4, 2
	sub	sp, sp, #1048576
	.cfi_def_cfa_offset 1048928
	mov	x1, 128
	add	x3, sp, 232
	add	x2, sp, 224
	movi	v31.4s, 0
	mov	w0, 0
	stp	x29, x30, [sp]
	.cfi_offset 29, -1048928
	.cfi_offset 30, -1048920
	mov	x29, sp
	stp	x19, x20, [sp, 16]
	stp	x21, x22, [sp, 32]
	stp	x23, x24, [sp, 48]
	stp	x25, x26, [sp, 64]
	str	x27, [sp, 80]
	str	d15, [sp, 88]
	stp	d13, d14, [sp, 96]
	.cfi_offset 19, -1048912
	.cfi_offset 20, -1048904
	.cfi_offset 21, -1048896
	.cfi_offset 22, -1048888
	.cfi_offset 23, -1048880
	.cfi_offset 24, -1048872
	.cfi_offset 25, -1048864
	.cfi_offset 26, -1048856
	.cfi_offset 27, -1048848
	.cfi_offset 79, -1048840
	.cfi_offset 77, -1048832
	.cfi_offset 78, -1048824
	str	xzr, [x3, 112]
	str	x4, [sp, 224]
	str	q31, [sp, 232]
	stp	q31, q31, [x3, 16]
	stp	q31, q31, [x3, 48]
	stp	q31, q31, [x3, 80]
	bl	sched_setaffinity
	mov	x1, 524288
	mov	x0, 64
	bl	aligned_alloc
	mov	x1, 524288
	mov	x20, x0
	mov	x0, 64
	bl	aligned_alloc
	mov	x21, x0
	mov	x0, 0
	ptrue	p7.b, all
	adrp	x1, .LC12
	add	x1, x1, :lo12:.LC12
	index	z31.d, #0, #1
	cntd	x2
	ld1rd	z30.d, p7/z, [x1]
	mov	z28.s, w2
	mov	w1, 65536
	mov	z29.d, #1
	whilelo	p6.d, wzr, w1
	.p2align 5,,15
.L35:
	movprfx	z27, z31
	sxtw	z27.d, p7/m, z31.d
	mad	z27.d, p7/m, z30.d, z29.d
	st1d	z27.d, p6, [x20, x0, lsl 3]
	add	z31.s, z31.s, z28.s
	incd	x0
	whilelo	p6.d, w0, w1
	b.any	.L35
	adrp	x23, .LANCHOR0
	add	x23, x23, :lo12:.LANCHOR0
	adrp	x0, .LC14
	add	x0, x0, :lo12:.LC14
	ldp	q31, q29, [x23]
	stp	q31, q29, [sp, 160]
	ldp	q30, q31, [x23, 32]
	stp	q30, q31, [sp, 192]
	bl	puts
	cntd	x4
	cntb	x3, all, mul #8
	mov	w2, 1000
	mov	w1, 65536
	adrp	x0, .LC15
	add	x0, x0, :lo12:.LC15
	bl	printf
	mov	x3, 57072
	mov	x1, x21
	movk	x3, 0x9abc, lsl 16
	mov	x0, x20
	movk	x3, 0x5678, lsl 32
	mov	x5, 65536
	movk	x3, 0x1234, lsl 48
	mov	x4, -59
	mov	x2, x3
	bl	scalar_schoolbook
#APP
// 24 "/home/hukeke/spec/src/SEAL/native/bench/bench_4way.c" 1
	mrs x19, cntvct_el0
// 0 "" 2
#NO_APP
	mov	x18, x2
	mov	w15, 1000
	.p2align 5,,15
.L36:
	mov	x2, 57072
	mov	x3, x18
	movk	x2, 0x9abc, lsl 16
	mov	x1, x21
	movk	x2, 0x5678, lsl 32
	mov	x0, x20
	mov	x5, 65536
	mov	x4, -59
	movk	x2, 0x1234, lsl 48
	bl	scalar_schoolbook
	subs	w15, w15, #1
	bne	.L36
#APP
// 24 "/home/hukeke/spec/src/SEAL/native/bench/bench_4way.c" 1
	mrs x0, cntvct_el0
// 0 "" 2
#NO_APP
	sub	x0, x0, x19
	fmov	d13, 2.3e+1
	mov	x1, 70368744177664
	add	x24, sp, 160
	ucvtf	d15, x0
	movk	x1, 0x418f, lsl 48
	fmov	d14, x1
	add	x26, x24, 48
	adrp	x25, .LC16
	add	x25, x25, :lo12:.LC16
	mov	x0, x25
	adrp	x1, .LC0
	fmul	d15, d15, d13
	add	x1, x1, :lo12:.LC0
	fdiv	d15, d15, d14
	fdiv	d1, d15, d15
	fmov	d0, d15
	bl	printf
.L38:
	ldr	x22, [x24, 24]
	mov	x2, 57072
	mov	x3, 57072
	movk	x2, 0x9abc, lsl 16
	movk	x3, 0x9abc, lsl 16
	movk	x2, 0x5678, lsl 32
	movk	x3, 0x5678, lsl 32
	mov	x1, x21
	mov	x0, x20
	mov	x5, 65536
	mov	x4, -59
	movk	x2, 0x1234, lsl 48
	movk	x3, 0x1234, lsl 48
	blr	x22
#APP
// 24 "/home/hukeke/spec/src/SEAL/native/bench/bench_4way.c" 1
	mrs x27, cntvct_el0
// 0 "" 2
#NO_APP
	mov	w19, 1000
	.p2align 5,,15
.L37:
	mov	x2, 57072
	mov	x3, 57072
	movk	x2, 0x9abc, lsl 16
	movk	x3, 0x9abc, lsl 16
	movk	x2, 0x5678, lsl 32
	movk	x3, 0x5678, lsl 32
	mov	x1, x21
	mov	x0, x20
	mov	x5, 65536
	mov	x4, -59
	movk	x3, 0x1234, lsl 48
	movk	x2, 0x1234, lsl 48
	blr	x22
	subs	w19, w19, #1
	bne	.L37
#APP
// 24 "/home/hukeke/spec/src/SEAL/native/bench/bench_4way.c" 1
	mrs x2, cntvct_el0
// 0 "" 2
#NO_APP
	sub	x2, x2, x27
	mov	x0, x25
	ldr	x1, [x24, 16]!
	ucvtf	d0, x2
	fmul	d0, d0, d13
	fdiv	d0, d0, d14
	fdiv	d1, d15, d0
	bl	printf
	cmp	x26, x24
	bne	.L38
	mov	x3, 57072
	add	x27, sp, 352
	movk	x3, 0x9abc, lsl 16
	mov	x1, x27
	mov	x0, x20
	movk	x3, 0x5678, lsl 32
	movk	x3, 0x1234, lsl 48
	mov	x4, -59
	mov	x2, x3
	mov	x5, 65536
	bl	scalar_int128
	add	x19, sp, 524288
	ldp	x0, x1, [x23, 64]
	mov	x25, x3
	ldr	x4, [x23, 80]
	mov	x26, 1
	ldp	x2, x3, [x23, 88]
	add	x19, x19, 352
	add	x24, sp, 112
	mov	w22, w26
	stp	x0, x1, [sp, 112]
	ldr	x0, [x23, 104]
	str	x4, [sp, 128]
	stp	x2, x3, [sp, 136]
	str	x0, [sp, 152]
.L44:
	add	x0, x24, x26, lsl 3
	mov	x2, 57072
	movk	x2, 0x9abc, lsl 16
	mov	x1, x19
	movk	x2, 0x5678, lsl 32
	mov	x3, x25
	ldr	x6, [x0, -8]
	mov	x5, 65536
	mov	x0, x20
	movk	x2, 0x1234, lsl 48
	mov	x4, -59
	lsl	x23, x26, 3
	blr	x6
	adrp	x1, .LC21
	mov	x0, x27
	index	z29.s, #0, #1
	ldr	q28, [x1, #:lo12:.LC21]
	mov	x1, x19
	mvni	v25.4s, 0x3
	movi	v24.4s, 0x4
	b	.L43
	.p2align 2,,3
.L39:
	add	v29.4s, v29.4s, v24.4s
	add	v28.4s, v28.4s, v25.4s
	cmp	x19, x0
	beq	.L41
.L43:
	ldp	q30, q31, [x0], 32
	ldp	q27, q26, [x1], 32
	cmeq	v31.2d, v31.2d, v26.2d
	cmeq	v30.2d, v30.2d, v27.2d
	not	v31.16b, v31.16b
	orn	v31.16b, v31.16b, v30.16b
	umaxp	v31.4s, v31.4s, v31.4s
	fmov	x2, d31
	cbz	x2, .L39
	fmov	w0, s29
	ldr	x1, [x19, w0, sxtw 3]
	ldr	x0, [x27, w0, sxtw 3]
	cmp	x1, x0
	bne	.L46
	fmov	w0, s28
	cmp	w0, 1
	beq	.L41
	fmov	w0, s29
	add	w2, w0, 1
	ldr	x1, [x27, w2, sxtw 3]
	ldr	x0, [x19, w2, sxtw 3]
	cmp	x1, x0
	bne	.L40
	fmov	w0, s28
	cmp	w0, 2
	beq	.L41
	fmov	w0, s29
	add	w2, w0, 2
	ldr	x1, [x19, w2, sxtw 3]
	ldr	x0, [x27, w2, sxtw 3]
	cmp	x1, x0
	bne	.L40
	fmov	w0, s28
	cmp	w0, 3
	beq	.L41
	fmov	w0, s29
	add	w2, w0, 3
	ldr	x1, [x19, w2, sxtw 3]
	ldr	x0, [x27, w2, sxtw 3]
	cmp	x1, x0
	bne	.L40
.L41:
	add	x26, x26, 1
	cmp	x26, 4
	bne	.L44
	cmp	w22, 0
	adrp	x1, .LC10
	adrp	x2, .LC11
	add	x1, x1, :lo12:.LC10
	add	x2, x2, :lo12:.LC11
	adrp	x0, .LC19
	csel	x1, x2, x1, eq
	add	x0, x0, :lo12:.LC19
	bl	printf
	mov	x0, x20
	bl	free
	mov	x0, x21
	bl	free
	ldr	x27, [sp, 80]
	mov	w0, 0
	ldr	d15, [sp, 88]
	ldp	x29, x30, [sp]
	ldp	x19, x20, [sp, 16]
	ldp	x21, x22, [sp, 32]
	ldp	x23, x24, [sp, 48]
	ldp	x25, x26, [sp, 64]
	ldp	d13, d14, [sp, 96]
	.cfi_remember_state
	.cfi_restore 27
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_restore 29
	.cfi_restore 30
	.cfi_restore 79
	.cfi_restore 77
	.cfi_restore 78
	add	sp, sp, 352
	.cfi_def_cfa_offset 1048576
	add	sp, sp, 1048576
	.cfi_def_cfa_offset 0
	ret
.L46:
	.cfi_restore_state
	fmov	w2, s29
.L40:
	add	x1, sp, x23
	adrp	x0, .LC18
	mov	w22, 0
	add	x0, x0, :lo12:.LC18
	ldr	x1, [x1, 128]
	bl	printf
	b	.L41
	.cfi_endproc
.LFE32:
	.size	main, .-main
	.section	.rodata.str1.8
	.align	3
.LC6:
	.string	"schoolbook"
	.align	3
.LC7:
	.string	"vec-schoolbook"
	.align	3
.LC8:
	.string	"vec-int128"
	.align	3
.LC1:
	.string	"scalar __uint128_t"
	.align	3
.LC2:
	.string	"vector schoolbook"
	.align	3
.LC3:
	.string	"vector int128(svmulh)"
	.section	.rodata.cst8,"aM",@progbits,8
	.align	3
.LC12:
	.xword	123456789
	.section	.rodata.cst16,"aM",@progbits,16
	.align	4
.LC21:
	.word	65536
	.word	65535
	.word	65534
	.word	65533
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
.LC13:
	.xword	.LC0
	.xword	scalar_schoolbook
	.xword	.LC1
	.xword	scalar_int128
	.xword	.LC2
	.xword	vector_schoolbook
	.xword	.LC3
	.xword	vector_int128
.LC5:
	.xword	scalar_schoolbook
	.xword	vector_schoolbook
	.xword	vector_int128
.LC17:
	.xword	.LC6
	.xword	.LC7
	.xword	.LC8
	.ident	"GCC: (spec-study gcc-15.1.0) 15.1.0"
	.section	.note.GNU-stack,"",@progbits
