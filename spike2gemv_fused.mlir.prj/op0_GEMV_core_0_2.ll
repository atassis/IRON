; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@A_L3L1_0_cons_buff_1 = external global [32 x [128 x bfloat]]
@A_L3L1_0_cons_buff_0 = external global [32 x [128 x bfloat]]
@B_L3L1_0_cons_buff_0 = external global [128 x bfloat]
@C_L1L3_0_buff_1 = external global [128 x bfloat]
@C_L1L3_0_buff_0 = external global [128 x bfloat]

declare void @debug_i32(i32)

; Unknown intrinsic
declare void @llvm.aie2p.event(i32)

; Unknown intrinsic
declare void @llvm.aie2p.put.ms(i32, i32)

; Unknown intrinsic
declare { i32, i32 } @llvm.aie2p.get.ss()

; Unknown intrinsic
declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32)

; Unknown intrinsic
declare <16 x i32> @llvm.aie2p.scd.read.vec(i32)

; Unknown intrinsic
declare void @llvm.aie2p.acquire(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.release(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.set.ctrl.reg(i32, i32)

declare void @op0_matvec_vectorized_bf16_bf16(i32, i32, ptr, ptr, ptr)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %81, %0
  %2 = phi i64 [ %82, %81 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775806
  br i1 %3, label %4, label %83

4:                                                ; preds = %29, %1
  %5 = phi i64 [ %30, %29 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 4294967294
  br i1 %6, label %7, label %31

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ %17, %11 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 4
  br i1 %10, label %11, label %18

11:                                               ; preds = %8
  %12 = trunc i64 %9 to i32
  %13 = mul i32 %12, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %13, ptr @A_L3L1_0_cons_buff_0, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %14 = add i64 %9, 1
  %15 = trunc i64 %14 to i32
  %16 = mul i32 %15, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %16, ptr @A_L3L1_0_cons_buff_1, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %17 = add i64 %9, 2
  br label %8

18:                                               ; preds = %8
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %19

19:                                               ; preds = %22, %18
  %20 = phi i64 [ %28, %22 ], [ 0, %18 ]
  %21 = icmp slt i64 %20, 4
  br i1 %21, label %22, label %29

22:                                               ; preds = %19
  %23 = trunc i64 %20 to i32
  %24 = mul i32 %23, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %24, ptr @A_L3L1_0_cons_buff_0, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %25 = add i64 %20, 1
  %26 = trunc i64 %25 to i32
  %27 = mul i32 %26, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %27, ptr @A_L3L1_0_cons_buff_1, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %28 = add i64 %20, 2
  br label %19

29:                                               ; preds = %19
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %30 = add i64 %5, 2
  br label %4

31:                                               ; preds = %4
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %32

32:                                               ; preds = %35, %31
  %33 = phi i64 [ %41, %35 ], [ 0, %31 ]
  %34 = icmp slt i64 %33, 4
  br i1 %34, label %35, label %42

35:                                               ; preds = %32
  %36 = trunc i64 %33 to i32
  %37 = mul i32 %36, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %37, ptr @A_L3L1_0_cons_buff_0, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %38 = add i64 %33, 1
  %39 = trunc i64 %38 to i32
  %40 = mul i32 %39, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %40, ptr @A_L3L1_0_cons_buff_1, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %41 = add i64 %33, 2
  br label %32

42:                                               ; preds = %32
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  br label %43

43:                                               ; preds = %68, %42
  %44 = phi i64 [ %69, %68 ], [ 0, %42 ]
  %45 = icmp slt i64 %44, 4294967294
  br i1 %45, label %46, label %70

46:                                               ; preds = %43
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %47

47:                                               ; preds = %50, %46
  %48 = phi i64 [ %56, %50 ], [ 0, %46 ]
  %49 = icmp slt i64 %48, 4
  br i1 %49, label %50, label %57

50:                                               ; preds = %47
  %51 = trunc i64 %48 to i32
  %52 = mul i32 %51, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %52, ptr @A_L3L1_0_cons_buff_0, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %53 = add i64 %48, 1
  %54 = trunc i64 %53 to i32
  %55 = mul i32 %54, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %55, ptr @A_L3L1_0_cons_buff_1, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %56 = add i64 %48, 2
  br label %47

57:                                               ; preds = %47
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %58

58:                                               ; preds = %61, %57
  %59 = phi i64 [ %67, %61 ], [ 0, %57 ]
  %60 = icmp slt i64 %59, 4
  br i1 %60, label %61, label %68

61:                                               ; preds = %58
  %62 = trunc i64 %59 to i32
  %63 = mul i32 %62, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %63, ptr @A_L3L1_0_cons_buff_0, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %64 = add i64 %59, 1
  %65 = trunc i64 %64 to i32
  %66 = mul i32 %65, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %66, ptr @A_L3L1_0_cons_buff_1, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %67 = add i64 %59, 2
  br label %58

68:                                               ; preds = %58
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %69 = add i64 %44, 2
  br label %43

70:                                               ; preds = %43
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %71

71:                                               ; preds = %74, %70
  %72 = phi i64 [ %80, %74 ], [ 0, %70 ]
  %73 = icmp slt i64 %72, 4
  br i1 %73, label %74, label %81

74:                                               ; preds = %71
  %75 = trunc i64 %72 to i32
  %76 = mul i32 %75, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %76, ptr @A_L3L1_0_cons_buff_0, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %77 = add i64 %72, 1
  %78 = trunc i64 %77 to i32
  %79 = mul i32 %78, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %79, ptr @A_L3L1_0_cons_buff_1, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %80 = add i64 %72, 2
  br label %71

81:                                               ; preds = %71
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %82 = add i64 %2, 2
  br label %1

83:                                               ; preds = %108, %1
  %84 = phi i64 [ %109, %108 ], [ 0, %1 ]
  %85 = icmp slt i64 %84, 4294967294
  br i1 %85, label %86, label %110

86:                                               ; preds = %83
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %87

87:                                               ; preds = %90, %86
  %88 = phi i64 [ %96, %90 ], [ 0, %86 ]
  %89 = icmp slt i64 %88, 4
  br i1 %89, label %90, label %97

90:                                               ; preds = %87
  %91 = trunc i64 %88 to i32
  %92 = mul i32 %91, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %92, ptr @A_L3L1_0_cons_buff_0, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %93 = add i64 %88, 1
  %94 = trunc i64 %93 to i32
  %95 = mul i32 %94, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %95, ptr @A_L3L1_0_cons_buff_1, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %96 = add i64 %88, 2
  br label %87

97:                                               ; preds = %87
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %98

98:                                               ; preds = %101, %97
  %99 = phi i64 [ %107, %101 ], [ 0, %97 ]
  %100 = icmp slt i64 %99, 4
  br i1 %100, label %101, label %108

101:                                              ; preds = %98
  %102 = trunc i64 %99 to i32
  %103 = mul i32 %102, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %103, ptr @A_L3L1_0_cons_buff_0, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %104 = add i64 %99, 1
  %105 = trunc i64 %104 to i32
  %106 = mul i32 %105, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %106, ptr @A_L3L1_0_cons_buff_1, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %107 = add i64 %99, 2
  br label %98

108:                                              ; preds = %98
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %109 = add i64 %84, 2
  br label %83

110:                                              ; preds = %83
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %111

111:                                              ; preds = %114, %110
  %112 = phi i64 [ %120, %114 ], [ 0, %110 ]
  %113 = icmp slt i64 %112, 4
  br i1 %113, label %114, label %121

114:                                              ; preds = %111
  %115 = trunc i64 %112 to i32
  %116 = mul i32 %115, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %116, ptr @A_L3L1_0_cons_buff_0, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %117 = add i64 %112, 1
  %118 = trunc i64 %117 to i32
  %119 = mul i32 %118, 32
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @op0_matvec_vectorized_bf16_bf16(i32 32, i32 %119, ptr @A_L3L1_0_cons_buff_1, ptr @B_L3L1_0_cons_buff_0, ptr @C_L1L3_0_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  %120 = add i64 %112, 2
  br label %111

121:                                              ; preds = %111
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
