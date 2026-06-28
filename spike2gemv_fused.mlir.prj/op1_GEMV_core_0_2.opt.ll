; ModuleID = 'spike2gemv_fused.mlir.prj/op1_GEMV_core_0_2.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@A_L3L1_0_cons_buff_1 = external global [32 x [128 x bfloat]]
@A_L3L1_0_cons_buff_0 = external global [32 x [128 x bfloat]]
@B_L3L1_0_cons_buff_0 = external global [128 x bfloat]
@C_L1L3_0_buff_1 = external global [128 x bfloat]
@C_L1L3_0_buff_0 = external global [128 x bfloat]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @op1_matvec_vectorized_bf16_bf16(i32, i32, ptr, ptr, ptr) local_unnamed_addr

define void @core_0_2() local_unnamed_addr {
  br label %.preheader12

.preheader12:                                     ; preds = %0, %13
  %1 = phi i64 [ 0, %0 ], [ %14, %13 ]
  br label %2

2:                                                ; preds = %5, %.preheader12
  %3 = phi i64 [ 0, %.preheader12 ], [ %6, %5 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 0, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 32, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 64, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 96, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 0, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 32, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 64, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 96, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %4 = icmp samesign ult i64 %3, 4294967292
  br i1 %4, label %5, label %7

5:                                                ; preds = %2
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 0, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 32, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 64, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 96, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 0, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 32, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 64, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 96, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %6 = add nuw nsw i64 %3, 4
  br label %2

7:                                                ; preds = %2
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 0, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 32, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 64, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 96, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ 0, %7 ], [ %12, %11 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 0, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 32, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 64, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 96, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 0, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 32, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 64, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 96, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %10 = icmp samesign ult i64 %9, 4294967292
  br i1 %10, label %11, label %13

11:                                               ; preds = %8
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 0, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 32, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 64, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 96, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 0, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 32, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 64, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 96, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %12 = add nuw nsw i64 %9, 4
  br label %8

13:                                               ; preds = %8
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 0, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 32, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 64, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 96, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %14 = add nuw nsw i64 %1, 2
  %.not = icmp eq i64 %14, 9223372036854775806
  br i1 %.not, label %.preheader, label %.preheader12

.preheader:                                       ; preds = %13, %.preheader.1
  %15 = phi i64 [ %17, %.preheader.1 ], [ 0, %13 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 0, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 32, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 64, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 96, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 0, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 32, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 64, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 96, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %16 = icmp samesign ult i64 %15, 4294967292
  br i1 %16, label %.preheader.1, label %18

.preheader.1:                                     ; preds = %.preheader
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 0, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 32, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 64, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 96, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 0, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 32, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 64, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 96, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %17 = add nuw nsw i64 %15, 4
  br label %.preheader

18:                                               ; preds = %.preheader
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 0, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 32, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 64, ptr nonnull @A_L3L1_0_cons_buff_0, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @op1_matvec_vectorized_bf16_bf16(i32 32, i32 96, ptr nonnull @A_L3L1_0_cons_buff_1, ptr nonnull @B_L3L1_0_cons_buff_0, ptr nonnull @C_L1L3_0_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
