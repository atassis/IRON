module {
  aie.device(npu2) @op0_GEMV {
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %C_L1L3_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "C_L1L3_0_cons_prod_lock_0"}
    %C_L1L3_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "C_L1L3_0_cons_cons_lock_0"}
    %C_L1L3_0_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L3_0_buff_0"} : memref<128xbf16> 
    %C_L1L3_0_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "C_L1L3_0_buff_1"} : memref<128xbf16> 
    %C_L1L3_0_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "C_L1L3_0_prod_lock_0"}
    %C_L1L3_0_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "C_L1L3_0_cons_lock_0"}
    %B_L3L1_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 9216 : i32, mem_bank = 0 : i32, sym_name = "B_L3L1_0_cons_buff_0"} : memref<128xbf16> 
    %B_L3L1_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "B_L3L1_0_cons_prod_lock_0"}
    %B_L3L1_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "B_L3L1_0_cons_cons_lock_0"}
    %B_L3L1_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "B_L3L1_0_prod_lock_0"}
    %B_L3L1_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "B_L3L1_0_cons_lock_0"}
    %A_L3L1_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "A_L3L1_0_cons_buff_0"} : memref<32x128xbf16> 
    %A_L3L1_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "A_L3L1_0_cons_buff_1"} : memref<32x128xbf16> 
    %A_L3L1_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "A_L3L1_0_cons_prod_lock_0"}
    %A_L3L1_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "A_L3L1_0_cons_cons_lock_0"}
    %A_L3L1_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "A_L3L1_0_prod_lock_0"}
    %A_L3L1_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "A_L3L1_0_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    func.func private @op0_matvec_vectorized_bf16_bf16(i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) attributes {link_with = "op0_gemv_128k_64vs.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c4294967294 = arith.constant 4294967294 : index
      %c32_i32 = arith.constant 32 : i32
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb26
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb27
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb10
      %3 = arith.cmpi slt, %2, %c4294967294 : index
      cf.cond_br %3, ^bb4, ^bb11
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c4 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %6 = index.casts %4 : index to i32
      %7 = arith.muli %6, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %7, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %8 = arith.addi %4, %c1 : index
      %9 = index.casts %8 : index to i32
      %10 = arith.muli %9, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %10, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %11 = arith.addi %4, %c2 : index
      cf.br ^bb5(%11 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb8(%c0 : index)
    ^bb8(%12: index):  // 2 preds: ^bb7, ^bb9
      %13 = arith.cmpi slt, %12, %c4 : index
      cf.cond_br %13, ^bb9, ^bb10
    ^bb9:  // pred: ^bb8
      %14 = index.casts %12 : index to i32
      %15 = arith.muli %14, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %15, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_1) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %16 = arith.addi %12, %c1 : index
      %17 = index.casts %16 : index to i32
      %18 = arith.muli %17, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %18, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_1) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %19 = arith.addi %12, %c2 : index
      cf.br ^bb8(%19 : index)
    ^bb10:  // pred: ^bb8
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      %20 = arith.addi %2, %c2 : index
      cf.br ^bb3(%20 : index)
    ^bb11:  // pred: ^bb3
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb12(%c0 : index)
    ^bb12(%21: index):  // 2 preds: ^bb11, ^bb13
      %22 = arith.cmpi slt, %21, %c4 : index
      cf.cond_br %22, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      %23 = index.casts %21 : index to i32
      %24 = arith.muli %23, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %24, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %25 = arith.addi %21, %c1 : index
      %26 = index.casts %25 : index to i32
      %27 = arith.muli %26, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %27, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %28 = arith.addi %21, %c2 : index
      cf.br ^bb12(%28 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      cf.br ^bb15(%c0 : index)
    ^bb15(%29: index):  // 2 preds: ^bb14, ^bb22
      %30 = arith.cmpi slt, %29, %c4294967294 : index
      cf.cond_br %30, ^bb16, ^bb23
    ^bb16:  // pred: ^bb15
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb17(%c0 : index)
    ^bb17(%31: index):  // 2 preds: ^bb16, ^bb18
      %32 = arith.cmpi slt, %31, %c4 : index
      cf.cond_br %32, ^bb18, ^bb19
    ^bb18:  // pred: ^bb17
      %33 = index.casts %31 : index to i32
      %34 = arith.muli %33, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %34, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_1) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %35 = arith.addi %31, %c1 : index
      %36 = index.casts %35 : index to i32
      %37 = arith.muli %36, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %37, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_1) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %38 = arith.addi %31, %c2 : index
      cf.br ^bb17(%38 : index)
    ^bb19:  // pred: ^bb17
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb20(%c0 : index)
    ^bb20(%39: index):  // 2 preds: ^bb19, ^bb21
      %40 = arith.cmpi slt, %39, %c4 : index
      cf.cond_br %40, ^bb21, ^bb22
    ^bb21:  // pred: ^bb20
      %41 = index.casts %39 : index to i32
      %42 = arith.muli %41, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %42, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %43 = arith.addi %39, %c1 : index
      %44 = index.casts %43 : index to i32
      %45 = arith.muli %44, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %45, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %46 = arith.addi %39, %c2 : index
      cf.br ^bb20(%46 : index)
    ^bb22:  // pred: ^bb20
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      %47 = arith.addi %29, %c2 : index
      cf.br ^bb15(%47 : index)
    ^bb23:  // pred: ^bb15
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb24(%c0 : index)
    ^bb24(%48: index):  // 2 preds: ^bb23, ^bb25
      %49 = arith.cmpi slt, %48, %c4 : index
      cf.cond_br %49, ^bb25, ^bb26
    ^bb25:  // pred: ^bb24
      %50 = index.casts %48 : index to i32
      %51 = arith.muli %50, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %51, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_1) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %52 = arith.addi %48, %c1 : index
      %53 = index.casts %52 : index to i32
      %54 = arith.muli %53, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %54, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_1) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %55 = arith.addi %48, %c2 : index
      cf.br ^bb24(%55 : index)
    ^bb26:  // pred: ^bb24
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      %56 = arith.addi %0, %c2 : index
      cf.br ^bb1(%56 : index)
    ^bb27:  // pred: ^bb1
      cf.br ^bb28(%c0 : index)
    ^bb28(%57: index):  // 2 preds: ^bb27, ^bb35
      %58 = arith.cmpi slt, %57, %c4294967294 : index
      cf.cond_br %58, ^bb29, ^bb36
    ^bb29:  // pred: ^bb28
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb30(%c0 : index)
    ^bb30(%59: index):  // 2 preds: ^bb29, ^bb31
      %60 = arith.cmpi slt, %59, %c4 : index
      cf.cond_br %60, ^bb31, ^bb32
    ^bb31:  // pred: ^bb30
      %61 = index.casts %59 : index to i32
      %62 = arith.muli %61, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %62, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %63 = arith.addi %59, %c1 : index
      %64 = index.casts %63 : index to i32
      %65 = arith.muli %64, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %65, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %66 = arith.addi %59, %c2 : index
      cf.br ^bb30(%66 : index)
    ^bb32:  // pred: ^bb30
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb33(%c0 : index)
    ^bb33(%67: index):  // 2 preds: ^bb32, ^bb34
      %68 = arith.cmpi slt, %67, %c4 : index
      cf.cond_br %68, ^bb34, ^bb35
    ^bb34:  // pred: ^bb33
      %69 = index.casts %67 : index to i32
      %70 = arith.muli %69, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %70, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_1) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %71 = arith.addi %67, %c1 : index
      %72 = index.casts %71 : index to i32
      %73 = arith.muli %72, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %73, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_1) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %74 = arith.addi %67, %c2 : index
      cf.br ^bb33(%74 : index)
    ^bb35:  // pred: ^bb33
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      %75 = arith.addi %57, %c2 : index
      cf.br ^bb28(%75 : index)
    ^bb36:  // pred: ^bb28
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb37(%c0 : index)
    ^bb37(%76: index):  // 2 preds: ^bb36, ^bb38
      %77 = arith.cmpi slt, %76, %c4 : index
      cf.cond_br %77, ^bb38, ^bb39
    ^bb38:  // pred: ^bb37
      %78 = index.casts %76 : index to i32
      %79 = arith.muli %78, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %79, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %80 = arith.addi %76, %c1 : index
      %81 = index.casts %80 : index to i32
      %82 = arith.muli %81, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op0_matvec_vectorized_bf16_bf16(%c32_i32, %82, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %83 = arith.addi %76, %c2 : index
      cf.br ^bb37(%83 : index)
    ^bb39:  // pred: ^bb37
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      aie.end
    } {link_files = ["op0_gemv_128k_64vs.o"]}
    aie.runtime_sequence(%arg0: memref<16384xbf16>, %arg1: memref<128xbf16>, %arg2: memref<128xbf16>) {
      %0 = aiex.dma_configure_task_for @B_L3L1_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<128xbf16>, 0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @A_L3L1_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384xbf16>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @C_L1L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<128xbf16>, 0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%0)
    }
    aie.shim_dma_allocation @A_L3L1_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_0_cons_buff_0 : memref<32x128xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_0_cons_buff_1 : memref<32x128xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_0_cons_buff_0 : memref<128xbf16>, 0, 128) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%C_L1L3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_0_buff_0 : memref<128xbf16>, 0, 128) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%C_L1L3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_0_buff_1 : memref<128xbf16>, 0, 128) {bd_id = 4 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb8:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @B_L3L1_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @C_L1L3_0_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
  aie.device(npu2) @op1_GEMV {
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %C_L1L3_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "C_L1L3_0_cons_prod_lock_0"}
    %C_L1L3_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "C_L1L3_0_cons_cons_lock_0"}
    %C_L1L3_0_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L3_0_buff_0"} : memref<128xbf16> 
    %C_L1L3_0_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "C_L1L3_0_buff_1"} : memref<128xbf16> 
    %C_L1L3_0_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "C_L1L3_0_prod_lock_0"}
    %C_L1L3_0_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "C_L1L3_0_cons_lock_0"}
    %B_L3L1_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 9216 : i32, mem_bank = 0 : i32, sym_name = "B_L3L1_0_cons_buff_0"} : memref<128xbf16> 
    %B_L3L1_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "B_L3L1_0_cons_prod_lock_0"}
    %B_L3L1_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "B_L3L1_0_cons_cons_lock_0"}
    %B_L3L1_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "B_L3L1_0_prod_lock_0"}
    %B_L3L1_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "B_L3L1_0_cons_lock_0"}
    %A_L3L1_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "A_L3L1_0_cons_buff_0"} : memref<32x128xbf16> 
    %A_L3L1_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "A_L3L1_0_cons_buff_1"} : memref<32x128xbf16> 
    %A_L3L1_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "A_L3L1_0_cons_prod_lock_0"}
    %A_L3L1_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "A_L3L1_0_cons_cons_lock_0"}
    %A_L3L1_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "A_L3L1_0_prod_lock_0"}
    %A_L3L1_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "A_L3L1_0_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    func.func private @op1_matvec_vectorized_bf16_bf16(i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) attributes {link_with = "op1_gemv_128k_64vs.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c4294967294 = arith.constant 4294967294 : index
      %c32_i32 = arith.constant 32 : i32
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb26
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb27
    ^bb2:  // pred: ^bb1
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb10
      %3 = arith.cmpi slt, %2, %c4294967294 : index
      cf.cond_br %3, ^bb4, ^bb11
    ^bb4:  // pred: ^bb3
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c4 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %6 = index.casts %4 : index to i32
      %7 = arith.muli %6, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %7, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %8 = arith.addi %4, %c1 : index
      %9 = index.casts %8 : index to i32
      %10 = arith.muli %9, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %10, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %11 = arith.addi %4, %c2 : index
      cf.br ^bb5(%11 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb8(%c0 : index)
    ^bb8(%12: index):  // 2 preds: ^bb7, ^bb9
      %13 = arith.cmpi slt, %12, %c4 : index
      cf.cond_br %13, ^bb9, ^bb10
    ^bb9:  // pred: ^bb8
      %14 = index.casts %12 : index to i32
      %15 = arith.muli %14, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %15, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_1) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %16 = arith.addi %12, %c1 : index
      %17 = index.casts %16 : index to i32
      %18 = arith.muli %17, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %18, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_1) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %19 = arith.addi %12, %c2 : index
      cf.br ^bb8(%19 : index)
    ^bb10:  // pred: ^bb8
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      %20 = arith.addi %2, %c2 : index
      cf.br ^bb3(%20 : index)
    ^bb11:  // pred: ^bb3
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb12(%c0 : index)
    ^bb12(%21: index):  // 2 preds: ^bb11, ^bb13
      %22 = arith.cmpi slt, %21, %c4 : index
      cf.cond_br %22, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      %23 = index.casts %21 : index to i32
      %24 = arith.muli %23, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %24, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %25 = arith.addi %21, %c1 : index
      %26 = index.casts %25 : index to i32
      %27 = arith.muli %26, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %27, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %28 = arith.addi %21, %c2 : index
      cf.br ^bb12(%28 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      cf.br ^bb15(%c0 : index)
    ^bb15(%29: index):  // 2 preds: ^bb14, ^bb22
      %30 = arith.cmpi slt, %29, %c4294967294 : index
      cf.cond_br %30, ^bb16, ^bb23
    ^bb16:  // pred: ^bb15
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb17(%c0 : index)
    ^bb17(%31: index):  // 2 preds: ^bb16, ^bb18
      %32 = arith.cmpi slt, %31, %c4 : index
      cf.cond_br %32, ^bb18, ^bb19
    ^bb18:  // pred: ^bb17
      %33 = index.casts %31 : index to i32
      %34 = arith.muli %33, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %34, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_1) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %35 = arith.addi %31, %c1 : index
      %36 = index.casts %35 : index to i32
      %37 = arith.muli %36, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %37, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_1) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %38 = arith.addi %31, %c2 : index
      cf.br ^bb17(%38 : index)
    ^bb19:  // pred: ^bb17
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb20(%c0 : index)
    ^bb20(%39: index):  // 2 preds: ^bb19, ^bb21
      %40 = arith.cmpi slt, %39, %c4 : index
      cf.cond_br %40, ^bb21, ^bb22
    ^bb21:  // pred: ^bb20
      %41 = index.casts %39 : index to i32
      %42 = arith.muli %41, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %42, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %43 = arith.addi %39, %c1 : index
      %44 = index.casts %43 : index to i32
      %45 = arith.muli %44, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %45, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %46 = arith.addi %39, %c2 : index
      cf.br ^bb20(%46 : index)
    ^bb22:  // pred: ^bb20
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      %47 = arith.addi %29, %c2 : index
      cf.br ^bb15(%47 : index)
    ^bb23:  // pred: ^bb15
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb24(%c0 : index)
    ^bb24(%48: index):  // 2 preds: ^bb23, ^bb25
      %49 = arith.cmpi slt, %48, %c4 : index
      cf.cond_br %49, ^bb25, ^bb26
    ^bb25:  // pred: ^bb24
      %50 = index.casts %48 : index to i32
      %51 = arith.muli %50, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %51, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_1) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %52 = arith.addi %48, %c1 : index
      %53 = index.casts %52 : index to i32
      %54 = arith.muli %53, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %54, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_1) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %55 = arith.addi %48, %c2 : index
      cf.br ^bb24(%55 : index)
    ^bb26:  // pred: ^bb24
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      %56 = arith.addi %0, %c2 : index
      cf.br ^bb1(%56 : index)
    ^bb27:  // pred: ^bb1
      cf.br ^bb28(%c0 : index)
    ^bb28(%57: index):  // 2 preds: ^bb27, ^bb35
      %58 = arith.cmpi slt, %57, %c4294967294 : index
      cf.cond_br %58, ^bb29, ^bb36
    ^bb29:  // pred: ^bb28
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb30(%c0 : index)
    ^bb30(%59: index):  // 2 preds: ^bb29, ^bb31
      %60 = arith.cmpi slt, %59, %c4 : index
      cf.cond_br %60, ^bb31, ^bb32
    ^bb31:  // pred: ^bb30
      %61 = index.casts %59 : index to i32
      %62 = arith.muli %61, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %62, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %63 = arith.addi %59, %c1 : index
      %64 = index.casts %63 : index to i32
      %65 = arith.muli %64, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %65, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %66 = arith.addi %59, %c2 : index
      cf.br ^bb30(%66 : index)
    ^bb32:  // pred: ^bb30
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb33(%c0 : index)
    ^bb33(%67: index):  // 2 preds: ^bb32, ^bb34
      %68 = arith.cmpi slt, %67, %c4 : index
      cf.cond_br %68, ^bb34, ^bb35
    ^bb34:  // pred: ^bb33
      %69 = index.casts %67 : index to i32
      %70 = arith.muli %69, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %70, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_1) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %71 = arith.addi %67, %c1 : index
      %72 = index.casts %71 : index to i32
      %73 = arith.muli %72, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %73, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_1) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %74 = arith.addi %67, %c2 : index
      cf.br ^bb33(%74 : index)
    ^bb35:  // pred: ^bb33
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      %75 = arith.addi %57, %c2 : index
      cf.br ^bb28(%75 : index)
    ^bb36:  // pred: ^bb28
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%C_L1L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      cf.br ^bb37(%c0 : index)
    ^bb37(%76: index):  // 2 preds: ^bb36, ^bb38
      %77 = arith.cmpi slt, %76, %c4 : index
      cf.cond_br %77, ^bb38, ^bb39
    ^bb38:  // pred: ^bb37
      %78 = index.casts %76 : index to i32
      %79 = arith.muli %78, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %79, %A_L3L1_0_cons_buff_0, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %80 = arith.addi %76, %c1 : index
      %81 = index.casts %80 : index to i32
      %82 = arith.muli %81, %c32_i32 : i32
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @op1_matvec_vectorized_bf16_bf16(%c32_i32, %82, %A_L3L1_0_cons_buff_1, %B_L3L1_0_cons_buff_0, %C_L1L3_0_buff_0) : (i32, i32, memref<32x128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, Release, 1)
      %83 = arith.addi %76, %c2 : index
      cf.br ^bb37(%83 : index)
    ^bb39:  // pred: ^bb37
      aie.use_lock(%C_L1L3_0_cons_lock_0, Release, 1)
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, Release, 1)
      aie.end
    } {link_files = ["op1_gemv_128k_64vs.o"]}
    aie.runtime_sequence(%arg0: memref<16384xbf16>, %arg1: memref<128xbf16>, %arg2: memref<128xbf16>) {
      %0 = aiex.dma_configure_task_for @B_L3L1_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<128xbf16>, 0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @A_L3L1_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<16384xbf16>, 0, 16384, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16384, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @C_L1L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<128xbf16>, 0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%0)
    }
    aie.shim_dma_allocation @A_L3L1_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_0_cons_buff_0 : memref<32x128xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L1_0_cons_buff_1 : memref<32x128xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%B_L3L1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L1_0_cons_buff_0 : memref<128xbf16>, 0, 128) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L3L1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%C_L1L3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_0_buff_0 : memref<128xbf16>, 0, 128) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%C_L1L3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L3_0_buff_1 : memref<128xbf16>, 0, 128) {bd_id = 4 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%C_L1L3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb8:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @B_L3L1_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @C_L1L3_0_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
  aie.device(npu2) {
    aie.runtime_sequence(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>, %arg2: memref<32896xbf16>) {
      aiex.configure @op0_GEMV {
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16384], strides: [1] : memref<32896xbf16> to memref<16384xbf16>
        %subview = memref.subview %arg2[16384] [128] [1] : memref<32896xbf16> to memref<128xbf16, strided<[1], offset: 16384>>
        %reinterpret_cast_0 = memref.reinterpret_cast %subview to offset: [0], sizes: [128], strides: [1] : memref<128xbf16, strided<[1], offset: 16384>> to memref<128xbf16>
        aiex.run @sequence(%reinterpret_cast, %arg0, %reinterpret_cast_0) : (memref<16384xbf16>, memref<128xbf16>, memref<128xbf16>)
      }
      aiex.configure @op1_GEMV {
        %subview = memref.subview %arg2[16512] [16384] [1] : memref<32896xbf16> to memref<16384xbf16, strided<[1], offset: 16512>>
        %reinterpret_cast = memref.reinterpret_cast %subview to offset: [0], sizes: [16384], strides: [1] : memref<16384xbf16, strided<[1], offset: 16512>> to memref<16384xbf16>
        %subview_0 = memref.subview %arg2[16384] [128] [1] : memref<32896xbf16> to memref<128xbf16, strided<[1], offset: 16384>>
        %reinterpret_cast_1 = memref.reinterpret_cast %subview_0 to offset: [0], sizes: [128], strides: [1] : memref<128xbf16, strided<[1], offset: 16384>> to memref<128xbf16>
        aiex.run @sequence(%reinterpret_cast, %reinterpret_cast_1, %arg1) : (memref<16384xbf16>, memref<128xbf16>, memref<128xbf16>)
      }
    }
  }
}
