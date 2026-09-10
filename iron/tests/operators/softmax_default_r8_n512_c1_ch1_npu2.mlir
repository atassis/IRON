module {
  aie.device(npu2) {
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_0 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in1_0_0(%logical_shim_noc, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<512xbf16>>  
    aie.objectfifo @out_0_0(%logical_core, {%logical_shim_noc_0}, 2 : i32) : !aie.objectfifo<memref<512xbf16>>  
    func.func private @softmax_bf16(memref<512xbf16>, memref<512xbf16>, i32) attributes {link_with = "softmax.o"}
    func.func private @mask_bf16(memref<512xbf16>, i32, i32) attributes {link_with = "softmax.o"}
    %rtp_0_0 = aie.buffer(%logical_core) {sym_name = "rtp_0_0"} : memref<1xi32> 
    %0 = aie.lock(%logical_core)
    %1 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%0, Acquire, %c1_i32)
        %c0_1 = arith.constant 0 : index
        %2 = memref.load %rtp_0_0[%c0_1] : memref<1xi32>
        %c0_2 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c8 step %c1_3 {
          %3 = aie.objectfifo.acquire @in1_0_0(Consume, 1) : memref<512xbf16>
          %4 = aie.objectfifo.acquire @out_0_0(Produce, 1) : memref<512xbf16>
          %c512_i32 = arith.constant 512 : i32
          func.call @mask_bf16(%3, %2, %c512_i32) : (memref<512xbf16>, i32, i32) -> ()
          %c512_i32_4 = arith.constant 512 : i32
          func.call @softmax_bf16(%3, %4, %c512_i32_4) : (memref<512xbf16>, memref<512xbf16>, i32) -> ()
          aie.objectfifo.release @in1_0_0(Consume, 1)
          aie.objectfifo.release @out_0_0(Produce, 1)
        }
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<4096xbf16>, %arg1: memref<4096xbf16>) {
      %c512_i32 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_0_0, 0, %c512_i32) : i32
      aiex.set_lock(%0, 1)
      %2 = aiex.dma_configure_task_for @in1_0_0 {
        aie.dma_bd(%arg0 : memref<4096xbf16> offset = 0 len = 4096 sizes = [1, 1, 1, 4096] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @out_0_0 {
        aie.dma_bd(%arg1 : memref<4096xbf16> offset = 0 len = 4096 sizes = [1, 1, 1, 4096] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%3)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
    }
  }
}
