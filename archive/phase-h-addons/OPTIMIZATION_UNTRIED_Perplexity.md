# Optimization ที่ยังไม่ได้ลอง — Phase H

วันที่: 2026-09-07
แหล่งข้อมูลหลัก: `PERF_EXPERIMENTS_PHASE_H.md`

## Executive summary

จากสถานะล่าสุด ยังมี optimization หลายกลุ่มที่ยังไม่ได้ทดลองจริง แต่หลังจากเพิ่ม GPU timestamp แล้วพบว่า **GPU compute ใช้เพียงประมาณ 5% ของเวลา create** (2048²: 93.4 ms wall เทียบกับ 5.0 ms GPU-busy; 4096²: 344 ms เทียบกับ 21.4 ms) ดังนั้นควรไล่จาก host-side preparation, memory และ allocation ก่อน shader optimization.

สิ่งที่ควรลองเป็นลำดับแรก:

1. วัดและแก้ `prep` อย่างเป็นระบบ โดยเฉพาะ page faults, mapping และ allocator churn.
2. ทำ staging ring / scratch arena แบบ persistent.
3. เพิ่ม zero-copy path สำหรับ UMA และ ReBAR/host-visible device-local memory.
4. ลด descriptor และ barrier overhead.
5. ค่อยทดลอง 2D dispatch, shared-memory fusion และ specialization constants หลังมี baseline ที่ถูกต้อง.

## รายการที่ยังไม่ได้ลอง

### 1. Persistent staging ring และ scratch arena

**แนวคิด:** ใช้ host-visible staging buffer ขนาดใหญ่แบบ ring ที่สร้างครั้งเดียว แล้ว reuse สำหรับ upload หลายรอบ แทนการ allocate/map/free ต่อ call พร้อมทำ device-local scratch arena แบบ sub-allocation.

**เหตุผลที่น่าลองตอนนี้:** timestamp ชี้ว่า `prep` ใช้เวลาประมาณ 70–90 ms ที่ 2048² และมีการ upload pyramid ขนาดราว 89 MB การสร้าง allocation ใหม่และ first-touch page fault จึงเป็นผู้ต้องสงสัยหลัก.

**สิ่งที่ควรวัด:**

- เวลา allocate, map, pack/interleave, flush และ submit แยกกัน.
- minor/major page faults ระหว่าง first run กับ warm run.
- ผลของการ reuse allocation เทียบกับ allocation ใหม่.
- cached mapping เทียบกับ write-combined mapping หาก platform รองรับ.

**ความเสี่ยง:** ต้องจัดการ ring wraparound, lifetime ของ GPU และ non-coherent flush/invalidate ให้ถูกต้อง.

**เกณฑ์ผ่าน:** ลดเวลา `prep` อย่างมีนัยสำคัญโดย parity ไม่เปลี่ยน และ peak allocation ไม่เพิ่มตามจำนวน scale.

### 2. Zero-copy สำหรับ UMA และ memory ที่ host-visible

**แนวคิด:** บนอุปกรณ์ UMA หรือ discrete GPU ที่มี memory type เป็นทั้ง `DEVICE_LOCAL | HOST_VISIBLE` ให้เขียนข้อมูลตรงเข้า allocation ที่ GPU ใช้ได้ โดยข้าม staging copy.

**เหตุผลที่น่าลอง:** เอกสารระบุว่า UMA ไม่จำเป็นต้องใช้ staging และลด transfer overhead ได้มาก นอกจากนี้ discrete GPU ที่มี ReBAR อาจได้ประโยชน์เช่นกัน แต่ต้องตรวจ memory flags ด้วย `contains()` ไม่ใช่ OR.

**สิ่งที่ควรทำ:**

- แยก path ตาม memory properties จริงของ adapter.
- ทดสอบ discrete non-ReBAR ให้ตกกลับไป staging ตามปกติ.
- วัด cold run และ warm run แยกกัน.

**ความเสี่ยง:** throughput ของ host-visible memory อาจต่ำกว่า device-local และต้อง flush ให้ถูกต้อง.

### 3. Descriptor-set preallocation และ dynamic offsets

**แนวคิด:** pre-allocate descriptor sets ต่อ frame/pass slot หรือใช้ dynamic storage-buffer offsets ใน arena เดียว แทนการ allocate/reset/update descriptors ทุก dispatch.

**เหตุผลที่น่าลอง:** ปัจจุบันยังมี allocation-per-pass และ pool reset-per-submit; จึงมี CPU driver overhead เหลืออยู่ โดยเฉพาะภาพเล็ก.

**แนวทางทดลอง:**

- สร้าง descriptor pool แบบ persistent.
- จัด buffer offsets ให้ตรง `minStorageBufferOffsetAlignment`.
- ใช้ push constants สำหรับค่าที่เปลี่ยนบ่อยต่อ pass.
- ตรวจ CPU trace ว่า steady state ไม่มี `vkUpdateDescriptorSets` ที่ไม่จำเป็น.

**ความเสี่ยง:** ต่ำถึงปานกลาง แต่ต้องตรวจ lifetime ของ resource และความเข้ากันได้ของ pipeline layout.

### 4. Timeline semaphore หรือ fence ring

**แนวคิด:** ลดการรอ fence แบบ synchronous ด้วย timeline semaphore หรือ fence ring แล้วทำ async readback/pipeline งาน CPU กับ GPU.

**เหตุผลที่ยังพอมีประโยชน์:** เดิมออกแบบมาเพื่อแก้กรณี 120+ submits แต่หลัง T1 เหลือเพียง 1–2 submits ต่อ comparison แล้ว จึงไม่ใช่ตัวเร่งหลักอีกต่อไป.

**ควรลองเมื่อ:** มีหลายภาพหรือ batch workload ที่ทำให้ overlap ระหว่าง comparison ได้จริง.

**เกณฑ์ผ่าน:** ลด idle/stall ที่วัดได้ ไม่ใช่เพียงเปลี่ยนโค้ด synchronization.

### 5. Tighten pipeline barriers

**แนวคิด:** ลด barrier ที่กว้างเกินไป โดยใช้ source/destination stage และ access mask ตาม dependency จริง เช่น COMPUTE→COMPUTE สำหรับ intermediate และ TRANSFER เฉพาะตอน copy.

**เหตุผลที่ยังไม่ได้ revisit:** `dispatch_sequence` ยังทำให้ buffer ที่ pass แตะต้องมองเห็นได้กับ consumer ถัด ๆ ไปอย่างกว้าง อาจมีประมาณ 40 full barriers ต่อ create.

**สิ่งที่ต้องระวัง:** ห้ามลบ barrier จาก write→read dependency; ใช้ validation และ GPU timestamp ตรวจว่าการลด barrier ไม่ทำให้เกิด race.

### 6. 2D dispatch และ workgroup tuning

**แนวคิด:** เปลี่ยนจาก 1D dispatch ที่คำนวณ `x/y` ด้วย division และ modulo เป็น 2D dispatch เช่น 16×16 และทดลองขนาด 128/256 threads ต่อ workgroup.

**สถานะ:** ยังไม่ได้ลอง และทุก shader ยังเป็น local size 64 แบบ 1D.

**ข้อควรทราบ:** จาก timestamp ล่าสุด shader compute ไม่ใช่ bottleneck หลัก จึงควรทำหลังแก้ `prep` และวัดผลใหม่.

**เกณฑ์ผ่าน:** parity เหมือนเดิมและ GPU-busy ดีขึ้นอย่างวัดได้ โดยไม่อ้าง wall-time ที่ถูก host-side เตรียมข้อมูลบัง.

### 7. Fusion ระหว่าง pass ด้วย shared memory

**แนวคิด:** ใช้ shared-memory tile เพื่อลดการเขียน intermediate ลง global memory เช่น fuse H5/V5 หรือ fuse blur กับ combine.

**แนวคิดย่อยที่เสี่ยงน้อยกว่า:** `h5(img)` สำหรับ mu และ `h5_mul(img,img)` สำหรับ sq อ่าน plane เดียวกัน จึงอาจ fuse การอ่าน input ได้ โดยคง expression และลำดับการคำนวณเดิม.

**สถานะ:** ยังไม่มี `shared` declaration ใน shader.

**ความเสี่ยง:** สูงมากต่อ bitwise parity โดยเฉพาะบริเวณ sigma cancellation และ boundary taps.

**เกณฑ์ผ่าน:** ต้อง bitwise-equal กับ shader เดิมบน fixtures ทั้งหมดก่อนวัด performance; ห้ามเพิ่ม tolerance เพื่อให้ผ่าน.

### 8. Specialization constants / pipeline variants

**แนวคิด:** ทำ pipeline variant สำหรับค่าที่คงที่ใน workload เช่น channels, workgroup size หรือรูปแบบภาพ เพื่อให้ driver/compiler constant-fold การคำนวณบางส่วน.

**เหตุผล:** เป็น idea ใหม่ที่ยังไม่อยู่ใน track เดิม โดย shader อ่าน width/height/stride และพารามิเตอร์จาก push constants ต่อ invocation.

**ข้อเสีย:** จำนวน pipeline อาจเพิ่มแบบคูณกัน และไม่คุ้มถ้าขนาดภาพเปลี่ยนบ่อย.

**เหมาะกับ:** benchmark หรือ production workload ที่มีชุดขนาดภาพมาตรฐานไม่กี่แบบ.

### 9. Batch mode / resident reference

**แนวคิด:** เปรียบเทียบ original กับหลาย modified images โดยเก็บ reference pyramid ไว้บน GPU เพียงครั้งเดียว.

**ผลที่คาดหวัง:** amortize ค่า create/upload ของ reference และอาจเปิดทางให้ pipeline หลายคู่.

**สถานะ:** ยังไม่มี CLI/API สำหรับ multi-mod batch จึงเป็น feature-level optimization มากกว่า micro-optimization.

**เหมาะกับ:** workflow ที่มีภาพต้นฉบับหนึ่งภาพและต้องเทียบหลายผลลัพธ์จริง.

## สิ่งที่ไม่ควรเริ่มก่อน

- อย่าเริ่มจาก fused blur เต็มรูปแบบก่อนแก้ `prep`; GPU-busy ปัจจุบันต่ำเมื่อเทียบกับ wall time.
- อย่าเพิ่ม workgroup size โดยดูเฉพาะ wall time; ต้องดู GPU timestamp แยก.
- อย่าใช้ benchmark จาก fused-ssim เป็น prediction ตรง ๆ เพราะ algorithm, baseline CPU และ precision ต่างกัน.
- อย่าเพิ่ม tolerance เพื่อรับผลจากการเปลี่ยนลำดับ floating-point operations.

## แผนทดลองที่แนะนำ

| ลำดับ | การทดลอง | เหตุผล | ผลที่ต้องเก็บ |
|---|---|---|---|
| 1 | แยกเวลา prep: allocate/map/pack/flush | ยืนยัน bottleneck | cold/warm, page faults, bytes/s |
| 2 | persistent staging + arena | ตรงกับอาการ allocation/page fault | wall, prep, peak memory |
| 3 | zero-copy memory path | ลด upload/copy โดยตรง | UMA, ReBAR, non-ReBAR |
| 4 | tighten barriers + descriptors | ลด host/driver overheadที่เหลือ | CPU trace, GPU idle |
| 5 | batch resident-reference | amortize fixed cost | pair vs batch 5/10 |
| 6 | 2D dispatch | compute optimization หลัง baseline | GPU-busy, parity |
| 7 | safe mu/sq input-read fusion | ลด traffic ความเสี่ยงต่ำกว่า full fusion | bandwidth, bitwise parity |
| 8 | full shared-memory fusion | upside สูงสุดแต่เสี่ยงสูง | GPU-busy, map equality |
| 9 | specialization constants | optimize fixed-size deployments | pipeline count, compile/startup cost |

## สรุป

คำตอบคือ **ยังมีสิ่งที่ไม่ได้ลองหลายอย่าง** แต่จากข้อมูลล่าสุด ตัวที่ควรลองจริงก่อนคือ persistent staging/arena และ zero-copy path เพราะหลักฐานชี้ไปที่ host-side `prep` ไม่ใช่ shader compute. 2D dispatch, shared-memory fusion และ specialization constants ยังมีคุณค่าเชิงเทคนิค แต่ควรเลื่อนไปหลังจากแก้ memory behavior และมี baseline ที่วัดด้วย GPU timestamps แล้ว.
