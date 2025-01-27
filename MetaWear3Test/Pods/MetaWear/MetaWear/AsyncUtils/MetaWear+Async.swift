/**
 * MetaWear+Async.swift
 * MetaWear
 *
 * Created by Stephen Schiffli on 5/3/18.
 * Copyright 2018 MbientLab Inc. All rights reserved.
 *
 * IMPORTANT: Your use of this Software is limited to those specific rights
 * granted under the terms of a software license agreement between the user who
 * downloaded the software, his/her employer (which must be your employer) and
 * MbientLab Inc, (the "License").  You may not use this Software unless you
 * agree to abide by the terms of the License which can be found at
 * www.mbientlab.com/terms.  The License limits your use, and you acknowledge,
 * that the Software may be modified, copied, and distributed when used in
 * conjunction with an MbientLab Inc, product.  Other than for the foregoing
 * purpose, you may not use, reproduce, copy, prepare derivative works of,
 * modify, distribute, perform, display or sell this Software and/or its
 * documentation for any purpose.
 *
 * YOU FURTHER ACKNOWLEDGE AND AGREE THAT THE SOFTWARE AND DOCUMENTATION ARE
 * PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTY OF MERCHANTABILITY, TITLE,
 * NON-INFRINGEMENT AND FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL
 * MBIENTLAB OR ITS LICENSORS BE LIABLE OR OBLIGATED UNDER CONTRACT, NEGLIGENCE,
 * STRICT LIABILITY, CONTRIBUTION, BREACH OF WARRANTY, OR OTHER LEGAL EQUITABLE
 * THEORY ANY DIRECT OR INDIRECT DAMAGES OR EXPENSES INCLUDING BUT NOT LIMITED
 * TO ANY INCIDENTAL, SPECIAL, INDIRECT, PUNITIVE OR CONSEQUENTIAL DAMAGES, LOST
 * PROFITS OR LOST DATA, COST OF PROCUREMENT OF SUBSTITUTE GOODS, TECHNOLOGY,
 * SERVICES, OR ANY CLAIMS BY THIRD PARTIES (INCLUDING BUT NOT LIMITED TO ANY
 * DEFENSE THEREOF), OR OTHER SIMILAR COSTS.
 *
 * Should you have any questions regarding your right to use this Software,
 * contact MbientLab via email: hello@mbientlab.com
 */

import MetaWearCpp
import BoltsSwift


extension MetaWear {
    public func clearAndReset() {
        mbl_mw_logging_stop(board)
        mbl_mw_metawearboard_tear_down(board)
        mbl_mw_logging_clear_entries(board)
        mbl_mw_macro_erase_all(board)
        mbl_mw_debug_reset_after_gc(board)
        mbl_mw_debug_disconnect(board)
    }
    
    public func macroEndRecord() -> Task<Int32> {
        let source = TaskCompletionSource<Int32>()
        mbl_mw_macro_end_record(board, bridgeRetained(obj: source)) { (context, board, value) in
            let source: TaskCompletionSource<Int32> = bridgeTransfer(ptr: context!)
            source.set(result: value)
        }
        return source.task
    }
    
    public func createAnonymousDatasignals() -> Task<[OpaquePointer]> {
        let source = TaskCompletionSource<[OpaquePointer]>()
        mbl_mw_metawearboard_create_anonymous_datasignals(board, bridgeRetained(obj: source))
        { (context, board, anonymousSignals, size) in
            let source: TaskCompletionSource<[OpaquePointer]> = bridgeTransfer(ptr: context!)
            if let anonymousSignals = anonymousSignals {
                if size == 0 {
                    source.set(error: MetaWearError.operationFailed(
                        message: "device is not logging any sensor data"))
                } else {
                    let array = (0..<size).map { anonymousSignals[Int($0)]! }
                    source.set(result: array)
                }
            } else {
                source.set(error: MetaWearError.operationFailed(
                    message: "failed to create anonymous data signals (status = \(size))"))
            }
        }
        return source.task
    }
    
    public func timerCreate(period: UInt32, repetitions: UInt16 = 0xFFFF, immediateFire: Bool = false) -> Task<OpaquePointer> {
        let source = TaskCompletionSource<OpaquePointer>()
        mbl_mw_timer_create(board, period, repetitions, immediateFire ? 0 : 1, bridgeRetained(obj: source)) { (context, timer) in
            let source: TaskCompletionSource<OpaquePointer> = bridgeTransfer(ptr: context!)
            if let timer = timer {
                source.set(result: timer)
            } else {
                source.set(error: MetaWearError.operationFailed(message: "could not create timer"))
            }
        }
        return source.task
    }
}
