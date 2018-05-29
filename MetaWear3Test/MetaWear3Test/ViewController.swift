//
//  ViewController.swift
//  MetaWear3Test
//
//  Created by Andreas Hauenstein on 2018-05-07.
//  Copyright © 2018 AHN. All rights reserved.
//

import UIKit
import MetaWear
import MetaWearCpp
import BoltsSwift

//===================
extension MetaWear
{
    //----------------------------------------------------------------------
    public func getAccuracy(_ data: MblMwSensorFusionData) -> Task<UInt8>
    {
        let source = TaskCompletionSource<UInt8>()
        let signal = mbl_mw_sensor_fusion_get_data_signal(board, data)
        mbl_mw_datasignal_subscribe(signal, bridgeRetained(obj: source)) { (contextPtr, pointer) in
            let source: TaskCompletionSource<UInt8> = bridgeTransfer(ptr: contextPtr!)
            let value: MblMwCorrectedCartesianFloat = pointer!.pointee.valueAs()
            source.trySet(result: value.accuracy)
        }
        return source.task.continueWithTask {
            mbl_mw_datasignal_unsubscribe(mbl_mw_sensor_fusion_get_data_signal(self.board, data))
            return $0
        }
    } // getAccuracy()
} // extension MetaWear

var g_handlers = MblMwLogDownloadHandler()

//=========================================
class ViewController: UIViewController
{
    
    @IBOutlet weak var txtLog: UITextView!
    var ccount:Int64 = 0
    var device:MetaWear!
    var logger_id:UInt8!
//    static var handlers:MblMwLogDownloadHandler!
    
    //-----------------------------
    override func viewDidLoad()
    {
        super.viewDidLoad()
        g_handlers.context = bridge(obj: self)
        g_handlers.received_progress_update = { (context, entriesLeft, totalEntries) in
            let this: ViewController = bridge(ptr: context!)
            print( String( format: ">>>>>>>>> left: %d / %d", entriesLeft, totalEntries))
            if entriesLeft == 0 {
                this.device.clearAndReset()
            }
        }
        g_handlers.received_unknown_entry = { (context, id, epoch, data, length) in
            let this: ViewController = bridge(ptr: context!)
            var tt = 42
            tt += 1
        }
        g_handlers.received_unhandled_entry = { (context, data) in
            let this: ViewController = bridge(ptr: context!)
            var tt = 42
            tt += 1
        }

    } // viewDidLoad()
    
    // Show a string on our textview
    //-------------------------------
    func pr(_ s:String) -> Void
    {
        DispatchQueue.main.async {
            self.txtLog.text.append( "\n\(s)")
        }
    }
    
    //---------------------------
    func clr() -> Void
    {
        txtLog.text = ""
    }
    
    //-----------------------------------------
    func startAcc( device:MetaWear) -> Void
    {
        // Configuring the accelerometer
        mbl_mw_acc_set_range( device.board, 2.0)
        mbl_mw_acc_set_odr( device.board, 10)
        mbl_mw_acc_write_acceleration_config( device.board)
        
        let accSignal = mbl_mw_acc_get_acceleration_data_signal( device.board)
        mbl_mw_datasignal_subscribe( accSignal, bridge(obj: self)) { (context, dataPtr) in
            // Since this is a C callback, it can't capture outside variables,
            // so you need to pass in a ‘context’ pointer.  It’s just a void*,
            // so it can point to any type, just cast to back to what you expect here.
            let this:ViewController = bridge( ptr: context!)
            print( dataPtr!.pointee.valueAs() as MblMwCartesianFloat)
            // The dataPtr is only valid during the duration of this callback,
            // if you need to keep the data longer, you must copy it somewhere else
            //this.data.append( dataPtr!.pointee.copy())
        }
        // Starting the accelerometer
        mbl_mw_acc_enable_acceleration_sampling( device.board)
        mbl_mw_acc_start( device.board)
    } // startAcc()
    
    //---------------------------------------
    func stopAcc( device:MetaWear) -> Void
    {
        mbl_mw_acc_stop( device.board)
        mbl_mw_acc_disable_acceleration_sampling( device.board)
        let accSignal = mbl_mw_acc_get_acceleration_data_signal( device.board)
        mbl_mw_datasignal_unsubscribe( accSignal)
    } // stopAcc()
    
    //-------------------------------------------
    func startFusion( device:MetaWear) -> Void
    {
        // Configuration
        mbl_mw_sensor_fusion_set_mode( device.board, MBL_MW_SENSOR_FUSION_MODE_NDOF)
        mbl_mw_sensor_fusion_set_acc_range( device.board, MBL_MW_SENSOR_FUSION_ACC_RANGE_2G)
        mbl_mw_sensor_fusion_write_config( device.board)
        
        let sig = mbl_mw_sensor_fusion_get_data_signal( device.board,
                                                        MBL_MW_SENSOR_FUSION_DATA_LINEAR_ACC)
        
        self.ccount = 0
        mbl_mw_datasignal_subscribe( sig, bridge(obj: self), { (context, dataPtr) in
            let this:ViewController = bridge( ptr: context!)
            this.ccount += 1
            let acc = dataPtr!.pointee.valueAs() as MblMwCartesianFloat
            if this.ccount % 10 == 0 {
                print( acc)
            }
        })
        
        // Start
        mbl_mw_sensor_fusion_enable_data( device.board, MBL_MW_SENSOR_FUSION_DATA_LINEAR_ACC)
        mbl_mw_sensor_fusion_start( device.board)
        
    } // startFusion()
    
    //--------------------------------------
    func stopFusion( device:MetaWear) -> Void
    {
        mbl_mw_sensor_fusion_stop( device.board)
        mbl_mw_sensor_fusion_clear_enabled_mask( device.board)
        let sig = mbl_mw_sensor_fusion_get_data_signal( device.board,
                                                        MBL_MW_SENSOR_FUSION_DATA_LINEAR_ACC)
        mbl_mw_datasignal_unsubscribe( sig)
    } // stopFusion()
    
    //-------------------------------------------
    func startQuat( device:MetaWear) -> Void
    {
        // Configuration
        mbl_mw_sensor_fusion_set_mode( device.board, MBL_MW_SENSOR_FUSION_MODE_IMU_PLUS)
        //mbl_mw_sensor_fusion_set_acc_range( device.board, MBL_MW_SENSOR_FUSION_ACC_RANGE_2G)
        mbl_mw_sensor_fusion_set_gyro_range( device.board, MBL_MW_SENSOR_FUSION_GYRO_RANGE_2000DPS)
        mbl_mw_sensor_fusion_write_config( device.board)
        
        let sig = mbl_mw_sensor_fusion_get_data_signal( device.board,
                                                        MBL_MW_SENSOR_FUSION_DATA_QUATERION)
        
        self.ccount = 0
        mbl_mw_datasignal_subscribe( sig, bridge(obj: self), { (context, dataPtr) in
            let this:ViewController = bridge( ptr: context!)
            this.ccount += 1
            let quat = dataPtr!.pointee.valueAs() as MblMwQuaternion
            if this.ccount % 10 == 0 {
                print( quat.x)
            }
        })
        
        // Start
        mbl_mw_sensor_fusion_enable_data( device.board, MBL_MW_SENSOR_FUSION_DATA_QUATERION)
        mbl_mw_sensor_fusion_start( device.board)
        
    } // startQuat()
    
    //--------------------------------------
    func stopQuat( device:MetaWear) -> Void
    {
        mbl_mw_sensor_fusion_stop( device.board)
        mbl_mw_sensor_fusion_clear_enabled_mask( device.board)
        let sig = mbl_mw_sensor_fusion_get_data_signal( device.board,
                                                        MBL_MW_SENSOR_FUSION_DATA_QUATERION)
        mbl_mw_datasignal_unsubscribe( sig)
    } // stopQuat()
    
    //-------------------------------------------
    func startEuler( device:MetaWear) -> Void
    {
        // Configuration
        mbl_mw_sensor_fusion_set_mode( device.board, MBL_MW_SENSOR_FUSION_MODE_NDOF)
        //mbl_mw_sensor_fusion_set_acc_range( device.board, MBL_MW_SENSOR_FUSION_ACC_RANGE_2G)
        mbl_mw_sensor_fusion_write_config( device.board)
        
        let sig = mbl_mw_sensor_fusion_get_data_signal( device.board,
                                                        MBL_MW_SENSOR_FUSION_DATA_EULER_ANGLE)
        
        self.ccount = 0
        mbl_mw_datasignal_subscribe( sig, bridge(obj: self), { (context, dataPtr) in
            let this:ViewController = bridge( ptr: context!)
            this.ccount += 1
            let euler = dataPtr!.pointee.valueAs() as MblMwEulerAngles
            if this.ccount % 10 == 0 {
                print( euler.pitch)
            }
        })
        
        // Start
        mbl_mw_sensor_fusion_enable_data( device.board, MBL_MW_SENSOR_FUSION_DATA_EULER_ANGLE)
        mbl_mw_sensor_fusion_start( device.board)
        
    } // startEuler()
    
    //--------------------------------------
    func stopEuler( device:MetaWear) -> Void
    {
        mbl_mw_sensor_fusion_stop( device.board)
        mbl_mw_sensor_fusion_clear_enabled_mask( device.board)
        let sig = mbl_mw_sensor_fusion_get_data_signal( device.board,
                                                        MBL_MW_SENSOR_FUSION_DATA_EULER_ANGLE)
        mbl_mw_datasignal_unsubscribe( sig)
    } // stopEuler()
    
    // IBActions
    //==============
    
    //-------------------------------------
    @IBAction func btnClr(_ sender: Any)
    {
        clr()
    } // btnClr()
    
    
    
    // Get battery state, stream quaternions
    //---------------------------------------
    @IBAction func btnGo(_ sender: Any)
    {
        MetaWearScanner.shared.startScan(allowDuplicates: true) { (device) in
            // We found a MetaWear board, see if it is close
            if device.rssi > -70 {
                // We found a MetaWear board, so stop scanning for more
                MetaWearScanner.shared.stopScan()
                // Connect to the board we found
                device.connectAndSetup().continueWith { t in
                    if let error = t.error {
                        // Sorry we couldn't connect
                        print(error)
                    }
                    else {
                        // We connected to a MetaWear board
                        var pattern = MblMwLedPattern()
                        mbl_mw_led_load_preset_pattern( &pattern, MBL_MW_LED_PRESET_PULSE)
                        mbl_mw_led_stop_and_clear( device.board)
                        mbl_mw_led_write_pattern( device.board, &pattern, MBL_MW_LED_COLOR_GREEN)
                        mbl_mw_led_play( device.board)
                        mbl_mw_led_stop( device.board)
                        mbl_mw_settings_get_battery_state_data_signal( device.board).read().continueOnSuccessWith {
                            let battery: MblMwBatteryState = $0.valueAs()
                            print( battery.charge)
                        }
                        self.startQuat( device: device)
                        DispatchQueue.main.asyncAfter( deadline: .now() + 300.0, execute: {
                            self.stopQuat( device: device)
                        })
                    }
                } // connectAndSetup()
            } // if rssi
        } // startscan()
    } // btnGo()
    
    var logger:OpaquePointer!
    var accSignal:OpaquePointer!
    
    class func received_progress_update (context:UnsafeMutableRawPointer?, entriesLeft:UInt32, totalEntries:UInt32) {
        let this:ViewController = bridge(ptr: context!)
        print( String( format: ">>>>>>>>> left: %d / %d", entriesLeft, totalEntries))
        if entriesLeft == 0 {
            this.device.clearAndReset()
        }
    }
    
    // Start accelerometer logging //@@@
    //--------------------------------------------
    @IBAction func btnLogAccel(_ sender: Any) {
        MetaWearScanner.shared.startScan(allowDuplicates: true) { (device) in
            if device.rssi > -70 {
                MetaWearScanner.shared.stopScan()
                // Connect to the board we found
                device.connectAndSetup().continueWith { t in
                    if let error = t.error {
                        print(error); return
                    }
                    self.device = device
                    //mbl_mw_metawearboard_tear_down( self.device.board)
                    //mbl_mw_logging_stop( self.device.board)
                    //self.device.clearAndReset()
                    mbl_mw_logging_stop(self.device.board)
                    mbl_mw_metawearboard_tear_down(self.device.board)
                    mbl_mw_logging_clear_entries(self.device.board)
                    mbl_mw_macro_erase_all(self.device.board)


                    // Configuring sensor fusion
                    mbl_mw_sensor_fusion_set_mode( device.board, MBL_MW_SENSOR_FUSION_MODE_IMU_PLUS)
                    //mbl_mw_sensor_fusion_set_acc_range( device.board, MBL_MW_SENSOR_FUSION_ACC_RANGE_2G)
                    mbl_mw_sensor_fusion_set_gyro_range( device.board, MBL_MW_SENSOR_FUSION_GYRO_RANGE_2000DPS)
                    mbl_mw_sensor_fusion_write_config( device.board)
                    
                    self.accSignal = mbl_mw_sensor_fusion_get_data_signal( device.board,
                                                                    MBL_MW_SENSOR_FUSION_DATA_LINEAR_ACC)
                    
                    // Start
                    mbl_mw_sensor_fusion_enable_data( device.board, MBL_MW_SENSOR_FUSION_DATA_LINEAR_ACC)
                    mbl_mw_sensor_fusion_start( device.board)

                    
//                    mbl_mw_acc_set_range( device.board, 2.0)
//                    mbl_mw_acc_set_odr( device.board, 100)
//                    mbl_mw_acc_write_acceleration_config( device.board)
                    
//                    self.accSignal = mbl_mw_acc_get_acceleration_data_signal( device.board)

                    // Start the accelerometer
//                    mbl_mw_acc_enable_acceleration_sampling( device.board)
//                    mbl_mw_acc_start( device.board)
                    
                    print("1")
                    mbl_mw_logging_clear_entries( self.device.board)
                    print("2")
                    mbl_mw_datasignal_log( self.accSignal, bridge(obj: self)) { (context, logger) in
                        let this:ViewController = bridge( ptr: context!)
                        if let logger = logger {
                            this.logger = logger
                            this.logger_id = mbl_mw_logger_get_id( logger)
                            //mbl_mw_logging_download(this.device.board!, 1, &g_handlers)
                            print("3")
//                            mbl_mw_logger_subscribe( this.logger, context) { (context, dataPtr) in
//                                print("4")
//                                let this:ViewController = bridge( ptr: context!)
//                                let xyz = dataPtr!.pointee.valueAs() as MblMwCartesianFloat
//                                //this.pr( String(format: "%.2f %.2f %.2f", xyz.x, xyz.y, xyz.z))
//                                print( String(format: "%d %.2f %.2f %.2f", this.ccount, xyz.x, xyz.y, xyz.z))
//                                this.ccount += 1
//                            }
                            print("5")
                            mbl_mw_logging_start( this.device.board, 1)
                            print("6")
                            this.wait_then_download()
                            print("7")
                        }
                        else {
                            this.pr( "failed to get logger")
                            //mbl_mw_metawearboard_tear_down( this.device.board)
                            this.device.clearAndReset()
                            //this.device.cancelConnection()
                        }
                    } // mbl_mw_datasignal_log()
                } // connectAndSetup()
            } // id(rssi)
        } // startScan
    } // btnLogAccel()
    
    
    //------------------------------
    func wait_then_download() {
        DispatchQueue.main.asyncAfter(deadline: DispatchTime.now() + .seconds(5) )
        {
            print( ">>>>>>> timeout")
            mbl_mw_logger_subscribe( self.logger, bridge(obj: self)) { (context, data) in
                let this:ViewController = bridge( ptr: context!)
                let xyz = data!.pointee.valueAs() as MblMwCartesianFloat
                //this.pr( String(format: "%.2f %.2f %.2f", xyz.x, xyz.y, xyz.z))
                print( String(format: "%d %.2f %.2f %.2f", this.ccount, xyz.x, xyz.y, xyz.z))
                this.ccount += 1
            }
            
            // Setup the handlers for events during the download
            var handlers = MblMwLogDownloadHandler()
            handlers.context = bridge(obj: self)
            handlers.received_progress_update = { (context, entriesLeft, totalEntries) in
                let this:ViewController = bridge( ptr: context!)
                print( String( format: ">>>>>>>>> left: %d / %d", entriesLeft, totalEntries))
                if entriesLeft == 0 {
                    this.device.clearAndReset()
                }
            }
            handlers.received_unknown_entry = { (context, id, epoch, data, length) in
                let this:ViewController = bridge( ptr: context!)
            }
            handlers.received_unhandled_entry = { (context, data) in
                let this:ViewController = bridge( ptr: context!)
            }
            
            // Start the log download
            mbl_mw_logging_download(self.device.board!, 100, &handlers)

//            self.logger = mbl_mw_logger_lookup_id( self.device.board, self.logger_id)
            //mbl_mw_logging_stop( self.device.board)
            // Setup the handlers for events during the download
            
//            // Start the log download
//            self.ccount = 0
//            mbl_mw_logging_download(self.device.board!, 100, &g_handlers)
//            var tt = 42
//            tt += 1
        }
    } // wait_then_download
    
    
    // Stop logging and download data
    //---------------------------------------------
    @IBAction func btnDownload(_ sender: Any) {
        
    } // btnDownload()
    
    // Query sensors for accuracy
    //----------------------------------------
    @IBAction func btnCalib(_ sender: Any)
    {
        MetaWearScanner.shared.startScan(allowDuplicates: true) { (device) in
            // We found a MetaWear board, see if it is close
            if device.rssi > -70 {
                // Hooray! We found a MetaWear board, so stop scanning for more
                MetaWearScanner.shared.stopScan()
                // Connect to the board we found
                device.connectAndSetup().continueWith { t in
                    if let error = t.error {
                        // Sorry we couldn't connect
                        print(error)
                    }
                    else {
                        // We connected to a MetaWear board
                        let board = device.board
                        mbl_mw_sensor_fusion_enable_data(board, MBL_MW_SENSOR_FUSION_DATA_CORRECTED_ACC)
                        mbl_mw_sensor_fusion_enable_data(board, MBL_MW_SENSOR_FUSION_DATA_CORRECTED_GYRO)
                        mbl_mw_sensor_fusion_enable_data(board, MBL_MW_SENSOR_FUSION_DATA_CORRECTED_MAG)
                        mbl_mw_sensor_fusion_start(board)
                        
                        device.getAccuracy(MBL_MW_SENSOR_FUSION_DATA_CORRECTED_ACC).continueOnSuccessWith {
                            self.pr( "accel: \($0)")
                        }
                        device.getAccuracy(MBL_MW_SENSOR_FUSION_DATA_CORRECTED_GYRO).continueOnSuccessWith {
                            self.pr( "gyro : \($0)")
                        }
                        device.getAccuracy(MBL_MW_SENSOR_FUSION_DATA_CORRECTED_MAG).continueOnSuccessWith {
                            self.pr( "mag  : \($0)")
                        }
                    }
                } // connectAndSetup
            } // if rssi
        } // startScan
    } // btnCalib()
    
    // Blink led green for ms milliseconds
    //-------------------------------------------
    func led_on(_ device:MetaWear, _ ms:UInt16)
    {
        let b = device.board
        var p = MblMwLedPattern()
        mbl_mw_led_load_preset_pattern( &p, MBL_MW_LED_PRESET_BLINK)
        p.high_time_ms = ms
        p.pulse_duration_ms = ms+1
        p.repeat_count = 1
        mbl_mw_led_stop_and_clear( b)
        mbl_mw_led_write_pattern( b, &p, MBL_MW_LED_COLOR_GREEN)
        mbl_mw_led_play( b)
    } // led_on
    
    var counter = 0
    var devices:[MetaWear?] = []
    // Connect and disconnect in a loop
    //-------------------------------------------
    @IBAction func btn_cloop(_ sender: Any)
    {
        self.counter = 0
        self.devices.removeAll()
        MetaWearScanner.shared.startScan(allowDuplicates: true) { (device) in
            // We found a MetaWear board, see if it is close
            if device.rssi > -70 {
                if !self.devices.contains { $0 === device }
                {
                    self.devices.append( device)
                    device.connectAndSetup().continueWith { t in
                        if let error = t.error { self.pr( error.localizedDescription) }
                        else {
                            self.pr( "connect \(self.counter) #devices: \(self.devices.count)")
                            self.counter += 1
                            self.led_on( device, 20000)
                            //device.cancelConnection()
                            let idx = self.devices.index { $0 === device }
                            if idx != nil {
                                self.devices.remove(at: idx!)
                            }
                        }
                    } // connectAndSetup()
                } // if (!contains)
            } // if rssi
        } // startScan
    } // btn_cloop()
    
//    // Connect, immediately disconnect, repeat
//    //------------------------------------------
//    func connectLoop()
//    {
//        self.connectDevice!.connectAndSetup().continueWith { t in
//            if let error = t.error {
//                // Sorry we couldn't connect
//                self.pr( error.localizedDescription)
//            }
//            else {
//                self.pr( "\(self.connectCount) connected")
//                self.connectCount += 1
//                let xx = self.connectDevice?.connectAndSetup()
//                self.connectDevice!.cancelConnection()
//                let tt = self.connectDevice?.connectAndSetup()
//                self.connectLoop()
//            }
//        }
//    } // connectLoop()
    
} // class ViewController

