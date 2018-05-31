//
//  ViewController.swift
//  mblogging
//
//  Created by Andreas Hauenstein on 2018-05-31.
//  Copyright © 2018 AHN. All rights reserved.
//

// Simple project to demonstrate log->disconnect->pull cycle on one sensor

import UIKit
import MetaWear
import MetaWearCpp

class ViewController: UIViewController {

    @IBOutlet weak var txtLog: UITextView!
    
    var ccount:Int64 = 0
    var device:MetaWear!
    var mac:String = ""
    var logger:OpaquePointer!
    var accSignal:OpaquePointer!
    var passThrough:OpaquePointer!
    var logger_id:UInt8!
    let LOG_N_SAMPLES:UInt16 = 500
    var handlers = MblMwLogDownloadHandler()
    enum Sensorstate { case none, logging, pulling }
    var sensorState:Sensorstate = .none

    //-----------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        txtLog.layer.borderWidth = 1
        
        // Always make sure you have handlers in place
        handlers.context = bridge(obj: self)
        handlers.received_progress_update = { (context, entriesLeft, totalEntries) in
            let this:ViewController = bridge( ptr: context!)
            if this.sensorState == .pulling {
                this.pr( String( format: "left: %d / %d", entriesLeft, totalEntries))
                if entriesLeft == 0 {
                    this.device.clearAndReset()
                    this.pr("disconnected")
                }
            }
        }
        handlers.received_unknown_entry = { (context, id, epoch, data, length) in }
        handlers.received_unhandled_entry = { (context, data) in }
    } // viewDidLoad()

    //-----------------------------------------
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    // Show a string on our textview
    //-------------------------------
    func pr(_ s:String) -> Void {
        DispatchQueue.main.async {
            self.txtLog.text.append( "\n\(s)")
            let stringLength:Int = self.txtLog.text.count
            self.txtLog.scrollRangeToVisible( NSMakeRange(stringLength-1, 0))
        }
    } // pr()
    
    // Set a key,value pair in the user defaults
    //--------------------------------------------------------
    func setProp( key: String!, value: String!)->Void {
        DispatchQueue.main.async {
            UserDefaults.standard.set(value, forKey: key)
            UserDefaults.standard.synchronize()
        }
    }
    
    // Get a value for a key from the user defaults
    //--------------------------------------------------------
    func getProp(_ key: String)->String? {
        let res = UserDefaults.standard.object( forKey: key) as? String
        return res
    }

    // Start logging and disconnect
   //----------------------------------------
    @IBAction func btnStart(_ sender: Any) {
        self.pr( "scanning ...")
        MetaWearScanner.shared.startScan(allowDuplicates: false) { (device) in
            if device.rssi > -70 {
                MetaWearScanner.shared.stopScan()
                // Connect to the board we found
                device.connectAndSetup().continueWith { t in
                    if let error = t.error {
                        self.pr( error.localizedDescription)
                        return
                    }
                    self.sensorState = .logging
                    self.device = device
                    // Fake download just to set the handlers.
                    // mbl_mw_logging_download( device.board!, 0, &self.handlers)
                    
                    mbl_mw_logging_clear_entries( self.device.board)
                    mbl_mw_metawearboard_tear_down( self.device.board)

                    self.mac = device.mac!
                    self.pr( "connected to \(self.mac)")

                    // Configure sensor fusion and get a signal
                    mbl_mw_sensor_fusion_set_mode( device.board, MBL_MW_SENSOR_FUSION_MODE_IMU_PLUS)
                    mbl_mw_sensor_fusion_set_acc_range( device.board, MBL_MW_SENSOR_FUSION_ACC_RANGE_2G)
                    mbl_mw_sensor_fusion_set_gyro_range( device.board, MBL_MW_SENSOR_FUSION_GYRO_RANGE_2000DPS) // necessary
                    mbl_mw_sensor_fusion_write_config( device.board)
                    
                    self.accSignal = mbl_mw_sensor_fusion_get_data_signal( device.board,
                                                                           MBL_MW_SENSOR_FUSION_DATA_LINEAR_ACC)
                    
                    // Make a passthrough proc for N samples
                    mbl_mw_dataprocessor_passthrough_create( self.accSignal, MBL_MW_PASSTHROUGH_MODE_COUNT, self.LOG_N_SAMPLES,
                                                             bridge(obj: self))
                    { (context, proc) in
                        let this:ViewController = bridge( ptr: context!)
                        this.passThrough = proc
                        mbl_mw_logging_clear_entries( this.device.board)
                        mbl_mw_datasignal_log( this.passThrough, context) { (context, logger) in
                            let this:ViewController = bridge( ptr: context!)
                            if let logger = logger {
                                this.logger = logger
                                // Remember logger_id in self
                                this.logger_id = mbl_mw_logger_get_id( logger)
                                
                                mbl_mw_logging_start( this.device.board, 1)
                                this.pr( "Logging kicked off for \(this.LOG_N_SAMPLES) samples.")
                                // You have to disconnect asynchronously with a little lag, otherwise nothing logs
                                DispatchQueue.main.asyncAfter( deadline: DispatchTime.now() + .seconds(1)) {
                                    this.device.cancelConnection()
                                    this.pr( "disconnected")
                                }
                            }
                            else {
                                this.pr( "failed to get logger")
                                this.device.clearAndReset()
                            }
                        } // mbl_mw_datasignal_log()
                        // Start
                        mbl_mw_sensor_fusion_enable_data( this.device.board, MBL_MW_SENSOR_FUSION_DATA_LINEAR_ACC)
                        mbl_mw_sensor_fusion_start( this.device.board)
                    } // passthrough_create()
                } // connectAndSetup()
            } // if(rssi)
        } // startScan
    } // btnStart()

    // Reconnect and pull data
    //------------------------------------------
    @IBAction func btnPull(_ sender: Any) {
        self.pr( "scanning ...")

        MetaWearScanner.shared.startScan(allowDuplicates: false) { (device) in
            if device.rssi > -70 {
                // Connect to the board we found
                device.connectAndSetup().continueWith { t in
                    if let error = t.error {
                        self.pr( error.localizedDescription)
                        return
                    }
                    self.sensorState = .pulling
                    self.device = device
                    //mbl_mw_logging_stop( self.device.board) // omitting this will crash
                    let mac = device.mac!
                    if mac != self.mac {
                        self.pr( "Unknown mac \(mac). Need \(self.mac).")
                        device.cancelConnection()
                        return
                    }
                    MetaWearScanner.shared.stopScan()
                    self.pr( "reconnected to \(self.mac) for log download")

                    // Get the logger back from the sensor
                    let recoveredLogger = mbl_mw_logger_lookup_id( device.board, self.logger_id!)
                    self.pr( "recovered the logger")
                    self.ccount = 0
                    
                    // The context is just a pointer to self.
                    mbl_mw_logger_subscribe( recoveredLogger, bridge(obj: self)) { (context, data) in
                        let this:ViewController = bridge( ptr: context!)
                        let xyz = data!.pointee.valueAs() as MblMwCartesianFloat
                        this.pr( String(format: "%d %.2f %.2f %.2f", this.ccount, xyz.x, xyz.y, xyz.z))
                        this.ccount += 1
                    }
                    // Start the log download
                    let n_notifies:UInt8 = 100
                    mbl_mw_logging_download( self.device.board!, n_notifies, &self.handlers)
                } // connectAndSetup()
            } // if(rssi)
        } // startScan
    } // btnPull()
} // class ViewController

