//
//  Crashlytics.swift
//  ahx-tiles
//
//  Created by Andreas Hauenstein on 2021-08-17.
//

// Helper functions for the Crashlytics crash logger

import Foundation
import FirebaseCrashlytics

//=========================
class AHXCrashlytics {
    // Leave a breadcrumb for Crashlytics when entering a function.
    // Example: func f() { clef() ... }
    //--------------------------------------------------------------------------------------
    class func clef( file: String = #file, function: String = #function, line: Int = #line)
    {
        let filename = URL(string: file)?.lastPathComponent.components(separatedBy: ".").first
        let msg = "\(filename!).\(function) line \(line) enter"
        Crashlytics.crashlytics().log( msg)
    } // clef()

    // Leave a breadcrumb for Crashlytics, any message
    // Example: clog( "tsup")
    //--------------------------------------------------------------------------------------
    class func clog(_ logMessage: String,
              file: String = #file, function: String = #function, line: Int = #line)
    {
        let filename = URL(string: file)?.lastPathComponent.components(separatedBy: ".").first
        let msg = "\(filename!).\(function) line \(line) \(logMessage)"
        Crashlytics.crashlytics().log( msg)
    } // clog()
} // class AHXCrashlytics

typealias ACL = AHXCrashlytics
