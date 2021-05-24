//
//  AHXGlobFuncs.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-05-23.
//

import Foundation
import Alamofire

// Write a log message
//----------------------------
func AHXLog(_ msg:String) {
    print( msg)
} // AHXLog()

//-------------------
func convert() {
    let api_key = "c37b6a0c288bf10ed187"
    let fromCur = "USD" // curs[fromPicker.getChoice()]
    let toCur = "EUR" // curs[toPicker.getChoice()]
    
    AF.request("https://api.mywebserver.com/v1/board", method: .get, parameters: ["title": "New York Highlights"])
        .validate(statusCode: 200..<300)
        .response { response in
            debugPrint(response)
    }
    
} // convert()
