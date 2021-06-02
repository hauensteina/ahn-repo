//
//  AHXRestFuncs.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-05-23.
//

/*
 A class collectiong functions to deal with hitting REST APIs
 */

import Alamofire
import SwiftyJSON

//=========================
class AHXRestFuncs {

    // Hit a GET endpoint with parameters, convert result to JSON
    //-----------------------------------------------------------------------------
    class func getURL( _ url:String, parms: [String:Any],
                       completion: @escaping (_ json:JSON?, _ err:String?) -> ())
    {
        AF.request( url,
                    method: .get,
                    parameters: parms)
            .validate(statusCode: 200..<300)
            .responseJSON { response in
                switch response.result {
                case .success(_):
                    guard let resultValue = response.data else { // make sure something came back
                        let msg = "ERROR: getURL(): Result value in response is nil"
                        AHP.errPopup( message: msg)
                        completion( nil, msg)
                        return
                    }
                    guard let json = try? JSON(data: resultValue) else { // make sure it's json
                        let msg = "ERROR: getURL(): failed to parse json"
                        AHP.errPopup( message: msg)
                        completion( nil, msg)
                        return
                    }
                    completion( json, nil) // return valid result
                    return
                case .failure( let error): // Oops, request failed
                    let msg = "ERROR: getURL(): \(error.localizedDescription)"
                    AHP.errPopup( message: msg)
                    completion( nil, msg)
                    return
                } // switch
            } // responseJSON
    } // getURL()

} // class RestFuncs

typealias AHR = AHXRestFuncs
