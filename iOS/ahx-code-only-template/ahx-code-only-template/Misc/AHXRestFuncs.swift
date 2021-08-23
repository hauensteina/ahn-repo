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

    // Hit a GET endpoint with parameters, convert result to JSON.
    // If user and password are given, use simple auth.
    //----------------------------------------------------------------------------------
    class func hitURL( _ url:String, parms: [String:Any],
                       method:String="GET", user:String? = nil, passwd:String? = nil,
                       completion: @escaping (_ json:JSON?, _ err:String?) -> ())
    {
        var meth = HTTPMethod.get
        if method == "POST" { meth = HTTPMethod.post }
        var request:DataRequest!
        if meth == .post {
            request = AF.request( url,
                                  method:meth,
                                  parameters: parms,
                                  encoding: JSONEncoding.default )
        } else {
            request = AF.request( url,
                                  method:meth,
                                  parameters: parms)
        }
        if user != nil && passwd != nil {
            request = request.authenticate( username: user!, password: passwd!)
        }
        request.validate(statusCode: 200..<300)
            .responseJSON { response in
                switch response.result {
                case .success(_):
                    guard let resultValue = response.data else { // make sure something came back
                        let msg = "ERROR: hitURL(): Result value in response is nil"
                        AHP.errPopup( message: msg)
                        completion( nil, msg)
                        return
                    }
                    guard let json = try? JSON(data: resultValue) else { // make sure it's json
                        let msg = "ERROR: hitURL(): failed to parse json"
                        AHP.errPopup( message: msg)
                        completion( nil, msg)
                        return
                    }
                    completion( json, nil) // return valid result
                    return
                case .failure( let error): // Oops, request failed
                    let msg = "ERROR: hitURL(): \(error.localizedDescription)"
                    AHP.errPopup( message: msg)
                    completion( nil, msg)
                    return
                } // switch
            } // responseJSON
    } // hitURL()

} // class RestFuncs

typealias AHR = AHXRestFuncs

