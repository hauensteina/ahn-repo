//
//  FirstVC.swift
//  ahx-tiles
//
//  Created by Andreas Hauenstein on 2021-09-14.
//

/*
 Layout a bunch of clickable tiles to demonstrate the AHXViewPos layout class.
 */

import UIKit
//import Alamofire
//import SwiftyJSON

//=========================================
class TilesVC: AHXVC {
    var dlayout = Array<Dictionary<String,Any>>()
    var vw1 = UIView()
    var vw2 = UIView()
    var vw3 = UIView()
    var vw4 = UIView()
    var vw5 = UIView()
    var vw6 = UIView()
    var vw7 = UIView()
    var vw8 = UIView()

    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        dlayout = [
            [ // The Main Column
                "width":"100pct",
                "rows":
                    [
                        // row 1
                        [ "height":"20pct",
                          "columns":
                            [
                                [ "width":"60pct", "leaf":vw1 ],
                                [ "width":"40pct", "leaf":vw2 ]
                            ]
                        ],
                        // row 2
                        [ "height":"60pct",
                          "columns":
                            [
                                // col 21
                                [ "width":"40pct",
                                  "rows":
                                    [
                                        [ "height":"30pct", "leaf":vw3 ],
                                        [ "height":"30pct", "leaf":vw5 ]
                                    ]
                                ],
                                // col 22
                                [ "width":"60pct",
                                  "rows":
                                    [
                                        [ "height":"20pct", "leaf":vw4 ],
                                        [ "height":"40pct", "leaf":vw6 ]
                                    ]
                                ]
                            ]
                        ],
                        // row 3
                        [ "height":"20pct",
                          "columns": [
                            [ "width":"60pct", "leaf":vw7 ],
                            [ "width":"40pct", "leaf":vw8 ]
                          ]
                        ]
                    ] // rows
            ] // The Main Column
        ] // dlayout
    } // viewDidLoad()

    //-----------------------------------------------
    override func viewDidAppear(_ animated: Bool) {
        layout()
    } // viewDidAppear()
    
    //------------------------
    override func layout() {
        let errStr = AHXViewPos.layout( rootView: self.view, layout: dlayout, border:true)
        if errStr != "ok" {
            AHXPopups.errPopup( "TilesVC.layout(): \(errStr)")
        }
    } // layout()
    
    
} // class TilesVC
