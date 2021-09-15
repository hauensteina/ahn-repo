//
//  BackEndVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-08-21.
//
/*
 Some buttons to interact with a back end
 */

import UIKit
import SwiftyJSON

//===============================
class BackEndVC: AHXVC {
    var btnSquare: UIButton!
    var tfx: UITextField!
    var lby: UILabel!
    var vwSquare: UIView!
    
    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        self.vwSquare = UIView()
        self.btnSquare = UIButton( type:.system,
                                   primaryAction: UIAction() { _ in
                                    self.square()
                                   })
        self.tfx = UITextField()
        tfx.keyboardType = .decimalPad
        tfx.placeholder = " Enter x"
        self.lby = UILabel()
    } // viewDidLoad()
    
    //-------------------------
    override func layout() {
        
        // Square a number on the back end and display result
        btnSquare.setTitle( "Square", for: .normal)
        AHL.height( btnSquare, AHC.btnHeight)
        AHL.height( tfx, AHC.btnHeight)
        AHL.border( tfx)
        AHL.height( lby, AHC.btnHeight)
        AHL.border( lby)
        AHL.height( vwSquare, 2 * AHC.btnHeight)
        AHL.width( vwSquare, AHC.w - AHC.lmarg - AHC.rmarg)
        AHL.subcenter( vwSquare, self.view)
        AHL.submiddle( vwSquare, self.view)
        AHL.hShare( container: vwSquare, subviews: [tfx, btnSquare, lby])
    } // layout()
    
    //-------------------------------------------------
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
    } // viewWillAppear()
    
    //-------------------
    func square() {
        guard let xstr = tfx.text else { return }
        let x = Double( xstr)
        if x == nil {
            AHP.errPopup( message: "Illegal Number")
            return
        }
        self.tfx.resignFirstResponder()
        AHR.hitURL( "https://ahx-code-only-template.herokuapp.com/sqr",
                    parms: ["x" : x as Any],
                    method:"POST",
                    user:"api",
                    passwd:"7fdd2233f7fb")
        { (json_:JSON?, err:String?) -> () in
            guard let json = json_ else { return }
            let y = json["y"].doubleValue
            self.lby.text = String(y)
        }
    } // square
    
} // class BackEndVC
