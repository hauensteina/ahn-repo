//
//  FirstVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-17.
//

import UIKit
import Alamofire
import SwiftyJSON

//=================================
class FirstVC: UIViewController {
    let curs = ["USD","EUR","CHF"]
    // Vert container for all lines
    let vcont = UIView()
    // Hor containers for each line
    let fromCont = UIView()
    let toCont = UIView()
    let btnCont = UIView()
    let resCont = UIView()
    // Widgets
    var fromPicker:AHXPicker!
    var toPicker:AHXPicker!
    let lbRes = UILabel()

    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        self.view.addSubview( vcont)
    } // viewDidLoad()
    
    //------------------------
    override func layout() {
        self.view.backgroundColor = AHC.bgcol
        // Container for all lines
        AHL.width( vcont, AHC.w * 0.9)
        AHL.height( vcont, AHC.h * 0.33 )
        AHL.subcenter( vcont, self.view)
        AHL.submiddle( vcont, self.view)
        AHL.border( vcont, .blue)
        AHL.border( toCont)
        AHL.border( btnCont)
        

        // Add the line containers
        AHL.vShare( container: vcont, subviews: [fromCont, toCont, btnCont, resCont])

        // From currency
        AHL.border( fromCont, .green)
        let lbFrom = UILabel()
        lbFrom.text = "from:"
        AHL.border( lbFrom)
        let tfFromCur = AHXPickerTf()
        tfFromCur.text = curs[0]
        fromPicker = AHXPicker(tf: tfFromCur, choices: curs) { (idx:Int) in }
        AHL.border( tfFromCur)
        AHL.hShare( container: fromCont, subviews: [lbFrom, tfFromCur])
        
        // To currency
        AHL.border( toCont, .green)
        let lbTo = UILabel()
        lbTo.text = "to:"
        AHL.border( lbTo)
        let tfToCur = AHXPickerTf()
        tfToCur.text = curs[1]
        toPicker = AHXPicker(tf: tfToCur, choices: curs, defaultChoice:1) { (idx:Int) in }
        AHL.border( tfToCur)
        AHL.hShare( container: toCont, subviews: [lbTo, tfToCur])
        
        // Convert button
        let btnConv = UIButton(type: .system)
        AHL.border( btnConv, .magenta)
        btnConv.setTitle( "Convert", for: .normal)
        btnConv.addAction { self.convert() }
        AHL.hShare( container: btnCont, subviews: [btnConv])
        
        // Conversion result. convert() fills this.
        AHL.hShare( container: resCont, subviews: [lbRes])
        convert()
    } // layout()
    
    // Swift type to parse and hold conversion API result
    //=======================================================
    class ConversionResult {
        var rate:Double = 0.0
        
        //------------------------------------------
        init( json:JSON, query:String) {
            self.rate = json[query].doubleValue
            if self.rate <= 0 {
                AHP.errPopup(message: "ConversionResult.init(): could not find rate")
            }
        } // init(json)
    } // class ConversionResult
    
    // Hit rest API to convert currencies
    //--------------------------------------
    func convert() {
        let fromCur = curs[fromPicker.getChoice()]
        let toCur = curs[toPicker.getChoice()]
        let query = "\(fromCur)_\(toCur)"
        let parms = [
            "apiKey":"c37b6a0c288bf10ed187",
            "q":query,
            "compact":"ultra"
        ]
        AHR.getURL( "https://free.currconv.com/api/v7/convert", parms: parms)
        { (json_:JSON?, err:String?) -> () in
            guard let json = json_ else { return }
            let res = ConversionResult( json: json, query:query)
            self.lbRes.text = String( format: "%.2f", res.rate)
        }
    } // convert()
    
} // class FirstVC
