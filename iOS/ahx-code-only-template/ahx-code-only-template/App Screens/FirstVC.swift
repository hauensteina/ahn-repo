//
//  FirstVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-17.
//

import UIKit

//=================================
class FirstVC: UIViewController {
    // Vert container for all lines
    var vcont:UIView!
    // Hor containers for each line
    var fromCont:UIView!
    var toCont:UIView!
    var btnCont:UIView!
    var resCont:UIView!
    // Widgets
    var fromPicker:AHXPicker!
    var toPicker:AHXPicker!

    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Container for all lines
        vcont = UIView()
        // Containers for the lines
        fromCont = UIView()
        toCont = UIView()
        btnCont = UIView()
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
        let curs = ["USD","EUR","CHF"]

        // Add the line containers
        AHL.vShare( container: vcont, subviews: [fromCont, toCont, btnCont])

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
    } // layout()

    //-------------------
    func convert() {
        var tt = 42
    } // convert
} // class FirstVC

