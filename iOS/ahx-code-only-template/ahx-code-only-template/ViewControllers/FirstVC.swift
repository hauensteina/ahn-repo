//
//  FirstVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-17.
//

import UIKit

//=================================
class FirstVC: UIViewController {
    var vcont:UIView!
    var hcont1:UIView!
    var hcont2:UIView!
    var hcont3:UIView!

    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Container for all lines
        vcont = UIView()
        // Containers for the lines
        hcont1 = UIView()
        hcont2 = UIView()
        hcont3 = UIView()
        self.view.addSubview( vcont)
    } // viewDidLoad()
    
    //------------------------
    override func layout() {
        // Container for all lines
        AHL.width( vcont, AHC.w * 0.9)
        AHL.height( vcont, AHC.h * 0.33 )
        AHL.subcenter( vcont, self.view)
        AHL.submiddle( vcont, self.view)
        //AHL.subbottom( vcont, self.view)
        AHL.border( vcont, .blue)
        
        // Add the line containers
        AHL.vShare( container: vcont, subviews: [hcont1, hcont2, hcont3])

        // Top line (from currency)
        AHL.border( hcont1, .green)
        let lbFrom = UILabel()
        lbFrom.text = "from:"
        AHL.border( lbFrom)
        let ddFromCur = UILabel()
        ddFromCur.text = "USD"
        AHL.border( ddFromCur)
        AHL.hShare( container: hcont1, subviews: [lbFrom, ddFromCur])
        
        // Container for second line (to currency)
        AHL.border( hcont2)

        // Container for third line (amount)
        AHL.border( hcont3)

    } // layout()

} // class FirstVC

