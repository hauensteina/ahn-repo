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
        
    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Container for two lines
        vcont = UIView()
        // Container for top line
        hcont1 = UIView()
        // Container for bottom line
        hcont2 = UIView()
        self.view.addSubview( vcont)
    } // viewDidLoad()
    
    //------------------------
    override func layout() {
        // Container for two lines
        AHL.width( vcont, AHC.w * 0.9)
        AHL.height( vcont, AHC.h * 0.33 )
        AHL.subcenter( vcont, self.view)
        AHL.submiddle( vcont, self.view)
        //AHL.subbottom( vcont, self.view)
        AHL.border( vcont, .blue)
        return
        // Container for top line
        AHL.width( hcont1, vcont.frame.width * 0.9)
        AHL.border( hcont1)

        // Container for bottom line
        AHL.width( hcont2, vcont.frame.width * 0.9)
        AHL.border( hcont2)

        // Put two lines in container
        let vh = vcont.frame.height * 0.5
        AHL.vShare( container: vcont, subviews: [hcont1, hcont2],
                    points_: [vh, vh])
        AHL.subcenter( hcont1, vcont)
        AHL.subcenter( hcont2, vcont)
    } // layout()

} // class FirstVC

