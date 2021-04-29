//
//  MiddleVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-21.
//

// Parent VC for content occupying most of app screen

import UIKit

//==========================================
class MiddleVC: UIViewController {
    var lbInfo:UILabel!
    var nav:UINavigationController!
    var vcOne:FirstVC!
    var vcTwo:SecondVC!
    
    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        let rootVC = UIViewController() // This never gets popped off
        nav = UINavigationController( rootViewController: rootVC)
        AHU.vcContains(parent: self, child: nav)

        // Main area dimensions
        let v = self.view!
        AHL.width( v, AHC.w)
        AHL.height( v, AHC.main_height)
        AHL.left( v, 0)
        AHL.top( v, AHC.main_top)
    } // viewDidLoad()
    
} // class MiddleVC

