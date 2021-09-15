//
//  ContainerVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-21.
//

// A container View Controller with three subviews: topVC, middleVC, bottomVC.
// Think of a Nav bar at the top, a content window, and a Nav bar at the bottom.

import UIKit

//=========================================
class ContainerVC: UIViewController {
    static var shared:ContainerVC!
    var navVC: NavVC!

    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        setup()
    } // viewDidLoad
 
    // Insert all subviews. The middle one is a navigation VC.
    //------------------------------------------------------------
    func setup() {
        ContainerVC.shared = self
        view.backgroundColor = AHC.bgcol
        
        navVC = NavVC()
        _ = navVC.view
        AHU.vcAppend(parent: self, child: navVC)
        AHL.border( navVC.view, .green)
    } // setup()
} // class ContainerVC
