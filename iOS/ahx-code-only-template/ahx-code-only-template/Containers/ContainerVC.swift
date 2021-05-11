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
    var topVC: TopVC!
    var middleVC: MiddleVC!
    var bottomVC: BottomVC!

    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        setup()
    } // viewDidLoad
 
    // Insert all subviews. The middle one is a navigation VC.
    //------------------------------------------------------------
    func setup() {
        view.backgroundColor = AHC.bgcol

        topVC = TopVC()
        _ = topVC.view
        AHU.vcContains(parent: self, child: topVC)
        AHL.border( topVC.view, .red)
        
        middleVC = MiddleVC()
        _ = middleVC.view
        AHU.vcContains(parent: self, child: middleVC)
        AHL.border( middleVC.view, .green)

        bottomVC = BottomVC()
        _ = bottomVC.view
        AHU.vcContains(parent: self, child: bottomVC)
        AHL.border( bottomVC.view, .blue)
        
        ContainerVC.shared = self
    } // setup()
} // class ContainerVC
