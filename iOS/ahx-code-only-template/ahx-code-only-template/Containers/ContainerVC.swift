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
        ContainerVC.shared = self
        view.backgroundColor = AHC.bgcol

        
        middleVC = MiddleVC()
        _ = middleVC.view
        AHU.vcAppend(parent: self, child: middleVC)
        AHL.border( middleVC.view, .green)

        bottomVC = BottomVC()
        _ = bottomVC.view
        AHU.vcAppend(parent: self, child: bottomVC)
        AHL.border( bottomVC.view, .blue)
        
        // Must be last so it is on top, for the burger menu to show
        topVC = TopVC()
        _ = topVC.view
        AHU.vcAppend(parent: self, child: topVC)
        AHL.border( topVC.view, .red)
    } // setup()
} // class ContainerVC
