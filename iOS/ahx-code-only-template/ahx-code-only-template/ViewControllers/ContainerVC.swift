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
    static let shared = ContainerVC()
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
        
        topVC = TopVC()
        addChild( topVC)
        view.addSubview( topVC.view)
        topVC.layout()
        topVC.didMove( toParent: self)
        AHXLayout.border( topVC.view, .red)
        
        middleVC = MiddleVC()
        addChild( middleVC)
        view.addSubview( middleVC.view)
        //middleVC.layout()
        middleVC.didMove( toParent: self)
        AHXLayout.border( middleVC.view, .green)

        bottomVC = BottomVC()
        addChild( bottomVC)
        view.addSubview( bottomVC.view)
        bottomVC.layout()
        bottomVC.didMove( toParent: self)
        AHXLayout.border( middleVC.view, .blue)

    } // setup()
} // class ContainerVC
