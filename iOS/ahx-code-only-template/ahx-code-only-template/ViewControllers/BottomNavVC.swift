//
//  BottomNavVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-19.
//

// A strip at the bottom of the screen, with buttons to press for navigation.

import UIKit

//=========================================
class BottomNavVC: UIViewController {
    static var shared:BottomNavVC!

    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        BottomNavVC.shared = self
        layout()
    } // viewDidLoad
    
    //-----------------
    func layout() {
        
    } // layout()
} // class BottomNavVC
