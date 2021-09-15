//
//  AHXVC.swift
//  ahx-tiles
//
//  Created by Andreas Hauenstein on 2021-06-16.
//

/*
 A View controller subclass that makes navigation easier
 */

import UIKit

// Add an overrideable layout func to all View Controllers
//==========================================================
@objc extension UIViewController {
    func layout() {}
}

//=========================================
class AHXVC: UIViewController {
    static var navBusy = false
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear( animated)
        AHXVC.navBusy = false
    }
} // class AHXVC
