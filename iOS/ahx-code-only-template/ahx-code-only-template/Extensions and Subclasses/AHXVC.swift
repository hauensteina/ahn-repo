//
//  AHXVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-06-16.
//

/*
 A View controller subclass that makes navigatio easier
 */

import UIKit

//=========================================
class AHXVC: UIViewController {
    static var navBusy = false
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear( animated)
        AHXVC.navBusy = false
    }
} // class AHXVC
