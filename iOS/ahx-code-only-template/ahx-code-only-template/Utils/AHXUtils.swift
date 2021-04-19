//
//  AHXUtils.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-19.
//

// Misc helper functions

import UIKit

//===================
class AHXUtils {
    
    // Add a shared view (like a burger menu) to a ViewController.
    // Call sth like AHXUtils.addSharedView( self, BurgerVC) from viewDidAppear()
    // of any VC. BurgerVC must have BurgerVC.shared instantiated.
    //--------------------------------------------------------------------------------------
    class func addSharedView(_ container:UIViewController!, _ shared:UIViewController! ) {
        var found = false
        for s in container.view.subviews {
            if s === shared.view) {
                found = true
                break
                
            }
        }
        if !found {
            conainer.view.insertSubview( shared.view, at: 0)
        }
        container.view.bringSubview(toFront: shared.view)
    } // addSharedView()
} // AHXUtils
