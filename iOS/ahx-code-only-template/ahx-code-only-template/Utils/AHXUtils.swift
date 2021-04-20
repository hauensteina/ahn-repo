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
    // Call sth like AHXUtils.addSharedView( self, BurgerVC.shared)
    // from viewDidAppear() of any VC.
    //--------------------------------------------------------------------------------------
    class func addSharedView(_ container:UIViewController!, _ shared:UIViewController! ) {
        var found = false
        for s in container.view.subviews {
            if s === shared.view {
                found = true
                break
            }
        }
        if !found {
            container.view.insertSubview( shared.view, at: 0)
        }
        container.view.bringSubviewToFront( shared.view)
    } // addSharedView()
} // AHXUtils
