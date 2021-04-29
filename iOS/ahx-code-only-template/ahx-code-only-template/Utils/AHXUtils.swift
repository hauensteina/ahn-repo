//
//  AHXUtils.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-19.
//

// Misc helper functions

import UIKit

// Add an overrideable layout func to all View Controllers
//==========================================================
@objc extension UIViewController {
    func layout() {}
}

//===================
class AHXUtils {
    // Make a view controller the child of another
    //---------------------------------------------------------------------------
    class func vcContains( parent:UIViewController, child: UIViewController) {
        parent.addChild( child)
        parent.view.addSubview( child.view)
        child.layout()
        child.didMove( toParent: parent)
    } // vcContains()
    
    //-------------------------------------------------------------------------------------------------
    class func RGB( _ red:CGFloat, _ green:CGFloat, _ blue:CGFloat, alpha:CGFloat = 1.0) -> UIColor {
        return UIColor(red: red, green: green, blue: blue, alpha: alpha)
    } // RGB()
} // AHXUtils

typealias AHU = AHXUtils

