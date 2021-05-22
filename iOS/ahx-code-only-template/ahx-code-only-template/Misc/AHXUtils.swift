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
    
    // ViewControllers, Views, etc
    //===============================
    // Make a view controller the child of another
    //---------------------------------------------------------------------------
    class func vcAppend( parent:UIViewController, child: UIViewController) {
        parent.addChild( child)
        parent.view.addSubview( child.view)
        child.layout()
        child.didMove( toParent: parent)
    } // vcAppend()
    
    // Get View Controller from a view
    //----------------------------------------------------
    class func myVC(_ v:UIView)  -> UIViewController {
        var v:UIResponder? = v.next
        while( v != nil) {
            if let res = v as? UIViewController {
                return res
            }
            v = v!.next
        } // while
        return UIViewController()
    } // myVC()
    
    // Misc Utils
    //=============
    // Conveniently get a UIColor from RGB
    //---------------------------------------------------------------
    class func RGB( _ red:Int, _ green:Int, _ blue:Int,
                    alpha:CGFloat = 1.0) -> UIColor {
        return UIColor(red: CGFloat(red) / 256.0,
                       green: CGFloat(green) / 256.0,
                       blue: CGFloat(blue) / 256.0,
                       alpha: alpha)
    } // RGB()
    
    // Return nonil elements of arr
    //---------------------------------------------
    class func nonNils<T>(_ arr:[T?]) -> [T] {
        return arr.compactMap( {$0} )
    } // nonNils()
} // AHXUtils

typealias AHU = AHXUtils

