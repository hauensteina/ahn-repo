//
//  AHXOnAnyTouch.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-06-06.
//

/*
  Register a closure that will fire on any screen touch
 */

import UIKit

//=======================
class AHXOnAnyTouch {
    var action:( _ x:CGFloat, _ y:CGFloat) -> ()
    var gesture:UITapGestureRecognizer!
    
    //--------------------------------------------------
    init( action: @escaping ( _ x:CGFloat, _ y:CGFloat) -> () ) {
        self.action = action
        self.gesture = UITapGestureRecognizer( target: self, action:  #selector (self.tappedAnywhere (_:)))
        SceneDelegate.shared.window!.addGestureRecognizer( gesture)
    } // init()
    
    //--------------------------------------------------------------
    @objc func tappedAnywhere(_ sender:UITapGestureRecognizer) {
        let loc = sender.location( in: SceneDelegate.shared.win.rootViewController?.view)
        self.action( loc.x, loc.y)
    } // tappedAnywhere()

} // class AHXOnAnyTouch
