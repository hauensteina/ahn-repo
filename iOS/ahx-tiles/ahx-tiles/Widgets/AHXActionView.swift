//
//  AHXActionView.swift
//  ahx-tiles
//
//  Created by Andreas Hauenstein on 2021-06-09.
//

/*
 A UIView which executes a closure on click.
 */

import UIKit

//================================
class AHXActionView:UIView {
    var action = { ()->() in }
    
    //----------------------------------------------
    func setAction( action: @escaping ()->() ) {
        self.action = action
        let gesture = UITapGestureRecognizer( target: self, action: #selector(clicked(_:)))
        self.addGestureRecognizer( gesture)
    } // setAction()
    
    //-------------------------------------
    @objc func clicked(_ sender: Any) {
        self.action()
    } // clicked()
    
} // class AHXActionView


