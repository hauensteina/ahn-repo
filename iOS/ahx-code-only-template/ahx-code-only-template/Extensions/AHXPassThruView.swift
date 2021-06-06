//
//  AHXPassThruView.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-06-02.
//

/*
 A view which passes events to its children, even if they are outside the frame
 */

import UIKit

//=============================
class AHXPassThruView: UIView {
        
    //------------------------------------------------------------------------------
    override func point( inside point: CGPoint, with event: UIEvent?) -> Bool {
        for subview in subviews as [UIView] {
            if !subview.isHidden && subview.alpha > 0 && subview.isUserInteractionEnabled &&
                subview.point( inside: convert( point, to: subview), with: event) {
                return true
            }
        }
        return false
    } // point()

} // class AHXPassThruView

