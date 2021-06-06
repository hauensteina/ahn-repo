//
//  AHXMenu.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-05-27.
//

/*
 A view with menu items that can slide in and out.
 Make this a subview of a viewcontroller spanning the whole screen width.
 Example usage:
 myVC.view = PassThruView() // in case it's not high enough
 let menu = AHXSlideMenu( ["One","Two"],
                          [
                             { () -> () in AHP.popup(title: "Menu", message: "One") },
                             { () -> () in AHP.popup(title: "Menu", message: "Two") }
                          ]
 )
 myVC.view.addSubview( menu)
 */

import UIKit

//=============================
class AHXSlideMenu:UIView
{
    var side = "right"
    var showing = false
    var showX:CGFloat = 0.0
    var hideX:CGFloat = 0.0
    var touchAction:AHXOnAnyTouch!
    
    //----------------------------------------------------------------------
    init( items:[String], actions:[()->()],
          width:CGFloat = AHC.w/3,
          itemHeight:CGFloat = UIScreen.main.bounds.height * 0.05,
          bgcol:UIColor = AHU.RGB( 120,120,120, alpha: 0.8),
          rightorleft:String = "right")
    {
        self.showX = UIScreen.main.bounds.width - width
        if side == "left" { showX = 0 }
        self.hideX = UIScreen.main.bounds.width
        if side == "left" { hideX = -width }
        let frame = CGRect( x: hideX, y: 0, width: width, height: UIScreen.main.bounds.height)
        super.init( frame:frame)

        self.backgroundColor = .red

        // Build the menu items
        var heights = [CGFloat?]()
        var itemViews = [UIView]()
        let sepHeight = itemHeight / 2.0
        let sepColor = AHU.RGB( 20,20,20, alpha: 1.0)
        for (idx,txt) in items.enumerated() {
            if txt == "" { // separator
                let v = UIView()
                v.backgroundColor = sepColor
                itemViews.append( v)
                heights.append( sepHeight)
            } else {
                let v = UIButton()
                v.AHXAddAction( { () -> () in self.hide();  actions[idx]() } )
                v.setTitle( txt, for:.normal)
                itemViews.append( v)
                heights.append( itemHeight)
            }
        } // for
        // Flexible expanding space at the end
        let spacer = UIView()
        itemViews.append( spacer)
        heights.append( nil)
        // Layout the menu items
        AHL.vShare( container: self, subviews: itemViews, heights_: heights)
        
        // Hide on any tap
        self.touchAction = AHXOnAnyTouch() { (x:CGFloat, y:CGFloat) in
            if self.showing { self.hide() }
        }
    } // init()

    //----------------
    func show() {
        showing = true
        var targetx = AHU.myVC( self).view.frame.width - self.frame.width
        if side == "left" { targetx = 0 }
        UIView.animate( withDuration: 0.5,
                        delay: 0,
                        usingSpringWithDamping: 0.8,
                        initialSpringVelocity: 0,
                        options: .curveEaseInOut,
                        animations: {
                            self.frame.origin.x = targetx },
                        completion: nil )
    } // show()
    
    //---------------
    func hide() {
        showing = false
        var targetx = AHU.myVC( self).view.frame.width
        if side == "left" { targetx = -self.frame.width }
        UIView.animate( withDuration: 0.5,
                        delay: 0,
                        usingSpringWithDamping: 0.8,
                        initialSpringVelocity: 0,
                        options: .curveEaseInOut,
                        animations: {
                            self.frame.origin.x = targetx },
                        completion: nil )
    } // hide()
    
    //---------------------------------
    required init?( coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

} // class AHXSlideMenu
