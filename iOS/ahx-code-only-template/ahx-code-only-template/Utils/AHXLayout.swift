//
//  AHXLayout.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-17.
//

// Helper functions to lay out UI elements.

import UIKit

// General purpose layout methods to replace constraints
//=========================================================
class AHXLayout
{
    // Set the width of a view
    //-------------------------------------------
    class func width(_ v:UIView, _ x:CGFloat) {
        v.frame.size = CGSize( width:x, height:v.frame.height)
    } // hsz()

    // Set the height of a view
    //---------------------------------------------
    class func height(_ v:UIView, _ y:CGFloat) {
        v.frame.size = CGSize( width:v.frame.width, height:y)
    } // height()

    // Left align a view
    //-------------------------------------------------------------------------
    class func left(_ v:UIView, _ x:CGFloat)
    {
        let left = x
        v.frame.origin = CGPoint( x: left, y: v.frame.minY)
    } // left

    // Right align a view
    //-------------------------------------------------------------------------
    class func right(_ v:UIView, _ x:CGFloat)
    {
        let left = x - (v.frame.width - 1)
        v.frame.origin = CGPoint( x: left, y: v.frame.minY)
    } // right

    // Center a view
    //-------------------------------------------------------------------------
    class func center(_ v:UIView, _ x:CGFloat)
    {
        let left = x - v.frame.width / 2
        v.frame.origin = CGPoint( x: left, y: v.frame.minY)
    } // center
    
    // Align view top
    //---------------------------------------------------
    class func top( _ v:UIView, _ y:CGFloat)
    {
        let top = y
        v.frame.origin = CGPoint( x: v.frame.minX, y: top)
    } // top()

    // Align view middle
    //---------------------------------------------------
    class func middle( _ v:UIView, _ y:CGFloat)
    {
        let top = y - v.frame.height / 2
        v.frame.origin = CGPoint( x: v.frame.minX, y: top)
    } // middle()

    // Align view bottom
    //---------------------------------------------------
    class func bottom( _ v:UIView, _ y:CGFloat)
    {
        let top = y - (v.frame.height - 1)
        v.frame.origin = CGPoint( x: v.frame.minX, y: top)
    } // bottom()
        
    // Give a view a color border
    //------------------------------------------------------------
    class func border( _ v:UIView, _ col:UIColor=UIColor.red)
    {
        v.layer.borderWidth = 1; v.layer.borderColor = col.cgColor
    } // border

} // class AHXLayout


