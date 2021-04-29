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

    // Left align a view to an x value
    //------------------------------------------
    class func left(_ v:UIView, _ x:CGFloat)
    {
        let left = x
        v.frame.origin = CGPoint( x: left, y: v.frame.minY)
    } // left
    
    // Left align a view to another view
    //--------------------------------------------
    class func left(_ v:UIView, _ other:UIView)
    {
        AHXLayout.left( v, other.frame.minX)
    } // left

    // Right align a view to an x value
    //-------------------------------------------
    class func right(_ v:UIView, _ x:CGFloat)
    {
        let left = x - (v.frame.width - 1)
        v.frame.origin = CGPoint( x: left, y: v.frame.minY)
    } // right
    
    // Right align a view to another view
    //----------------------------------------------
    class func right(_ v:UIView, _ other:UIView)
    {
        AHXLayout.right( v, other.frame.maxX)
    } // right

    // Center a view to an x value
    //---------------------------------------------
    class func center(_ v:UIView, _ x:CGFloat)
    {
        let left = x - v.frame.width / 2
        v.frame.origin = CGPoint( x: left, y: v.frame.minY)
    } // center
    
    // Center a view to another view
    //-----------------------------------------------
    class func center(_ v:UIView, _ other:UIView)
    {
        AHXLayout.center( v, other.frame.midX)
    } // center
    
    // Align view top to a y value
    //------------------------------------------
    class func top( _ v:UIView, _ y:CGFloat)
    {
        let top = y
        v.frame.origin = CGPoint( x: v.frame.minX, y: top)
    } // top()

    // Align view top to another view
    //----------------------------------------------
    class func top( _ v:UIView, _ other:UIView)
    {
        AHXLayout.top( v, other.frame.minY)
    } // top()

    // Align view middle to a y value
    //--------------------------------------------
    class func middle( _ v:UIView, _ y:CGFloat)
    {
        let top = y - v.frame.height / 2
        v.frame.origin = CGPoint( x: v.frame.minX, y: top)
    } // middle()

    // Align view middle to another view
    //------------------------------------------------
    class func middle( _ v:UIView, _ other:UIView)
    {
        AHXLayout.middle( v, other.frame.midY)
    } // middle()

    // Align view bottom to a y value
    //--------------------------------------------
    class func bottom( _ v:UIView, _ y:CGFloat)
    {
        let top = y - (v.frame.height - 1)
        v.frame.origin = CGPoint( x: v.frame.minX, y: top)
    } // bottom()
        
    // Align view bottom to another view
    //-------------------------------------------------
    class func bottom( _ v:UIView, _ other:UIView)
    {
        AHXLayout.bottom( v, other.frame.maxY)
    } // bottom()

    // Give a view a color border
    //------------------------------------------------------------
    class func border( _ v:UIView, _ col:UIColor=UIColor.red)
    {
        v.layer.borderWidth = 1; v.layer.borderColor = col.cgColor
    } // border

} // class AHXLayout

typealias AHL = AHXLayout

