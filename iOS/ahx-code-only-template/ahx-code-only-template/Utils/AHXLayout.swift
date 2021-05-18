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
    // View size
    //================
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

    // View Position
    //===================
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

    // Scale Images
    //================
    // Preserve height, change width to no distort the Image
    //-------------------------------------------------------
    class func scaleWidth( _ v:UIView, likeImage:UIImage)
    {
        let rat = likeImage.size.width / likeImage.size.height
        AHL.width( v, rat * v.frame.height)
    } // scaleWidth()

    // Preserve width, change height to not distort the Image
    //-------------------------------------------------------
    class func scaleHeight( _ v:UIView, likeImage:UIImage)
    {
        let rat = likeImage.size.height / likeImage.size.width
        AHL.height( v, rat * v.frame.width)
    } // scaleHeight()

    // Flex layout views
    //=======================
    
    // Vertically layout subviews inside container.
    // The subviews are made subviews here automatically.
    // points: Array with height for each subview. If a view has a nil points
    //   value, dynamically take an equal share of whatever space is left.
    // topmarg: Margin before the top view, in points
    // botmarg: Margin after the last view, in points
    // space: Space between subviews, in points
    // minheight: Minimum height of a dynamic (nil points) subview, in points
    //----------------------------------------------------------------------------
    class func vShare( container:UIView, subviews:[UIView], points_:[CGFloat?],
                       topmarg_:CGFloat?=nil, botmarg_:CGFloat?=nil, space_:CGFloat?=nil,
                       minheight_:CGFloat?=nil) {
        
        // Make them suvbviews
        for v in subviews {
            if !v.isDescendant(of: container) {
                container.addSubview( v)
            }
        } // for
        
        let ch = container.frame.height
        let topmarg = topmarg_ ?? ch * 0.05
        let botmarg = botmarg_ ?? ch * 0.05
        let space = space_ ?? ch * 0.05
        let minheight = minheight_ ?? ch * 0.05
        let usable_points = (ch - topmarg - botmarg) - space * (CGFloat(subviews.count) - 1.0)
        
        let used_points = AHU.nonNils( points_).reduce( 0,+) // sum non nil ones
        let ndyn = points_.count - AHU.nonNils( points_).count
        let dynh = max( minheight, (usable_points - used_points) / CGFloat(ndyn) )

        // How many vertical points do we need
        var points = [CGFloat]()
        for p in points_ {
            if p != nil { points.append( p!); continue }
            points.append( dynh)
        } // for
        
        let total_points = points.reduce( 0,+)
        // Deal with needing more points than we got
        if total_points > usable_points {
            let shrink = usable_points / total_points
            points = points.map( { $0 * shrink })
        }
        // Position and size the subviews vertically.
        // Leave x and width unchanged.
        var pos = topmarg
        for (i,v) in subviews.enumerated() {
            AHL.height( v, points[i])
            AHL.top( v, pos)
            pos += points[i]
            pos += space
        } // for
    } // vShare()
    
    // Misc
    //=========
    // Give a view a color border
    //------------------------------------------------------------
    class func border( _ v:UIView, _ col:UIColor=UIColor.red)
    {
        v.layer.borderWidth = 1; v.layer.borderColor = col.cgColor
    } // border

} // class AHXLayout

typealias AHL = AHXLayout

