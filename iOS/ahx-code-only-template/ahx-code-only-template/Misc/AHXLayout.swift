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
    } // width()
    //-------------------------------------------
    class func width(_ v:UIView, _ other:UIView) {
        AHL.width( v, other.frame.width)
    } // width()

    // Set the height of a view
    //---------------------------------------------
    class func height(_ v:UIView, _ y:CGFloat) {
        v.frame.size = CGSize( width:v.frame.width, height:y)
    } // height()
    //-----------------------------------------------
    class func height(_ v:UIView, _ other:UIView) {
        AHL.height( v, other.frame.height)
    } // height()

    // View Position
    //===================
    // Left align a view to an x value
    //------------------------------------------
    class func left(_ v:UIView, _ x:CGFloat) {
        let left = x
        v.frame.origin = CGPoint( x: left, y: v.frame.minY)
    } // left()
    
    // Left align a view to another view
    //--------------------------------------------
    class func left(_ v:UIView, _ other:UIView) {
        AHXLayout.left( v, other.frame.minX)
    } // left()

    // Make view a subview, then left align
    //------------------------------------------------------
    class func subleft(_ v:UIView, _ container:UIView) {
        AHL.addSubview( container, v)
        AHXLayout.left( v, 0)
    } // subleft()

    // Right align a view to an x value
    //-------------------------------------------
    class func right(_ v:UIView, _ x:CGFloat) {
        let left = x - (v.frame.width - 1)
        v.frame.origin = CGPoint( x: left, y: v.frame.minY)
    } // right()
    
    // Right align a view to another view
    //----------------------------------------------
    class func right(_ v:UIView, _ other:UIView) {
        AHXLayout.right( v, other.frame.maxX)
    } // right()

    // Make view a subview, then right align
    //------------------------------------------------------
    class func subright(_ v:UIView, _ container:UIView) {
        AHL.addSubview( container, v)
        AHXLayout.right( v, container.frame.width)
    } // subright()

    // Center a view to an x value
    //---------------------------------------------
    class func center(_ v:UIView, _ x:CGFloat) {
        let left = x - v.frame.width / 2
        v.frame.origin = CGPoint( x: left, y: v.frame.minY)
    } // center()
    
    // Center a view to another view
    //-----------------------------------------------
    class func center(_ v:UIView, _ other:UIView) {
        AHXLayout.center( v, other.frame.midX)
    } // center()

    // Make view a subview, then center
    //------------------------------------------------------
    class func subcenter(_ v:UIView, _ container:UIView) {
        AHL.addSubview( container, v)
        let left = (container.frame.width - v.frame.width) / 2.0
        AHXLayout.left( v, left)
    } // subcenter()

    // Align view top to a y value
    //------------------------------------------
    class func top( _ v:UIView, _ y:CGFloat) {
        let top = y
        v.frame.origin = CGPoint( x: v.frame.minX, y: top)
    } // top()

    // Align view top to another view
    //----------------------------------------------
    class func top( _ v:UIView, _ other:UIView) {
        AHXLayout.top( v, other.frame.minY)
    } // top()
    
    // Make view a subview, then top align
    //------------------------------------------------------
    class func subtop(_ v:UIView, _ container:UIView) {
        AHL.addSubview( container, v)
        AHXLayout.top( v, 0)
    } // subtop()

    // Align view middle to a y value
    //--------------------------------------------
    class func middle( _ v:UIView, _ y:CGFloat) {
        let top = y - v.frame.height / 2
        v.frame.origin = CGPoint( x: v.frame.minX, y: top)
    } // middle()

    // Align view middle to another view
    //------------------------------------------------
    class func middle( _ v:UIView, _ other:UIView) {
        AHXLayout.middle( v, other.frame.midY)
    } // middle()

    // Make view a subview, then vert center
    //------------------------------------------------------
    class func submiddle(_ v:UIView, _ container:UIView) {
        AHL.addSubview( container, v)
        AHXLayout.middle( v, container.frame.height / 2.0)
    } // submiddle()

    // Align view bottom to a y value
    //--------------------------------------------
    class func bottom( _ v:UIView, _ y:CGFloat) {
        let top = y - (v.frame.height - 1)
        v.frame.origin = CGPoint( x: v.frame.minX, y: top)
    } // bottom()
        
    // Align view bottom to another view
    //-------------------------------------------------
    class func bottom( _ v:UIView, _ other:UIView) {
        AHXLayout.bottom( v, other.frame.maxY)
    } // bottom()
    
    // Make view a subview, then align bottom
    //------------------------------------------------------
    class func subbottom(_ v:UIView, _ container:UIView) {
        AHL.addSubview( container, v)
        AHXLayout.bottom( v, container.frame.height)
    } // subbottom()

    // Scale Images
    //================
    // Preserve height, change width to not distort the Image
    //---------------------------------------------------------
    class func scaleWidth( _ v:UIView, likeImage:UIImage) {
        let rat = likeImage.size.width / likeImage.size.height
        AHL.width( v, rat * v.frame.height)
    } // scaleWidth()

    // Preserve width, change height to not distort the Image
    //---------------------------------------------------------
    class func scaleHeight( _ v:UIView, likeImage:UIImage) {
        let rat = likeImage.size.height / likeImage.size.width
        AHL.height( v, rat * v.frame.width)
    } // scaleHeight()

    // Flex layout views
    //=======================

    // Add subviews to container, if not already there
    //----------------------------------------------------------------
    class func addSubviews( _ container:UIView, _ subviews:[UIView]) {
        for v in subviews {
            AHL.addSubview( container, v)
        } // for
    } // addSubViews()
    
    // Add subview to container, if not already there
    //----------------------------------------------------------------
    class func addSubview( _ container:UIView, _ subview:UIView) {
            if !subview.isDescendant(of: container) {
                container.addSubview( subview)
            }
    } // addSubView()

    // Vertically layout subviews inside container.
    // The subviews are made subviews here automatically.
    // points: Array with height for each subview. If a view has a nil height
    //   value, dynamically take an equal share of whatever space is left.
    // topmarg: Margin before the top view, in points
    // botmarg: Margin after the last view, in points
    // space: Space between subviews, in points
    // minheight: Minimum height of a dynamic (nil points) subview, in points
    //---------------------------------------------------------------------------------------
    class func vShare( container:UIView, subviews:[UIView], heights_:[CGFloat?]?=nil,
                       topmarg_:CGFloat?=nil, botmarg_:CGFloat?=nil, space_:CGFloat?=nil,
                       minheight_:CGFloat?=nil) {

        AHL.addSubviews( container, subviews)

        var heights = [CGFloat?].init(repeating: nil, count: subviews.count)
        if heights_ != nil {
            heights = heights_!
        }
        let ch = container.frame.height
        let topmarg = topmarg_ ?? ch * 0.05
        let botmarg = botmarg_ ?? ch * 0.05
        let space = space_ ?? ch * 0.05
        let minheight = minheight_ ?? ch * 0.05
        let usable_points = (ch - topmarg - botmarg) - space * (CGFloat(subviews.count) - 1.0)
        
        let used_points = AHU.nonNils( heights).reduce( 0,+) // sum non nil ones
        let ndyn = heights.count - AHU.nonNils( heights).count
        let dynh = max( minheight, (usable_points - used_points) / CGFloat(ndyn) )

        // Fill in the dynamic heights
        var points = [CGFloat]()
        for p in heights {
            if p != nil { points.append( p!); continue }
            points.append( dynh)
        } // for

        // Deal with needing more points than we got
        let total_points = points.reduce( 0,+)
        if total_points > usable_points {
            let shrink = usable_points / total_points
            points = points.map( { $0 * shrink })
        }
        // Position and size the subviews vertically.
        // Leave x and width unchanged.
        var pos = topmarg
        for (i,v) in subviews.enumerated() {
            AHL.height( v, points[i])
            AHL.width( v, container.frame.width)
            AHL.top( v, pos)
            pos += points[i]
            pos += space
        } // for
    } // vShare()

    // Horizontally layout subviews inside container.
    // The subviews are made subviews here automatically.
    // points: Array with width for each subview. If a view has a nil points
    //   value, dynamically take an equal share of whatever space is left.
    // leftmarg: Margin before the leftmost view, in points
    // rightmarg: Margin to the very right, in points
    // space: Space between subviews, in points
    // minwidth: Minimum width of a dynamic (nil points) subview, in points
    //----------------------------------------------------------------------------------------
    class func hShare( container:UIView, subviews:[UIView], widths:[CGFloat?]?=nil,
                       leftmarg:CGFloat?=nil, rightmarg:CGFloat?=nil, space:CGFloat?=nil,
                       minwidth:CGFloat?=nil) {

        AHL.addSubviews( container, subviews)

        var wwidths = [CGFloat?].init(repeating: nil, count: subviews.count)
        if widths != nil {
            wwidths = widths!
        }

        let cw = container.frame.width
        let lleftmarg = leftmarg ?? cw * 0.05
        let rrightmarg = rightmarg ?? cw * 0.05
        let sspace = space ?? cw * 0.05
        let mminwidth = minwidth ?? cw * 0.05
        let usable_points = (cw - lleftmarg - rrightmarg) - sspace * (CGFloat(subviews.count) - 1.0)
        
        let used_points = AHU.nonNils( wwidths).reduce( 0,+) // sum non nil ones
        let ndyn = wwidths.count - AHU.nonNils( wwidths).count
        let dynw = max( mminwidth, (usable_points - used_points) / CGFloat(ndyn) )

        // Fill in the dynamic widths
        var points = [CGFloat]()
        for p in wwidths {
            if p != nil { points.append( p!); continue }
            points.append( dynw)
        } // for
        
        // Deal with needing more points than we got
        let total_points = points.reduce( 0,+)
        if total_points > usable_points {
            let shrink = usable_points / total_points
            points = points.map( { $0 * shrink })
        }
        // Position and size the subviews horizontally.
        var pos = lleftmarg
        for (i,v) in subviews.enumerated() {
            AHL.width( v, points[i])
            AHL.submiddle( v, container)
            AHL.left( v, pos)
            pos += points[i]
            pos += sspace
        } // for
    } // hShare()
    
    // Fonts
    //===========
    
    //----------------------------------------------------
    class func smallFont() -> UIFont {
        struct statics { static var res:UIFont? = nil }
        if statics.res == nil { statics.res = getSystemFont( linesPerScreen: 60) }
        return statics.res!
    } // smallFont()

    //----------------------------------------------------
    class func mediumFont() -> UIFont {
        struct statics { static var res:UIFont? = nil }
        if statics.res == nil { statics.res = getSystemFont( linesPerScreen: 40) }
        return statics.res!
    } // mediumFont()

    //----------------------------------------------------
    class func largeFont() -> UIFont {
        struct statics { static var res:UIFont? = nil }
        if statics.res == nil { statics.res = getSystemFont( linesPerScreen: 20) }
        return statics.res!
    } // largeFont()

    // Get a system font of a size such that n lines will fit the screen
    //----------------------------------------------------------------------
    class func getSystemFont( linesPerScreen:Int) -> UIFont {
        let w = UIScreen.main.bounds.width
        let h = UIScreen.main.bounds.height
        let text = String(repeating: "line\n", count: linesPerScreen)
        let font = UIFont.systemFont( ofSize: 100)
        let bestFont = AHL.bestFittingFont( text: text, width:w, height:h, font: font)
        return bestFont
    } // getSystemFont()
    
    // For a given font, fond the largest size that fits in width,height
    //--------------------------------------------------------------------------------------------------------
    class func bestFittingFont( text: String, width:CGFloat, height: CGFloat, font: UIFont) -> UIFont {
        let fontDescriptor = font.fontDescriptor
        let constrainingDimension = min( width, height)
        let properBounds = CGRect(x: 0, y: 0, width: width, height: height)
        var attributes = [NSAttributedString.Key:Any]()
         
        let infiniteBounds = CGSize(width: CGFloat.infinity, height: CGFloat.infinity)
        var bestFontSize: CGFloat = constrainingDimension
         
        for fontSize in stride( from: bestFontSize, through: 0, by: -1) {
            let newFont = UIFont(descriptor: fontDescriptor, size: fontSize)
            attributes[.font] = newFont
             
            let currentFrame = text.boundingRect( with: infiniteBounds, options: [.usesLineFragmentOrigin, .usesFontLeading], attributes: attributes, context: nil)
             
            if properBounds.contains(currentFrame) {
                bestFontSize = fontSize
                break
            }
        } // for fontSize
        let bestFont = UIFont( descriptor: fontDescriptor, size: bestFontSize)
        return bestFont
     } // bestFittingFont()
    
    // Misc
    //=========
    // Give a view a color border
    //------------------------------------------------------------
    class func border( _ v:UIView, _ col:UIColor=UIColor.red) {
        v.layer.borderWidth = 1; v.layer.borderColor = col.cgColor
    } // border

} // class AHXLayout

typealias AHL = AHXLayout

