//
//  AHXViewPos.swift
//  ahx-tiles
//
//  Created by Andreas Hauenstein on 2021-09-19.
//

/*
 Functions to lay out subviews in a hierarchical row/column manner, similar to JS flexgrids.
 */

import UIKit

//=====================
class AHXViewPos {
    static let TPAD:CGFloat = 5 // pct
    static let BPAD:CGFloat = 5
    static let LPAD:CGFloat = 5
    static let RPAD:CGFloat = 5

    //----------------------------------------------------------------------------------------------------
    class func layout( rootView:UIView, layout:Array<Dictionary<String,Any>>, border:Bool) -> String {
        AHU.removeSubviews( rootView)
        if layout[0]["width"] != nil {
            return AHXViewPos.layoutCols( parent:rootView, columns:layout)
        }
        else if layout[0]["height"] != nil {
            return AHXViewPos.layoutRows( parent:rootView, rows:layout)
        }
        else {
            AHXPopups.errPopup( "AHXViewPos.layout(): found neither width nor height")
            return "ERROR: found neither width nor height"
        }
    } // layout
    
    //---------------------------------------------------------------------------------------------------
    private class func layoutCols( parent: UIView, columns: Array<Dictionary<String,Any>>) -> String {
        let w = UIScreen.main.bounds.width
        var left:CGFloat = 0.0
        let top:CGFloat = 0.0
        var lpad = AVP.LPAD
        var rpad = AVP.RPAD
        let height = parent.frame.height
        
        for c in columns {
            let (ltype, lpv, lerr) = parseSize( c["lpad"])
            if lerr == nil { lpad = lpv }
            let (_, rpv, rerr) = parseSize( c["rpad"])
            if rerr == nil { rpad = rpv }
            let (wtype, wv, werr) = parseSize( c["width"])
            if werr != nil { return werr! }
            if wtype == "pct" {
                let width = wv / 100.0 * w
                let outer = UIView()
                parent.addSubview( outer)
                outer.frame = CGRect( x: left, y: top, width: width, height: height)
                let inner = UIView()
                outer.addSubview( inner)
                inner.frame = CGRect( x: lpad, y: 0, width: width - lpad - rpad, height: height)
                //@@@ cont here
            }
            else { return "ERROR: width type \(wtype) not implemented" }
        } // for
        return "ok"
    } // layoutCols

    //-------------------------------------------------------------------------------------------------
    private class func layoutRows( parent: UIView, rows: Array<Dictionary<String,Any>>) -> String {
        return "ok"
    } // layoutRows
    
    //-----------------------------------------------------------------------------------------
    private class func parseSize(_ sizeStr:Any?) -> (type:String, val:CGFloat, err:String?) {
        let size = String( describing: sizeStr)
        if size.hasSuffix( "pct") {
            guard let val = Double( size.components(separatedBy: "p")[0])
            else { return (type:"err", val:0.0, err:"ERROR: size not numeric") }
            return (type:"pct", val:CGFloat(val), err:nil)
        }
        else if size.hasSuffix( "flex") {
            guard let val = Double( size.components(separatedBy: "f")[0])
            else { return (type:"err", val:0.0, err:"ERROR: size not numeric") }
            return (type:"flex", val:CGFloat(val), err:nil)
        }
        else {
            return (type:"err", val:0.0, err:"ERROR: unknown size type")
        }
    } // parseSize()

} // class AHXViewPos

typealias AVP = AHXViewPos
