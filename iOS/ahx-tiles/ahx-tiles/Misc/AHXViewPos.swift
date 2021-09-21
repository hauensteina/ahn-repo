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
    static let TPAD = "5pct"
    static let BPAD = "5pct"
    static let LPAD = "5pct"
    static let RPAD = "5pct"

    //----------------------------------------------------------------------------------------------------
    class func layout( rootView:UIView, layout:Array<Dictionary<String,Any>>, border:Bool) -> String {
        AHU.removeSubviews( rootView)
        if layout[0]["width"] != nil {
            return AHXViewPos.layoutCols( parent:rootView, columns:layout, border:border)
        }
        else if layout[0]["height"] != nil {
            return AHXViewPos.layoutRows( parent:rootView, rows:layout, border:border)
        }
        else {
            AHXPopups.errPopup( "AHXViewPos.layout(): found neither width nor height")
            return "ERROR: found neither width nor height"
        }
    } // layout()
    
    //---------------------------------------------------------------------------------------------------
    private class func layoutCols( parent: UIView, columns: Array<Dictionary<String,Any>>,
                                   border:Bool) -> String
    {
        let w = UIScreen.main.bounds.width
        var left:CGFloat = 0.0
        let top:CGFloat = 0.0
        let height = parent.frame.height
        
        for c in columns {
            let (_, lpv, lerr) = parseSize( c["lpad"], AVP.LPAD)
            if lerr != nil { return lerr! }
            let (_, rpv, rerr) = parseSize( c["rpad"], AVP.RPAD)
            if rerr != nil { return rerr! }
            let (wtype, wv, werr) = parseSize( c["width"])
            if werr != nil { return werr! }
            if wtype == "pct" {
                let width = wv / 100.0 * w
                let outer = UIView()
                parent.addSubview( outer)
                outer.frame = CGRect( x: left, y: top, width: width, height: height)
                let inner = UIView()
                outer.addSubview( inner)
                inner.frame = CGRect( x: lpv, y: 0, width: width - lpv - rpv, height: height)
                if border {
                    inner.layer.borderWidth = 1
                    inner.layer.borderColor = AVP.randCol().cgColor
                }
                left += width
            }
            else { return "ERROR: width type \(wtype) not implemented" }
        } // for
        return "ok"
    } // layoutCols()

    //-------------------------------------------------------------------------------------------------
    private class func layoutRows( parent: UIView, rows: Array<Dictionary<String,Any>>,
                                   border:Bool) -> String
    {
        return "ok"
    } // layoutRows()
    
    //-----------------------------------------------------------------------------------------------------------
    private class func parseSize(_ sizeStr:Any?, _ ddefault:String = "0pct") -> (type:String, val:CGFloat, err:String?) {
        let size = sizeStr != nil ? sizeStr as! String : ddefault
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
    
    // Return a random color
    //--------------------------------------------
    private class func randCol() -> UIColor {
        let red = Int.random(in: 25...230)
        let green = Int.random(in: 25...230)
        let blue = Int.random(in: 25...230)
        let res = AHU.RGB(red, green, blue)
        return res
    } // randCol()
    

} // class AHXViewPos

typealias AVP = AHXViewPos
