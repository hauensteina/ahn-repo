//
//  FontVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-06-14.
//
/*
 Test our font sizes smallFont(), mediumFont(), largeFont()
 */

import UIKit

// View Controller displaying some images in a tableview
//========================================================
class FontVC: AHXVC {
    let lorem = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    """
    var lbsmall:UILabel!
    var lbmedium:UILabel!
    var lblarge:UILabel!

    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        self.lbsmall = UILabel()
        self.lbmedium = UILabel()
        self.lblarge = UILabel()
    } // viewDidLoad()
    
    //-------------------------
    override func layout() {
        AHL.width( self.lbsmall, AHC.w - AHC.lmarg - AHC.rmarg)
        AHL.width( self.lbmedium, AHC.w - AHC.lmarg - AHC.rmarg)
        AHL.width( self.lblarge, AHC.w - AHC.lmarg - AHC.rmarg)
        AHL.vShare( container: self.view, subviews: [self.lbsmall, self.lbmedium, self.lblarge])
        AHL.border( self.lbsmall)
        AHL.border( self.lbmedium)
        AHL.border( self.lblarge)
        
        lbsmall.font = AHL.smallFont()
        lbmedium.font = AHL.mediumFont()
        lblarge.font = AHL.largeFont()
        
        lbsmall.numberOfLines = 0
        lbmedium.numberOfLines = 0
        lblarge.numberOfLines = 0

        lbsmall.text = lorem
        lbmedium.text = lorem
        lblarge.text = lorem
    } // layout()
    
} // class SecondVC
