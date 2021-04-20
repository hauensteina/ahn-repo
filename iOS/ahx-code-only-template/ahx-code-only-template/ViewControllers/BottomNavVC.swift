//
//  BottomNavVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-19.
//

// A strip at the bottom of the screen, with buttons to press for navigation.

import UIKit

//==========================
class BottomNavView: UIView
{
    // Let touches through to the screen below, if appropriate
    //-----------------------------------------------------------------------------
    override func point( inside point: CGPoint, with event: UIEvent?) -> Bool {
        if BottomNavVC.shared.ivEur.frame.contains( point) { return true }
        if BottomNavVC.shared.ivCHF.frame.contains( point) { return true }
        // All others are passed down
        return false
    } // point()
} // class BottomNavView

//=========================================
class BottomNavVC: UIViewController {
    static var shared:BottomNavVC!
    var vwFrame:UIView!
    var ivEur:UIImageView!
    var ivCHF:UIImageView!

    //------------------------------
    override func loadView() {
        super.loadView()
        self.view = BottomNavView()
    } // loadView()

    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        BottomNavVC.shared = self
        vwFrame = UIView()
        self.view.addSubview( vwFrame)
        ivEur = UIImageView( image: UIImage( named: "eur"))
        ivCHF = UIImageView( image: UIImage( named: "chf"))
        vwFrame.addSubview( ivEur)
        vwFrame.addSubview( ivCHF)
        layout()
    } // viewDidLoad
    
    //-----------------
    func layout() {
        let inset = AHXMain.scene.win.safeAreaInsets
        let w = UIScreen.main.bounds.width
        let h = UIScreen.main.bounds.height - inset.top - inset.bottom
        let bottom = UIScreen.main.bounds.height - inset.bottom
        //let top = inset.top
        
        let barheight = 0.1 * h
        let imgHeight = barheight
        let imgWidth = imgHeight
        let marg = (w - 2 * imgWidth) / 3
        
        //let tt = self.safeAreaLayoutGuide.bottomAnchor
        var v = vwFrame!
        AHXLayout.width( v, w)
        AHXLayout.height( v, barheight)
        AHXLayout.center( v, w/2)
        AHXLayout.bottom( v, bottom)
        //AHXLayout.border( v, .green)
        
        v = ivEur!
        AHXLayout.width( v, imgWidth)
        AHXLayout.height( v, imgHeight)
        AHXLayout.left( v, marg)
        AHXLayout.middle( v, vwFrame.frame.height/2)

        v = ivCHF!
        AHXLayout.width( v, imgWidth)
        AHXLayout.height( v, imgHeight)
        AHXLayout.right( v, w-marg)
        AHXLayout.middle( v, vwFrame.frame.height/2)
    } // layout()
} // class BottomNavVC
