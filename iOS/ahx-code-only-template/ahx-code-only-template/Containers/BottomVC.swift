//
//  BottomVC.swift
//  ahx-code-only-template
//
//  Created by Andreas Hauenstein on 2021-04-21.
//

// Navigation bar at the bottom of the screen

import UIKit

//=================================
class BottomVC: UIViewController {
    var btnOne:UIButton!
    var btnTwo:UIButton!

    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
    } // viewDidLoad()
    
    //-------------------------------------------------
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        layout()
    } // viewWillAppear()
    
    //-------------------------
    override func layout() {
        // Layout root frame
        let v = self.view!
        AHL.width( v, AHC.w)
        AHL.height( v, AHC.bottom_nav_height)
        AHL.left( v, 0)
        AHL.bottom( v, AHC.bottom)
        
        // Layout view components
        let marg = 0.2 * AHC.w
        let btnw = (AHC.w - 3 * marg) / 2
        
        // Currency Button
        btnOne = UIButton( type: .custom,
                           primaryAction: UIAction() { _ in
                            if !self.btnOne.isSelected {
                                AHXMain.shared.topVC( "FirstVC")
                            }
                           })
        var btn = btnOne!
        var nimg = UIImage( named:"dollar_gray")!
        var simg = UIImage( named:"dollar_green")!
        btn.setImage( nimg, for: .normal)
        btn.setImage( simg, for: .selected)
        view.addSubview( btn)
        AHL.height( btn, view.frame.height)
        AHL.scaleWidth( btn, likeImage: nimg)
        AHL.left( btn, marg)
        AHL.top( btn, 0)
        AHL.border( btn)
        
        // Image Button
        btnTwo = UIButton( type: .custom,
                           primaryAction: UIAction() { _ in
                            if !self.btnTwo.isSelected {
                                AHXMain.shared.topVC( "SecondVC")
                            }
                           })
        btn = btnTwo!
        nimg = UIImage( named:"pic_gray")!
        simg = UIImage( named:"pic_black")!
        btn.setImage( nimg, for: .normal)
        btn.setImage( simg, for: .selected)
        view.addSubview( btn)
        AHL.height( btn, view.frame.height)
        AHL.scaleWidth( btn, likeImage: nimg)
        AHL.left( btn, btnw + 2 * marg )
        AHL.top( btn, 0)
        AHL.border( btn)
    } // layout()
    
    //-------------------------------------
    func selectButton( _ btnIdx:Int) {
        if btnIdx == 0 {
            btnOne.isSelected = true
            btnTwo.isSelected = false
        }
        else {
            btnOne.isSelected = false
            btnTwo.isSelected = true
        }
    } // selectButton()
} // class BottomVC
