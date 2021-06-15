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
    var buttons:[UIButton]!
    var flex = UIView()

    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        self.buttons = [UIButton]()
        // Currency Button
        self.buttons.append( UIButton( type: .custom,
                                       primaryAction: UIAction() { _ in
                                        if self.buttons[0].isSelected { return }
                                        self.selectButton( 0)
                                        AHXMain.shared.topVC( "FirstVC")
                                       }))
        // Image Button
        self.buttons.append( UIButton( type: .custom,
                                       primaryAction: UIAction() { _ in
                                        if self.buttons[1].isSelected { return }
                                        self.selectButton( 1)
                                        AHXMain.shared.topVC( "SecondVC")
                                       }))
        // T Button
        self.buttons.append( UIButton( type: .custom,
                                       primaryAction: UIAction() { _ in
                                        if self.buttons[2].isSelected { return }
                                        self.selectButton( 2)
                                        AHXMain.shared.topVC( "FontVC")
                                       }))
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
        //let btnw = (AHC.w - 3 * marg) / 2
        
        // Currency Button
        var btn = self.buttons[0]
        var nimg = UIImage( named:"dollar_gray")!
        var simg = UIImage( named:"dollar_green")!
        btn.setImage( nimg, for: .normal)
        btn.setImage( simg, for: .selected)
        AHL.height( btn, view.frame.height)
        AHL.scaleWidth( btn, likeImage: nimg)
        //AHL.left( btn, marg)
        //AHL.top( btn, 0)
        AHL.border( btn)
        
        // Image Button
        btn = self.buttons[1]
        nimg = UIImage( named:"pic_gray")!
        simg = UIImage( named:"pic_black")!
        btn.setImage( nimg, for: .normal)
        btn.setImage( simg, for: .selected)
        AHL.height( btn, view.frame.height)
        AHL.scaleWidth( btn, likeImage: nimg)
        //AHL.left( btn, btnw + 2 * marg )
        //AHL.top( btn, 0)
        AHL.border( btn)
        
        // T Button
        btn = self.buttons[2]
        nimg = UIImage( named:"letter_t")!
        simg = UIImage( named:"letter_t")!
        btn.setImage( nimg, for: .normal)
        btn.setImage( simg, for: .selected)
        AHL.height( btn, view.frame.height)
        AHL.scaleWidth( btn, likeImage: nimg)
        //AHL.left( btn, btnw + 2 * marg )
        //AHL.top( btn, 0)
        AHL.border( btn)
        
        AHL.hShare( container: self.view,
                    subviews: [flex, buttons[0], buttons[1], buttons[2], flex],
                    widths: [nil, buttons[0].frame.width, buttons[1].frame.width, buttons[2].frame.width, nil],
                    leftmarg:0, rightmarg:0 )

    } // layout()
    
    //-------------------------------------
    func selectButton( _ btnIdx:Int) {
        for btn in self.buttons {
            btn.isSelected = false
        }
        self.buttons[btnIdx].isSelected = true
    } // selectButton()
} // class BottomVC
