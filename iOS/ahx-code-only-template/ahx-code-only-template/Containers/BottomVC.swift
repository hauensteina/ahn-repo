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
    var rubber = UIView()

    //-------------------------------
    override func viewDidLoad() {
        super.viewDidLoad()
        self.buttons = [UIButton]()
        // Currency Button
        self.buttons.append( UIButton( type: .custom,
                                       primaryAction: UIAction() { _ in
                                        AHXMain.shared.topVC( "FirstVC") { self.selectButton( 0) }
                                       }))
        // Image Button
        self.buttons.append( UIButton( type: .custom,
                                       primaryAction: UIAction() { _ in
                                        AHXMain.shared.topVC( "SecondVC") { self.selectButton( 1) }
                                       }))
        // T Button
        self.buttons.append( UIButton( type: .custom,
                                       primaryAction: UIAction() { _ in
                                        AHXMain.shared.topVC( "FontVC") { self.selectButton( 2) }
                                       }))
        // W Button
        self.buttons.append( UIButton( type: .custom,
                                       primaryAction: UIAction() { _ in
                                        AHXMain.shared.topVC( "WebVC") { self.selectButton( 3) }
                                       }))
    } // viewDidLoad()
    
    //-------------------------------------------------
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        layout()
        self.selectButton( 0)
    } // viewWillAppear()
    
    //-------------------------
    override func layout() {
        // Layout root frame
        let v = self.view!
        AHL.width( v, AHC.w)
        AHL.height( v, AHC.bottom_nav_height)
        AHL.left( v, 0)
        AHL.bottom( v, AHC.bottom)
                
        let btnwidth = min( self.view.frame.width * 0.20, self.view.frame.height)
        
        // Currency Button
        var btn = self.buttons[0]
        var nimg = UIImage( named:"dollar_gray")!
        var simg = UIImage( named:"dollar_green")!
        btn.setImage( nimg, for: .normal)
        btn.setImage( simg, for: .selected)
        AHL.width( btn, btnwidth)
        AHL.scaleHeight( btn, likeImage: nimg)
        AHL.border( btn)
        
        // Image Button
        btn = self.buttons[1]
        nimg = UIImage( named:"pic_gray")!
        simg = UIImage( named:"pic_blue")!
        btn.setImage( nimg, for: .normal)
        btn.setImage( simg, for: .selected)
        AHL.width( btn, btnwidth)
        AHL.scaleHeight( btn, likeImage: nimg)
        AHL.border( btn)
        
        // T Button
        btn = self.buttons[2]
        nimg = UIImage( named:"letter_t_gray")!
        simg = UIImage( named:"letter_t_red")!
        btn.setImage( nimg, for: .normal)
        btn.setImage( simg, for: .selected)
        AHL.width( btn, btnwidth)
        AHL.scaleHeight( btn, likeImage: nimg)
        AHL.border( btn)

        // W Button
        btn = self.buttons[3]
        nimg = UIImage( named:"letter_w_gray")!
        simg = UIImage( named:"letter_w_red")!
        btn.setImage( nimg, for: .normal)
        btn.setImage( simg, for: .selected)
        AHL.width( btn, btnwidth)
        AHL.scaleHeight( btn, likeImage: nimg)
        AHL.border( btn)

        AHL.hShare( container: self.view,
                    subviews: [rubber, buttons[0], rubber, buttons[1], rubber, buttons[2], rubber, buttons[3], rubber],
                    widths: [nil, buttons[0].frame.width, nil, buttons[1].frame.width, nil, buttons[2].frame.width, nil, buttons[3].frame.width, nil])

    } // layout()
    
    //-------------------------------------
    func selectButton( _ btnIdx:Int) {
        for btn in self.buttons {
            btn.isSelected = false
        }
        self.buttons[btnIdx].isSelected = true
    } // selectButton()
} // class BottomVC
