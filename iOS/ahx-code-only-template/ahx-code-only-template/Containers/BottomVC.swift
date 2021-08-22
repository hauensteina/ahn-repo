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
        // B Button
        self.buttons.append( UIButton( type: .custom,
                                       primaryAction: UIAction() { _ in
                                        AHXMain.shared.topVC( "BackEndVC") { self.selectButton( 4) }
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
                        
        // Currency Button
        var btn = self.buttons[0]
        var nimg = UIImage( named:"dollar_gray")!
        var simg = UIImage( named:"dollar_green")!
        btn.setImage( nimg, for: .normal)
        btn.setImage( simg, for: .selected)
        AHL.border( btn)
        
        // Image Button
        btn = self.buttons[1]
        nimg = UIImage( named:"pic_gray")!
        simg = UIImage( named:"pic_blue")!
        btn.setImage( nimg, for: .normal)
        btn.setImage( simg, for: .selected)
        AHL.border( btn)
        
        // T Button
        btn = self.buttons[2]
        nimg = UIImage( named:"letter_t_gray")!
        simg = UIImage( named:"letter_t_red")!
        btn.setImage( nimg, for: .normal)
        btn.setImage( simg, for: .selected)
        AHL.border( btn)

        // W Button
        btn = self.buttons[3]
        nimg = UIImage( named:"letter_w_gray")!
        simg = UIImage( named:"letter_w_red")!
        btn.setImage( nimg, for: .normal)
        btn.setImage( simg, for: .selected)
        AHL.border( btn)

        // B Button
        btn = self.buttons[4]
        nimg = UIImage( named:"letter_b_gray")!
        simg = UIImage( named:"letter_b_red")!
        btn.setImage( nimg, for: .normal)
        btn.setImage( simg, for: .selected)
        AHL.border( btn)

        var subviews = [rubber]
        var widths:[CGFloat?] = [nil]
        let btnwidth = AHC.w / CGFloat(self.buttons.count + 1)
        for b in self.buttons {
            AHL.width( b, btnwidth)
            AHL.scaleHeight( b, likeImage: btn.image( for: .selected)!)
            subviews.append( b)
            subviews.append( rubber)
            widths.append( b.frame.width)
            widths.append( nil)
        }
        AHL.hShare( container: self.view,
                    subviews: subviews,
                    widths: widths)

    } // layout()
    
    //-------------------------------------
    func selectButton( _ btnIdx:Int) {
        for btn in self.buttons {
            btn.isSelected = false
        }
        self.buttons[btnIdx].isSelected = true
    } // selectButton()
} // class BottomVC
