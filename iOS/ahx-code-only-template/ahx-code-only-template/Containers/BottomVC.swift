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
        btnOne = UIButton( type: .system,
                           primaryAction: UIAction(title: "C", handler: { _ in
                            // do nothing
                           }))
        var btn = btnOne!
        view.addSubview( btn)
        AHL.width( btn, btnw)
        AHL.height( btn, view.frame.height)
        AHL.left( btn, marg)
        AHL.top( btn, 0)
        AHL.border( btn)
        
        // Image Button
        btnTwo = UIButton( type: .system,
                           primaryAction: UIAction(title: "I", handler: { _ in
                            // do nothing
                           }))
        btn = btnTwo!
        view.addSubview( btn)
        AHL.width( btn, btnw)
        AHL.height( btn, view.frame.height)
        AHL.left( btn, btnw + 2 * marg )
        AHL.top( btn, 0)
        AHL.border( btn)
    } // layout()
} // class BottomVC
