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
    
    //-----------------
    func layout() {
        let inset = AHXMain.scene.win.safeAreaInsets
        let w = UIScreen.main.bounds.width
        let h = UIScreen.main.bounds.height - inset.top - inset.bottom
        let bottom = UIScreen.main.bounds.height - inset.bottom
        let top = inset.top
        
        // Layout root frame
        let v = self.view!
        AHXLayout.width( v, w)
        AHXLayout.height( v, 0.1 * h)
        AHXLayout.left( v, 0)
        AHXLayout.bottom( v, bottom)
        
        // Layout view components
        let marg = 0.2 * w
        let btnw = (w - 3 * marg) / 2
        
        // Button One
        btnOne = UIButton( type: .system,
                           primaryAction: UIAction(title: "One", handler: { _ in
                            // do nothing
                           }))
        var btn = btnOne!
        view.addSubview( btn)
        AHXLayout.width( btn, btnw)
        AHXLayout.height( btn, view.frame.height)
        AHXLayout.left( btn, marg)
        AHXLayout.top( btn, 0)
        AHXLayout.border( btn)
        
        // Button Two
        btnTwo = UIButton( type: .system,
                           primaryAction: UIAction(title: "Two", handler: { _ in
                            // do nothing
                           }))
        btn = btnTwo! 
        view.addSubview( btn)
        AHXLayout.width( btn, btnw)
        AHXLayout.height( btn, view.frame.height)
        AHXLayout.left( btn, btnw + 2 * marg )
        AHXLayout.top( btn, 0)
        AHXLayout.border( btn)
    } // layout()
} // class BottomVC
